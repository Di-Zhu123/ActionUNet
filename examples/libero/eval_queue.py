#!/usr/bin/env python3
"""SQLite-backed, resumable LIBERO episode queue."""

# This worker runs on the pinned Python 3.8 LIBERO environment.
# ruff: noqa: FBT001, UP006, UP035, UP045

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import time
from typing import Any, Dict, Optional

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASKS_PER_SUITE = 10
TRIALS_PER_TASK = 50
EVAL_SEED = 7


def _job_specs(benchmark: str):
    if benchmark == "libero":
        yield from (
            (suite, task_id, trial_id)
            for suite in SUITES
            for task_id in range(TASKS_PER_SUITE)
            for trial_id in range(TRIALS_PER_TASK)
        )
        return
    if benchmark != "libero-plus":
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    classification_value = os.environ.get("LIBERO_EVAL_CATEGORY_CLASSIFICATION")
    if not classification_value:
        raise RuntimeError(
            "Set LIBERO_EVAL_CATEGORY_CLASSIFICATION to LIBERO-Plus's "
            "libero/libero/benchmark/task_classification.json"
        )
    classification_path = pathlib.Path(classification_value).resolve()
    with classification_path.open(encoding="utf-8") as handle:
        classification = json.load(handle)
    if set(classification) != set(SUITES):
        raise RuntimeError(
            f"LIBERO-Plus classification suite mismatch: {sorted(classification)}"
        )
    for suite in SUITES:
        entries = classification[suite]
        observed_ids = {int(entry["id"]) for entry in entries}
        expected_ids = set(range(1, len(entries) + 1))
        if observed_ids != expected_ids:
            raise RuntimeError(
                f"LIBERO-Plus classification IDs must be contiguous and one-based for {suite}"
            )
        for task_id in range(len(entries)):
            # LIBERO-Plus defines one rollout for each perturbation task.
            yield suite, task_id, 0


def connect(path: pathlib.Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=60.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def initialize(path: pathlib.Path, benchmark: str) -> Dict[str, int]:
    connection = connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suite TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'running', 'done', 'error')),
                worker_id TEXT,
                claimed_at REAL,
                completed_at REAL,
                duration_s REAL,
                success INTEGER CHECK(success IN (0, 1)),
                error TEXT,
                UNIQUE(suite, task_id, trial_id)
            );
            CREATE INDEX IF NOT EXISTS jobs_status_id ON jobs(status, id);
            CREATE INDEX IF NOT EXISTS jobs_worker_status ON jobs(worker_id, status);
            """
        )
        existing = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        if existing:
            if existing.get("benchmark") != benchmark or int(existing.get("seed", -1)) != EVAL_SEED:
                raise RuntimeError(
                    "Existing queue contract differs: "
                    f"benchmark={existing.get('benchmark')} seed={existing.get('seed')}"
                )
        else:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (("schema_version", "1"), ("benchmark", benchmark), ("seed", str(EVAL_SEED))),
                )
                connection.executemany(
                    "INSERT INTO jobs(suite, task_id, trial_id, seed) VALUES (?, ?, ?, ?)",
                    ((suite, task_id, trial_id, EVAL_SEED) for suite, task_id, trial_id in _job_specs(benchmark)),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        requeued = connection.execute(
            """
            UPDATE jobs
            SET status='pending', worker_id=NULL, claimed_at=NULL, error=NULL
            WHERE status='running'
            """
        ).rowcount
        counts = progress_connection(connection)
        counts["requeued"] = int(requeued)
        return counts
    finally:
        connection.close()


def claim(path: pathlib.Path, worker_id: str) -> Optional[Dict[str, Any]]:
    connection = connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        claimed_at = time.time()
        updated = connection.execute(
            """
            UPDATE jobs
            SET status='running', worker_id=?, claimed_at=?, completed_at=NULL,
                duration_s=NULL, success=NULL, error=NULL
            WHERE id=? AND status='pending'
            """,
            (worker_id, claimed_at, int(row["id"])),
        ).rowcount
        if updated != 1:
            connection.execute("ROLLBACK")
            return None
        connection.execute("COMMIT")
        result = dict(row)
        result.update(status="running", worker_id=worker_id, claimed_at=claimed_at)
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def finish(
    path: pathlib.Path,
    job_id: int,
    worker_id: str,
    success: bool,
    started_at: float,
) -> None:
    connection = connect(path)
    try:
        completed_at = time.time()
        changed = connection.execute(
            """
            UPDATE jobs
            SET status='done', completed_at=?, duration_s=?, success=?, error=NULL
            WHERE id=? AND status='running' AND worker_id=?
            """,
            (completed_at, completed_at - started_at, int(success), job_id, worker_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"Cannot finish unowned job id={job_id} worker={worker_id}")
    finally:
        connection.close()


def fail(path: pathlib.Path, job_id: int, worker_id: str, error: str, started_at: float) -> None:
    connection = connect(path)
    try:
        completed_at = time.time()
        changed = connection.execute(
            """
            UPDATE jobs
            SET status='error', completed_at=?, duration_s=?, success=0, error=?
            WHERE id=? AND status='running' AND worker_id=?
            """,
            (completed_at, completed_at - started_at, error[-8000:], job_id, worker_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"Cannot fail unowned job id={job_id} worker={worker_id}")
    finally:
        connection.close()


def requeue_worker(path: pathlib.Path, worker_id: str) -> int:
    connection = connect(path)
    try:
        return int(
            connection.execute(
                """
                UPDATE jobs
                SET status='pending', worker_id=NULL, claimed_at=NULL, error=NULL
                WHERE status='running' AND worker_id=?
                """,
                (worker_id,),
            ).rowcount
        )
    finally:
        connection.close()


def requeue_errors(path: pathlib.Path) -> int:
    connection = connect(path)
    try:
        return int(
            connection.execute(
                """
                UPDATE jobs
                SET status='pending', worker_id=NULL, claimed_at=NULL,
                    completed_at=NULL, duration_s=NULL, success=NULL, error=NULL
                WHERE status='error'
                """
            ).rowcount
        )
    finally:
        connection.close()


def progress_connection(connection: sqlite3.Connection) -> Dict[str, int]:
    counts = dict.fromkeys(("pending", "running", "done", "error"), 0)
    for row in connection.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"):
        counts[row["status"]] = int(row["count"])
    counts["total"] = sum(counts.values())
    row = connection.execute("SELECT COALESCE(SUM(success), 0) AS successes FROM jobs WHERE status='done'").fetchone()
    counts["success"] = int(row["successes"])
    return counts


def progress(path: pathlib.Path) -> Dict[str, int]:
    connection = connect(path)
    try:
        return progress_connection(connection)
    finally:
        connection.close()


def write_summary(path: pathlib.Path, output: pathlib.Path) -> None:
    connection = connect(path)
    try:
        rows = connection.execute(
            """
            SELECT suite,
                   COUNT(*) AS total,
                   SUM(status='pending') AS pending,
                   SUM(status='running') AS running,
                   SUM(status='done') AS done,
                   SUM(status='error') AS error,
                   COALESCE(SUM(CASE WHEN status='done' THEN success ELSE 0 END), 0) AS success
            FROM jobs GROUP BY suite ORDER BY MIN(id)
            """
        ).fetchall()
        lines = ["suite\ttotal\tpending\trunning\tdone\tsuccess\terror\tsuccess_rate"]
        totals = dict.fromkeys(("total", "pending", "running", "done", "success", "error"), 0)
        for row in rows:
            values = {key: int(row[key]) for key in totals}
            for key, value in values.items():
                totals[key] += value
            rate = values["success"] / values["done"] if values["done"] else 0.0
            lines.append(
                "\t".join(
                    [
                        str(row["suite"]),
                        *(str(values[key]) for key in ("total", "pending", "running", "done", "success", "error")),
                        f"{rate:.6f}",
                    ]
                )
            )
        overall_rate = totals["success"] / totals["done"] if totals["done"] else 0.0
        lines.append(
            "\t".join(
                [
                    "overall",
                    *(str(totals[key]) for key in ("total", "pending", "running", "done", "success", "error")),
                    f"{overall_rate:.6f}",
                ]
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--db", type=pathlib.Path, required=True)
    init_parser.add_argument("--benchmark", choices=("libero", "libero-plus"), required=True)

    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--db", type=pathlib.Path, required=True)
    progress_parser.add_argument("--shell", action="store_true")

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--db", type=pathlib.Path, required=True)
    summary_parser.add_argument("--output", type=pathlib.Path, required=True)

    requeue_parser = subparsers.add_parser("requeue-worker")
    requeue_parser.add_argument("--db", type=pathlib.Path, required=True)
    requeue_parser.add_argument("--worker-id", required=True)

    requeue_errors_parser = subparsers.add_parser("requeue-errors")
    requeue_errors_parser.add_argument("--db", type=pathlib.Path, required=True)

    claim_parser = subparsers.add_parser("claim")
    claim_parser.add_argument("--db", type=pathlib.Path, required=True)
    claim_parser.add_argument("--worker-id", required=True)

    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--db", type=pathlib.Path, required=True)
    finish_parser.add_argument("--job-id", type=int, required=True)
    finish_parser.add_argument("--worker-id", required=True)
    finish_parser.add_argument("--success", type=int, choices=(0, 1), required=True)
    finish_parser.add_argument("--started-at", type=float, required=True)

    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("--db", type=pathlib.Path, required=True)
    fail_parser.add_argument("--job-id", type=int, required=True)
    fail_parser.add_argument("--worker-id", required=True)
    fail_parser.add_argument("--error", required=True)
    fail_parser.add_argument("--started-at", type=float, required=True)

    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.db, args.benchmark)
        print("queue_initialized " + " ".join(f"{key}={value}" for key, value in result.items()))
    elif args.command == "progress":
        result = progress(args.db)
        if args.shell:
            print(" ".join(f"{key}={value}" for key, value in result.items()))
        else:
            print(json.dumps(result, sort_keys=True))
    elif args.command == "summary":
        write_summary(args.db, args.output)
    elif args.command == "requeue-worker":
        print(f"requeued={requeue_worker(args.db, args.worker_id)} worker={args.worker_id}")
    elif args.command == "requeue-errors":
        print(f"requeued={requeue_errors(args.db)} status=error")
    elif args.command == "claim":
        print(json.dumps(claim(args.db, args.worker_id), sort_keys=True))
    elif args.command == "finish":
        finish(args.db, args.job_id, args.worker_id, bool(args.success), args.started_at)
    elif args.command == "fail":
        fail(args.db, args.job_id, args.worker_id, args.error, args.started_at)


if __name__ == "__main__":
    main()
