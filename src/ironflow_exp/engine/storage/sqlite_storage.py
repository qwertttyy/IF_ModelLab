import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterable

from ironflow_exp.engine.domain import (
    EngineMetricRecord,
    ExperimentRecord,
    ExperimentStatus,
    ScheduleRecord,
    ServerRecord,
)


class SQLiteExperimentStorage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connection() as connection:
            connection.execute('PRAGMA foreign_keys = ON')
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    runner_type TEXT NOT NULL,
                    server_name TEXT,
                    config_path TEXT,
                    result_path TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    best_metric_name TEXT,
                    best_metric_value REAL,
                    error_type TEXT,
                    memo TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    macro_precision REAL,
                    macro_recall REAL,
                    macro_f1 REAL,
                    class_recall REAL,
                    class_ap50 REAL,
                    object_accuracy REAL,
                    map50 REAL,
                    map50_95 REAL,
                    num_predictions REAL,
                    num_gt REAL,
                    mask_count REAL,
                    mask_coverage REAL,
                    embedding_count REAL,
                    embedding_dim REAL,
                    retrieval_map REAL,
                    neighbor_purity REAL,
                    review_hit_rate REAL,
                    label_error_rate REAL,
                    latency_ms_per_image REAL,
                    p95_latency_ms REAL,
                    gpu_memory_mb REAL,
                    lr REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_experiment_epoch
                    ON metrics(experiment_id, epoch);

                CREATE TABLE IF NOT EXISTS schedules (
                    id TEXT PRIMARY KEY,
                    experiment_config_path TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    executed_at TEXT,
                    repeat_rule TEXT
                );

                CREATE TABLE IF NOT EXISTS servers (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    host TEXT,
                    port INTEGER,
                    username TEXT,
                    key_path TEXT,
                    remote_workspace TEXT,
                    last_checked_at TEXT,
                    last_status TEXT,
                    gpu_name TEXT,
                    total_vram TEXT
                );
                """,
            )
            self._ensure_metrics_columns(connection=connection)

    def _ensure_metrics_columns(self, *, connection: sqlite3.Connection) -> None:
        existing_columns = {
            str(row['name'])
            for row in connection.execute('PRAGMA table_info(metrics)').fetchall()
        }
        for column in (
            'precision',
            'recall',
            'macro_precision',
            'macro_recall',
            'macro_f1',
            'class_recall',
            'class_ap50',
            'object_accuracy',
            'num_predictions',
            'num_gt',
            'mask_count',
            'mask_coverage',
            'embedding_count',
            'embedding_dim',
            'retrieval_map',
            'neighbor_purity',
            'review_hit_rate',
            'label_error_rate',
            'latency_ms_per_image',
            'p95_latency_ms',
            'gpu_memory_mb',
        ):
            if column not in existing_columns:
                connection.execute(f'ALTER TABLE metrics ADD COLUMN {column} REAL')

    def save_experiment(self, record: ExperimentRecord) -> None:
        self.initialize()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id,
                    name,
                    task_type,
                    status,
                    runner_type,
                    server_name,
                    config_path,
                    result_path,
                    created_at,
                    started_at,
                    finished_at,
                    best_metric_name,
                    best_metric_value,
                    error_type,
                    memo,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    task_type = excluded.task_type,
                    status = excluded.status,
                    runner_type = excluded.runner_type,
                    server_name = excluded.server_name,
                    config_path = excluded.config_path,
                    result_path = excluded.result_path,
                    created_at = excluded.created_at,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    best_metric_name = excluded.best_metric_name,
                    best_metric_value = excluded.best_metric_value,
                    error_type = excluded.error_type,
                    memo = excluded.memo,
                    metadata_json = excluded.metadata_json
                """,
                self._experiment_values(record=record),
            )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        self.initialize()

        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM experiments WHERE id = ?',
                (experiment_id,),
            ).fetchone()

        if row is None:
            return None

        return self._experiment_from_row(row=row)

    def list_experiments(self) -> list[ExperimentRecord]:
        self.initialize()

        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM experiments ORDER BY created_at, id',
            ).fetchall()

        return [
            self._experiment_from_row(row=row)
            for row in rows
        ]

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        started_at: str | None = None,
        finished_at: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.initialize()

        with self._connection() as connection:
            connection.execute(
                """
                UPDATE experiments
                SET status = ?,
                    started_at = COALESCE(?, started_at),
                    finished_at = COALESCE(?, finished_at),
                    error_type = COALESCE(?, error_type)
                WHERE id = ?
                """,
                (
                    status.value,
                    started_at,
                    finished_at,
                    error_type,
                    experiment_id,
                ),
            )

    def save_metrics(self, records: Iterable[EngineMetricRecord]) -> None:
        metric_records = list(records)
        if not metric_records:
            return

        self.initialize()

        with self._connection() as connection:
            connection.executemany(
                """
                INSERT INTO metrics (
                    experiment_id,
                    epoch,
                    train_loss,
                    val_loss,
                    accuracy,
                    precision,
                    recall,
                    macro_precision,
                    macro_recall,
                    macro_f1,
                    class_recall,
                    class_ap50,
                    object_accuracy,
                    map50,
                    map50_95,
                    num_predictions,
                    num_gt,
                    mask_count,
                    mask_coverage,
                    embedding_count,
                    embedding_dim,
                    retrieval_map,
                    neighbor_purity,
                    review_hit_rate,
                    label_error_rate,
                    latency_ms_per_image,
                    p95_latency_ms,
                    gpu_memory_mb,
                    lr,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    self._metric_values(record=record)
                    for record in metric_records
                ],
            )

    def delete_metrics(self, experiment_id: str) -> None:
        self.initialize()

        with self._connection() as connection:
            connection.execute(
                'DELETE FROM metrics WHERE experiment_id = ?',
                (experiment_id,),
            )

    def list_metrics(self, experiment_id: str) -> list[EngineMetricRecord]:
        self.initialize()

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM metrics
                WHERE experiment_id = ?
                ORDER BY epoch IS NULL, epoch, id
                """,
                (experiment_id,),
            ).fetchall()

        return [
            self._metric_from_row(row=row)
            for row in rows
        ]

    def save_schedule(self, record: ScheduleRecord) -> None:
        self.initialize()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO schedules (
                    id,
                    experiment_config_path,
                    scheduled_at,
                    status,
                    created_at,
                    executed_at,
                    repeat_rule
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    experiment_config_path = excluded.experiment_config_path,
                    scheduled_at = excluded.scheduled_at,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    executed_at = excluded.executed_at,
                    repeat_rule = excluded.repeat_rule
                """,
                (
                    record.schedule_id,
                    record.experiment_config_path,
                    record.scheduled_at,
                    record.status,
                    record.created_at,
                    record.executed_at,
                    record.repeat_rule,
                ),
            )

    def get_schedule(self, schedule_id: str) -> ScheduleRecord | None:
        self.initialize()

        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM schedules WHERE id = ?',
                (schedule_id,),
            ).fetchone()

        if row is None:
            return None

        return self._schedule_from_row(row=row)

    def list_schedules(self) -> list[ScheduleRecord]:
        self.initialize()

        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM schedules ORDER BY scheduled_at, id',
            ).fetchall()

        return [
            self._schedule_from_row(row=row)
            for row in rows
        ]

    def update_schedule_status(
        self,
        schedule_id: str,
        status: str,
        executed_at: str | None = None,
    ) -> None:
        self.initialize()

        with self._connection() as connection:
            connection.execute(
                """
                UPDATE schedules
                SET status = ?,
                    executed_at = COALESCE(?, executed_at)
                WHERE id = ?
                """,
                (
                    status,
                    executed_at,
                    schedule_id,
                ),
            )

    def save_server(self, record: ServerRecord) -> None:
        self.initialize()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO servers (
                    name,
                    type,
                    host,
                    port,
                    username,
                    key_path,
                    remote_workspace,
                    last_checked_at,
                    last_status,
                    gpu_name,
                    total_vram
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    type = excluded.type,
                    host = excluded.host,
                    port = excluded.port,
                    username = excluded.username,
                    key_path = excluded.key_path,
                    remote_workspace = excluded.remote_workspace,
                    last_checked_at = excluded.last_checked_at,
                    last_status = excluded.last_status,
                    gpu_name = excluded.gpu_name,
                    total_vram = excluded.total_vram
                """,
                (
                    record.name,
                    record.server_type,
                    record.host,
                    record.port,
                    record.username,
                    record.key_path,
                    record.remote_workspace,
                    record.last_checked_at,
                    record.last_status,
                    record.gpu_name,
                    record.total_vram,
                ),
            )

    def get_server(self, name: str) -> ServerRecord | None:
        self.initialize()

        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM servers WHERE name = ?',
                (name,),
            ).fetchone()

        if row is None:
            return None

        return self._server_from_row(row=row)

    def list_servers(self) -> list[ServerRecord]:
        self.initialize()

        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM servers ORDER BY name',
            ).fetchall()

        return [
            self._server_from_row(row=row)
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _experiment_values(self, record: ExperimentRecord) -> tuple[object, ...]:
        return (
            record.experiment_id,
            record.name,
            record.task_type,
            record.status.value,
            record.runner_type,
            record.server_name,
            record.config_path,
            record.result_path,
            record.created_at,
            record.started_at,
            record.finished_at,
            record.best_metric_name,
            record.best_metric_value,
            record.error_type,
            record.memo,
            json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
        )

    def _metric_values(self, record: EngineMetricRecord) -> tuple[object, ...]:
        return (
            record.experiment_id,
            record.epoch,
            record.train_loss,
            record.val_loss,
            record.accuracy,
            record.precision,
            record.recall,
            record.macro_precision,
            record.macro_recall,
            record.macro_f1,
            record.class_recall,
            record.class_ap50,
            record.object_accuracy,
            record.map50,
            record.map50_95,
            record.num_predictions,
            record.num_gt,
            record.mask_count,
            record.mask_coverage,
            record.embedding_count,
            record.embedding_dim,
            record.retrieval_map,
            record.neighbor_purity,
            record.review_hit_rate,
            record.label_error_rate,
            record.latency_ms_per_image,
            record.p95_latency_ms,
            record.gpu_memory_mb,
            record.lr,
            record.created_at,
        )

    def _experiment_from_row(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=str(row['id']),
            name=str(row['name']),
            task_type=str(row['task_type']),
            status=ExperimentStatus(str(row['status'])),
            runner_type=str(row['runner_type']),
            server_name=self._optional_str(value=row['server_name']),
            config_path=self._optional_str(value=row['config_path']),
            result_path=self._optional_str(value=row['result_path']),
            created_at=str(row['created_at']),
            started_at=self._optional_str(value=row['started_at']),
            finished_at=self._optional_str(value=row['finished_at']),
            best_metric_name=self._optional_str(value=row['best_metric_name']),
            best_metric_value=self._optional_float(value=row['best_metric_value']),
            error_type=self._optional_str(value=row['error_type']),
            memo=str(row['memo']),
            metadata=self._metadata_from_json(value=str(row['metadata_json'])),
        )

    def _metric_from_row(self, row: sqlite3.Row) -> EngineMetricRecord:
        return EngineMetricRecord(
            experiment_id=str(row['experiment_id']),
            epoch=self._optional_int(value=row['epoch']),
            train_loss=self._optional_float(value=row['train_loss']),
            val_loss=self._optional_float(value=row['val_loss']),
            accuracy=self._optional_float(value=row['accuracy']),
            precision=self._optional_float(value=row['precision']),
            recall=self._optional_float(value=row['recall']),
            macro_precision=self._optional_float(value=row['macro_precision']),
            macro_recall=self._optional_float(value=row['macro_recall']),
            macro_f1=self._optional_float(value=row['macro_f1']),
            class_recall=self._optional_float(value=row['class_recall']),
            class_ap50=self._optional_float(value=row['class_ap50']),
            object_accuracy=self._optional_float(value=row['object_accuracy']),
            map50=self._optional_float(value=row['map50']),
            map50_95=self._optional_float(value=row['map50_95']),
            num_predictions=self._optional_float(value=row['num_predictions']),
            num_gt=self._optional_float(value=row['num_gt']),
            mask_count=self._optional_float(value=row['mask_count']),
            mask_coverage=self._optional_float(value=row['mask_coverage']),
            embedding_count=self._optional_float(value=row['embedding_count']),
            embedding_dim=self._optional_float(value=row['embedding_dim']),
            retrieval_map=self._optional_float(value=row['retrieval_map']),
            neighbor_purity=self._optional_float(value=row['neighbor_purity']),
            review_hit_rate=self._optional_float(value=row['review_hit_rate']),
            label_error_rate=self._optional_float(value=row['label_error_rate']),
            latency_ms_per_image=self._optional_float(value=row['latency_ms_per_image']),
            p95_latency_ms=self._optional_float(value=row['p95_latency_ms']),
            gpu_memory_mb=self._optional_float(value=row['gpu_memory_mb']),
            lr=self._optional_float(value=row['lr']),
            created_at=str(row['created_at']),
        )

    def _schedule_from_row(self, row: sqlite3.Row) -> ScheduleRecord:
        return ScheduleRecord(
            schedule_id=str(row['id']),
            experiment_config_path=str(row['experiment_config_path']),
            scheduled_at=str(row['scheduled_at']),
            status=str(row['status']),
            created_at=str(row['created_at']),
            executed_at=self._optional_str(value=row['executed_at']),
            repeat_rule=self._optional_str(value=row['repeat_rule']),
        )

    def _server_from_row(self, row: sqlite3.Row) -> ServerRecord:
        return ServerRecord(
            name=str(row['name']),
            server_type=str(row['type']),
            host=self._optional_str(value=row['host']),
            port=self._optional_int(value=row['port']),
            username=self._optional_str(value=row['username']),
            key_path=self._optional_str(value=row['key_path']),
            remote_workspace=self._optional_str(value=row['remote_workspace']),
            last_checked_at=self._optional_str(value=row['last_checked_at']),
            last_status=self._optional_str(value=row['last_status']),
            gpu_name=self._optional_str(value=row['gpu_name']),
            total_vram=self._optional_str(value=row['total_vram']),
        )

    def _metadata_from_json(self, value: str) -> dict[str, object]:
        loaded = json.loads(value or '{}')
        if not isinstance(loaded, dict):
            return {}

        return dict(loaded)

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None

        return int(value)

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None

        return float(value)
