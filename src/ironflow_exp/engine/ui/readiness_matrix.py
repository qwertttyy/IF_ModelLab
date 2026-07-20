from dataclasses import dataclass

from ironflow_exp.engine.ui.model_adapter_catalog import (
    AVAILABILITY_AVAILABLE_SMOKE,
    AVAILABILITY_PLANNED_REAL,
    READINESS_ADAPTER_READY,
    READINESS_PREPARED,
    READINESS_VALIDATED,
    RUNTIME_GPU,
    RUNTIME_LOCAL,
    RUNTIME_WSL,
    ModelAdapterCandidate,
    all_candidates,
    readiness_label_for_status,
)
from ironflow_exp.engine.ui.scenario_catalog import LOCAL_SCENARIOS, WSL_SCENARIOS, GuiScenario
from ironflow_exp.engine.ui.scenario_catalog import SCENARIO_PURPOSE_PREPARED_GATE


MATRIX_READY = 'ready'
MATRIX_NEEDS_PREFLIGHT = 'needs_preflight'
MATRIX_ADAPTER_READY = 'adapter_ready'
MATRIX_PREPARED = 'prepared'
MATRIX_NOT_TARGETED = 'not_targeted'
MATRIX_BLOCKED = 'blocked'

RUNTIME_LABELS = {
    RUNTIME_LOCAL: 'Local',
    RUNTIME_WSL: 'WSL',
    RUNTIME_GPU: 'GPU',
}
RUNTIME_ORDER = (RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU)


@dataclass(frozen=True, slots=True)
class ReadinessCell:
    status: str
    label: str
    detail: str = ''


@dataclass(frozen=True, slots=True)
class CandidateReadinessRow:
    candidate_id: str
    label: str
    task_type: str
    local: ReadinessCell
    wsl: ReadinessCell
    gpu: ReadinessCell


@dataclass(frozen=True, slots=True)
class ScenarioReadinessRow:
    label: str
    runtime_target: str
    flow_id: str
    status: str
    status_label: str
    next_gate: str


@dataclass(frozen=True, slots=True)
class RuntimeReadinessSummary:
    runtime_target: str
    ready_scenarios: int
    preflight_scenarios: int
    runnable_candidates: int
    adapter_ready_candidates: int
    prepared_candidates: int


def candidate_readiness_rows(
    candidates: tuple[ModelAdapterCandidate, ...] | None = None,
) -> tuple[CandidateReadinessRow, ...]:
    return tuple(
        CandidateReadinessRow(
            candidate_id=candidate.candidate_id,
            label=candidate.label,
            task_type=candidate.task_type,
            local=_candidate_runtime_cell(candidate=candidate, runtime_target=RUNTIME_LOCAL),
            wsl=_candidate_runtime_cell(candidate=candidate, runtime_target=RUNTIME_WSL),
            gpu=_candidate_runtime_cell(candidate=candidate, runtime_target=RUNTIME_GPU),
        )
        for candidate in (candidates or all_candidates())
    )


def scenario_readiness_rows() -> tuple[ScenarioReadinessRow, ...]:
    rows = [
        _scenario_row(scenario=scenario, runtime_target=RUNTIME_LOCAL)
        for scenario in LOCAL_SCENARIOS.values()
    ]
    rows.extend(
        _scenario_row(scenario=scenario, runtime_target=RUNTIME_WSL)
        for scenario in WSL_SCENARIOS.values()
    )

    return tuple(rows)


def runtime_readiness_summaries() -> tuple[RuntimeReadinessSummary, ...]:
    scenario_rows = scenario_readiness_rows()
    candidate_rows = candidate_readiness_rows()
    summaries: list[RuntimeReadinessSummary] = []
    for runtime_target in RUNTIME_ORDER:
        scenario_statuses = [
            row.status
            for row in scenario_rows
            if row.runtime_target == runtime_target
        ]
        candidate_cells = [
            _candidate_cell_for_runtime(row=row, runtime_target=runtime_target)
            for row in candidate_rows
        ]
        summaries.append(
            RuntimeReadinessSummary(
                runtime_target=runtime_target,
                ready_scenarios=scenario_statuses.count(MATRIX_READY),
                preflight_scenarios=scenario_statuses.count(MATRIX_NEEDS_PREFLIGHT),
                runnable_candidates=sum(1 for cell in candidate_cells if cell.status == MATRIX_READY),
                adapter_ready_candidates=sum(1 for cell in candidate_cells if cell.status == MATRIX_ADAPTER_READY),
                prepared_candidates=sum(1 for cell in candidate_cells if cell.status == MATRIX_PREPARED),
            ),
        )

    return tuple(summaries)


def gui_readiness_matrix_text() -> str:
    lines = ['Runtime readiness']
    for summary in runtime_readiness_summaries():
        lines.append(
            (
                f'{RUNTIME_LABELS[summary.runtime_target]}: '
                f'scenarios ready={summary.ready_scenarios}, '
                f'need preflight={summary.preflight_scenarios}, '
                f'runnable candidates={summary.runnable_candidates}, '
                f'adapter-ready candidates={summary.adapter_ready_candidates}, '
                f'prepared candidates={summary.prepared_candidates}'
            ),
        )

    lines.extend(
        [
            '',
            'Scenario gates',
            *[
                f'{RUNTIME_LABELS[row.runtime_target]} | {row.label} | {row.status_label} | {row.next_gate}'
                for row in scenario_readiness_rows()
            ],
            '',
            'GPU gate: live SSH -> GPU probe -> dependency probe -> short smoke',
        ],
    )

    return '\n'.join(lines)


def _candidate_runtime_cell(candidate: ModelAdapterCandidate, runtime_target: str) -> ReadinessCell:
    if runtime_target not in candidate.runtime_targets:
        return ReadinessCell(status=MATRIX_NOT_TARGETED, label='Not targeted')
    if candidate.availability == AVAILABILITY_AVAILABLE_SMOKE and candidate.readiness_status == READINESS_VALIDATED:
        return ReadinessCell(status=MATRIX_READY, label='Ready', detail='Runnable smoke')
    if candidate.availability == AVAILABILITY_PLANNED_REAL and candidate.readiness_status == READINESS_ADAPTER_READY:
        return ReadinessCell(status=MATRIX_ADAPTER_READY, label='Adapter-ready', detail='Not runnable from GUI yet')
    if candidate.availability == AVAILABILITY_PLANNED_REAL and candidate.readiness_status == READINESS_PREPARED:
        return ReadinessCell(status=MATRIX_PREPARED, label='Prepared', detail='Contract prepared; not runnable from GUI yet')

    return ReadinessCell(
        status=MATRIX_BLOCKED,
        label=readiness_label_for_status(candidate.readiness_status),
        detail='Blocked from runnable selectors',
    )


def _scenario_row(scenario: GuiScenario, runtime_target: str) -> ScenarioReadinessRow:
    if scenario.purpose == SCENARIO_PURPOSE_PREPARED_GATE:
        return ScenarioReadinessRow(
            label=scenario.label,
            runtime_target=runtime_target,
            flow_id=scenario.flow_id,
            status=MATRIX_PREPARED if runtime_target == RUNTIME_LOCAL else MATRIX_NOT_TARGETED,
            status_label='Prepared gate' if runtime_target == RUNTIME_LOCAL else 'Not targeted',
            next_gate='Run to inspect fail-closed metadata' if runtime_target == RUNTIME_LOCAL else 'No GUI scenario',
        )
    if runtime_target == RUNTIME_LOCAL:
        return ScenarioReadinessRow(
            label=scenario.label,
            runtime_target=runtime_target,
            flow_id=scenario.flow_id,
            status=MATRIX_READY,
            status_label='Ready',
            next_gate='Run',
        )
    if runtime_target == RUNTIME_WSL:
        return ScenarioReadinessRow(
            label=scenario.label,
            runtime_target=runtime_target,
            flow_id=scenario.flow_id,
            status=MATRIX_NEEDS_PREFLIGHT,
            status_label='Needs preflight',
            next_gate='Check -> Deps -> Execute',
        )

    return ScenarioReadinessRow(
        label=scenario.label,
        runtime_target=runtime_target,
        flow_id=scenario.flow_id,
        status=MATRIX_NOT_TARGETED,
        status_label='Not targeted',
        next_gate='No GUI scenario',
    )


def _candidate_cell_for_runtime(row: CandidateReadinessRow, runtime_target: str) -> ReadinessCell:
    if runtime_target == RUNTIME_LOCAL:
        return row.local
    if runtime_target == RUNTIME_WSL:
        return row.wsl

    return row.gpu
