import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ironflow_exp.engine.core import PredictionArtifactValidator
from ironflow_exp.engine.tasks.base import BaseTaskAdapter, TaskAdapterResult, TaskExecutionContext


@dataclass(slots=True)
class _TrackState:
    track_id: str
    class_id: str
    bbox_xyxy: list[float]
    last_frame_index: int


TRACKING_SKELETON_CONTRACTS: dict[str, dict[str, object]] = {
    'sort_tracker': {
        'default_model_id': 'sort',
        'family': 'tracking_by_detection',
        'input_contract': 'detection_sequence',
        'dependency_profile': 'tracking_motion',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json'),
        'output_contract': 'tracking_predictions.json',
    },
    'botsort_tracker': {
        'default_model_id': 'bot_sort',
        'family': 'tracking_by_detection_with_reid',
        'input_contract': 'detection_sequence_with_optional_appearance_embeddings',
        'dependency_profile': 'tracking_reid',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json', 'embeddings.npy optional'),
        'output_contract': 'tracking_predictions.json',
    },
    'ocsort_tracker': {
        'default_model_id': 'oc_sort',
        'family': 'tracking_by_detection',
        'input_contract': 'detection_sequence',
        'dependency_profile': 'tracking_motion',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json'),
        'output_contract': 'tracking_predictions.json',
    },
    'deepsort_tracker': {
        'default_model_id': 'deepsort',
        'family': 'tracking_by_detection_with_reid',
        'input_contract': 'detection_sequence_with_appearance_embeddings',
        'dependency_profile': 'tracking_reid',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json', 'embeddings.npy or reid checkpoint'),
        'output_contract': 'tracking_predictions.json',
    },
    'strongsort_tracker': {
        'default_model_id': 'strongsort',
        'family': 'tracking_by_detection_with_reid',
        'input_contract': 'detection_sequence_with_appearance_embeddings',
        'dependency_profile': 'tracking_reid',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json', 'embeddings.npy or reid checkpoint'),
        'output_contract': 'tracking_predictions.json',
    },
    'boosttrack_tracker': {
        'default_model_id': 'boosttrack_plus_plus',
        'family': 'tracking_by_detection_with_motion_boosting',
        'input_contract': 'detection_sequence',
        'dependency_profile': 'tracking_motion',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json'),
        'output_contract': 'tracking_predictions.json',
    },
    'tracktrack_tracker': {
        'default_model_id': 'tracktrack',
        'family': 'tracking_by_detection',
        'input_contract': 'detection_sequence',
        'dependency_profile': 'tracking_motion',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json'),
        'output_contract': 'tracking_predictions.json',
    },
    'pdsort_tracker': {
        'default_model_id': 'pd_sort',
        'family': 'tracking_by_detection',
        'input_contract': 'detection_sequence',
        'dependency_profile': 'tracking_motion',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json'),
        'output_contract': 'tracking_predictions.json',
    },
    'tdlp_tracker': {
        'default_model_id': 'tdlp',
        'family': 'tracking_by_detection',
        'input_contract': 'detection_sequence',
        'dependency_profile': 'tracking_motion',
        'required_artifacts': ('detection_predictions.json', 'frame_manifest.json'),
        'output_contract': 'tracking_predictions.json',
    },
}


class PlannedTrackingTaskAdapter(BaseTaskAdapter):
    adapter_key = 'planned_tracking'
    default_model_id = 'planned_tracking'

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        try:
            metadata = self._contract_metadata(context=context)
        except (KeyError, OSError, ValueError, TypeError) as error:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message=f'{self.adapter_key} tracking contract validation failed: {error}',
                metrics=[self._empty_metric_row()],
                metadata={
                    'adapter': self.adapter_key,
                    'model_id': context.record.model_id or self.default_model_id,
                    'failure_type': 'tracking_adapter_contract_invalid',
                    'error': str(error),
                },
            )

        return TaskAdapterResult(
            success=False,
            status='failed',
            message=f'{self.adapter_key} tracking adapter skeleton is not implemented yet',
            metrics=[self._empty_metric_row()],
            metadata={
                'adapter': self.adapter_key,
                'model_id': context.record.model_id or self.default_model_id,
                'failure_type': 'tracking_adapter_not_ready',
                **metadata,
            },
        )

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        contract = TRACKING_SKELETON_CONTRACTS.get(self.adapter_key)
        if contract is None:
            raise ValueError(f'unsupported planned tracking adapter: {self.adapter_key}')
        model_id = context.record.model_id or str(contract['default_model_id'])
        expected_model_id = str(contract['default_model_id'])
        if model_id != expected_model_id:
            raise ValueError(f'{self.adapter_key} expects model_id={expected_model_id}, got {model_id}')

        metadata: dict[str, object] = {
            'family': str(contract['family']),
            'input_contract': str(contract['input_contract']),
            'dependency_profile': str(contract['dependency_profile']),
            'required_artifacts': list(contract['required_artifacts']),
            'output_contract': str(contract['output_contract']),
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_skeleton_not_implemented',
            'detection_sequence_readiness': 'detection_predictions_required',
        }
        detection_prediction_path = self._optional_dependency_artifact_path(context=context, artifact_name='detection_predictions')
        if detection_prediction_path is not None:
            detection_payload = self._read_detection_payload(path=detection_prediction_path)
            records = detection_payload['records']
            metadata.update(
                {
                    'source_detection_predictions': str(detection_prediction_path),
                    'dataset_id': str(detection_payload['dataset_id']),
                    'detection_record_count': len(records),
                    'frame_count': self._frame_count(records=records),
                    'detection_sequence_readiness': 'detection_predictions_available',
                },
            )
        embeddings_path = self._optional_dependency_artifact_path(context=context, artifact_name='embeddings')
        metadata['appearance_embedding_readiness'] = (
            'embeddings_available'
            if embeddings_path is not None
            else 'embeddings_optional'
            if self.adapter_key == 'botsort_tracker'
            else 'not_required'
        )
        if embeddings_path is not None:
            metadata['source_embeddings'] = str(embeddings_path)

        return metadata

    def _optional_dependency_artifact_path(self, *, context: TaskExecutionContext, artifact_name: str) -> Path | None:
        try:
            return context.first_dependency_artifact_path(artifact_name)
        except KeyError:
            return None

    def _read_detection_payload(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('detection prediction payload must be an object')
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='detection')
        if not validation.is_valid:
            raise ValueError('invalid detection prediction artifact: ' + '; '.join(validation.error_messages()))
        records = payload.get('records')
        if not isinstance(records, list):
            raise ValueError('detection prediction payload records must be a list')

        return payload

    def _frame_count(self, *, records: list[Any]) -> int:
        frame_ids = {
            self._frame_id(record=record, fallback=index)
            for index, record in enumerate(records)
            if isinstance(record, dict)
        }

        return len(frame_ids)

    def _frame_id(self, *, record: dict[str, object], fallback: int) -> str:
        value = record.get('frame_id')
        if isinstance(value, str) and value:
            return value

        return f'frame_{fallback:08d}'

    def _empty_metric_row(self) -> dict[str, object]:
        return {
            'epoch': 1,
            'train_loss': '',
            'val_loss': '',
            'accuracy': '',
            'map50': '',
            'map50_95': '',
            'lr': '',
        }


class BotSortTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'botsort_tracker'
    default_model_id = 'bot_sort'


class SortTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'sort_tracker'
    default_model_id = 'sort'


class OCSortTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'ocsort_tracker'
    default_model_id = 'oc_sort'


class DeepSortTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'deepsort_tracker'
    default_model_id = 'deepsort'


class StrongSortTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'strongsort_tracker'
    default_model_id = 'strongsort'


class BoostTrackTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'boosttrack_tracker'
    default_model_id = 'boosttrack_plus_plus'


class TrackTrackTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'tracktrack_tracker'
    default_model_id = 'tracktrack'


class PdSortTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'pdsort_tracker'
    default_model_id = 'pd_sort'


class TdlpTrackingTaskAdapter(PlannedTrackingTaskAdapter):
    adapter_key = 'tdlp_tracker'
    default_model_id = 'tdlp'


class ByteTrackTaskAdapter(BaseTaskAdapter):
    adapter_key = 'bytetrack'
    default_model_id = 'bytetrack'

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        try:
            detection_prediction_path = self._detection_prediction_path(context=context)
            detection_payload = self._read_detection_payload(path=detection_prediction_path)
            records = self._tracking_records(
                detection_records=detection_payload['records'],
                iou_threshold=float(context.record.params.get('iou_threshold', 0.3)),
            )
        except (KeyError, OSError, ValueError, TypeError) as error:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message=f'bytetrack tracking adapter failed: {error}',
                metrics=[self._empty_metric_row()],
                metadata={
                    'adapter': self.adapter_key,
                    'failure_type': 'tracking_adapter_execution_failed',
                    'error': str(error),
                },
            )

        prediction_path = context.result_dir / 'predictions' / 'tracking_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'task': 'tracking',
            'model_id': context.record.model_id or self.default_model_id,
            'dataset_id': str(detection_payload['dataset_id']),
            'success': True,
            'records': records,
        }
        prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='tracking')
        if not validation.is_valid:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message='bytetrack tracking artifact validation failed: ' + '; '.join(validation.error_messages()),
                metrics=[self._empty_metric_row()],
                metadata={
                    'adapter': self.adapter_key,
                    'failure_type': 'tracking_adapter_output_invalid',
                },
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'bytetrack tracking completed: tracks={self._track_count(records)}, records={len(records)}',
            metrics=[self._empty_metric_row()],
            predictions=records,
            artifacts=[
                {
                    'name': 'tracking_predictions',
                    'path': 'predictions/tracking_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
            ],
            metadata={
                'adapter': self.adapter_key,
                'source_detection_predictions': str(detection_prediction_path),
                'track_count': self._track_count(records),
                'prediction_count': len(records),
            },
        )

    def _detection_prediction_path(self, context: TaskExecutionContext) -> Path:
        source_task_id = context.record.params.get('source_task_id')
        if isinstance(source_task_id, str) and source_task_id:
            return context.dependency_artifact_path(source_task_id, 'detection_predictions')

        return context.first_dependency_artifact_path('detection_predictions')

    def _read_detection_payload(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('detection prediction payload must be an object')
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='detection')
        if not validation.is_valid:
            raise ValueError('invalid detection prediction artifact: ' + '; '.join(validation.error_messages()))
        records = payload.get('records')
        if not isinstance(records, list):
            raise ValueError('detection prediction payload records must be a list')

        return payload

    def _tracking_records(
        self,
        *,
        detection_records: list[Any],
        iou_threshold: float,
    ) -> list[dict[str, object]]:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError('iou_threshold must be between 0.0 and 1.0')

        track_states: list[_TrackState] = []
        records: list[dict[str, object]] = []
        next_track_index = 1

        for detection_index, raw_detection in enumerate(self._sorted_detection_records(detection_records=detection_records)):
            if not isinstance(raw_detection, dict):
                raise ValueError('detection records must be objects')
            frame_index = self._frame_index(record=raw_detection, fallback=detection_index)
            class_id = str(raw_detection['class_id'])
            bbox = [float(value) for value in raw_detection['bbox_xyxy']]
            matched_track = self._best_track(
                track_states=track_states,
                class_id=class_id,
                bbox=bbox,
                frame_index=frame_index,
                iou_threshold=iou_threshold,
            )
            if matched_track is None:
                matched_track = _TrackState(
                    track_id=f'track_{next_track_index:08d}',
                    class_id=class_id,
                    bbox_xyxy=bbox,
                    last_frame_index=frame_index,
                )
                track_states.append(matched_track)
                next_track_index += 1
            else:
                matched_track.bbox_xyxy = bbox
                matched_track.last_frame_index = frame_index

            records.append(
                {
                    'frame_id': self._frame_id(record=raw_detection, frame_index=frame_index),
                    'frame_index': frame_index,
                    'image_id': str(raw_detection['image_id']),
                    'sample_id': str(raw_detection['sample_id']),
                    'track_id': matched_track.track_id,
                    'source_prediction_id': str(raw_detection['prediction_id']),
                    'class_id': class_id,
                    'score': float(raw_detection['score']),
                    'bbox_xyxy': bbox,
                    'image_width': float(raw_detection['image_width']),
                    'image_height': float(raw_detection['image_height']),
                },
            )

        return records

    def _sorted_detection_records(self, *, detection_records: list[Any]) -> list[Any]:
        return sorted(
            detection_records,
            key=lambda record: (
                self._frame_index(record=record, fallback=0) if isinstance(record, dict) else 0,
                str(record.get('image_id', '')) if isinstance(record, dict) else '',
                str(record.get('prediction_id', '')) if isinstance(record, dict) else '',
            ),
        )

    def _best_track(
        self,
        *,
        track_states: list[_TrackState],
        class_id: str,
        bbox: list[float],
        frame_index: int,
        iou_threshold: float,
    ) -> _TrackState | None:
        best_track: _TrackState | None = None
        best_iou = 0.0
        for track in track_states:
            if track.class_id != class_id:
                continue
            if track.last_frame_index > frame_index:
                continue
            iou = self._iou(track.bbox_xyxy, bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_track is None or best_iou < iou_threshold:
            return None

        return best_track

    def _iou(self, left: list[float], right: list[float]) -> float:
        left_x1, left_y1, left_x2, left_y2 = left
        right_x1, right_y1, right_x2, right_y2 = right
        intersection_x1 = max(left_x1, right_x1)
        intersection_y1 = max(left_y1, right_y1)
        intersection_x2 = min(left_x2, right_x2)
        intersection_y2 = min(left_y2, right_y2)
        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
        right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
        union_area = left_area + right_area - intersection_area
        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def _frame_index(self, *, record: dict[str, object], fallback: int) -> int:
        value = record.get('frame_index')
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)

        return fallback

    def _frame_id(self, *, record: dict[str, object], frame_index: int) -> str:
        value = record.get('frame_id')
        if isinstance(value, str) and value:
            return value

        return f'frame_{frame_index:08d}'

    def _track_count(self, records: list[dict[str, object]]) -> int:
        return len({
            str(record['track_id'])
            for record in records
        })

    def _empty_metric_row(self) -> dict[str, object]:
        return {
            'epoch': 1,
            'train_loss': '',
            'val_loss': '',
            'accuracy': '',
            'map50': '',
            'map50_95': '',
            'lr': '',
        }
