from dataclasses import dataclass, field
from statistics import median
from typing import TypeAlias

# *************************
# External Library
#   - polars : ver 1.40.0
# *************************
import polars as pl

from ironflow_exp.domain import (
    ClassificationPredictionRecord,
    DetectionPredictionRecord,
    EmbeddingPredictionRecord,
    MetricRecord,
    ObjectRecord,
    SegmentationPredictionRecord,
)


LatencyPredictionRecord: TypeAlias = (
    DetectionPredictionRecord
    | ClassificationPredictionRecord
    | SegmentationPredictionRecord
    | EmbeddingPredictionRecord
)

CLASSIFICATION_DEFAULT_METRICS = {
    'accuracy',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'class_recall',
    'unknown_rate',
}

DETECTION_DEFAULT_METRICS = {
    'precision',
    'recall',
    'f1',
    'map50',
    'map50_95',
    'class_ap50',
    'num_predictions',
    'num_gt',
}

DETECTION_SUPPORTED_METRICS = DETECTION_DEFAULT_METRICS


@dataclass(frozen=True, slots=True)
class ClassificationMetricResult:
    metrics: list[MetricRecord] = field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DetectionMetricResult:
    metrics: list[MetricRecord] = field(default_factory=list)
    matched_predictions: list[DetectionPredictionRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    prediction: DetectionPredictionRecord
    reference: ObjectRecord | None
    iou: float | None
    is_true_positive: bool


class MetricsService:
    def calculate_detection(
        self,
        run_id: str,
        model_id: str,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
        metric_names: list[str] | None = None,
        iou_threshold: float = 0.5,
    ) -> DetectionMetricResult:
        requested_metrics = set(metric_names or DETECTION_DEFAULT_METRICS)
        unsupported_metrics = requested_metrics - DETECTION_SUPPORTED_METRICS
        if unsupported_metrics:
            raise ValueError(f'unsupported detection metrics: {sorted(unsupported_metrics)}')

        if not predictions and not references:
            return DetectionMetricResult()

        metrics: list[MetricRecord] = []
        matched_predictions: list[DetectionPredictionRecord] = []

        for split in self._detection_splits(predictions=predictions, references=references):
            split_predictions = [prediction for prediction in predictions if prediction.split == split]
            split_references = [reference for reference in references if reference.split == split and reference.is_gt]
            matches = self._match_detection_predictions(
                predictions=split_predictions,
                references=split_references,
                iou_threshold=iou_threshold,
            )
            matched_predictions.extend(self._matched_prediction_records(matches=matches))
            metrics.extend(
                self._calculate_detection_group_metrics(
                    run_id=run_id,
                    model_id=model_id,
                    split=split,
                    scope='overall',
                    predictions=split_predictions,
                    references=split_references,
                    matches=matches,
                    requested_metrics=requested_metrics,
                    iou_threshold=iou_threshold,
                ),
            )

            for class_id in self._detection_class_ids(predictions=split_predictions, references=split_references):
                class_predictions = [
                    prediction
                    for prediction in split_predictions
                    if prediction.class_id == class_id
                ]
                class_references = [
                    reference
                    for reference in split_references
                    if reference.class_id == class_id
                ]
                class_matches = [
                    match
                    for match in matches
                    if match.prediction.class_id == class_id
                ]
                metrics.extend(
                    self._calculate_detection_class_metrics(
                        run_id=run_id,
                        model_id=model_id,
                        split=split,
                        class_id=class_id,
                        predictions=class_predictions,
                        references=class_references,
                        matches=class_matches,
                        requested_metrics=requested_metrics,
                    ),
                )

        return DetectionMetricResult(
            metrics=metrics,
            matched_predictions=matched_predictions,
        )

    def calculate_classification(
        self,
        run_id: str,
        model_id: str,
        predictions: list[ClassificationPredictionRecord],
        metric_names: list[str] | None = None,
    ) -> ClassificationMetricResult:
        requested_metrics = set(metric_names or CLASSIFICATION_DEFAULT_METRICS)
        dataframe = self._classification_dataframe(predictions=predictions)

        if dataframe.is_empty():
            return ClassificationMetricResult()

        metrics: list[MetricRecord] = []
        confusion_matrix: dict[str, dict[str, int]] = {}

        for split, input_source in self._classification_groups(dataframe=dataframe):
            group = dataframe.filter(
                (pl.col('split') == split)
                & (pl.col('input_source') == input_source)
                & pl.col('true_class_id').is_not_null()
            )
            scope = f'input_source={input_source}'
            metrics.extend(
                self._calculate_classification_group_metrics(
                    run_id=run_id,
                    model_id=model_id,
                    split=split,
                    scope=scope,
                    dataframe=group,
                    requested_metrics=requested_metrics,
                ),
            )
            confusion_matrix[f'{split}:{input_source}'] = self._confusion_matrix(dataframe=group)

        return ClassificationMetricResult(
            metrics=metrics,
            confusion_matrix=confusion_matrix,
        )

    def calculate_latency(
        self,
        run_id: str,
        task: str,
        model_id: str,
        predictions: list[LatencyPredictionRecord],
        split: str = 'all',
        scope: str = 'overall',
    ) -> list[MetricRecord]:
        latency_values = [
            prediction.latency_ms
            for prediction in predictions
            if prediction.latency_ms is not None
        ]

        if not latency_values:
            return []

        sorted_values = sorted(latency_values)
        total_latency = sum(sorted_values)

        return [
            self._metric(
                run_id=run_id,
                task=task,
                model_id=model_id,
                name='avg_latency_ms',
                value=total_latency / len(sorted_values),
                split=split,
                scope=scope,
            ),
            self._metric(
                run_id=run_id,
                task=task,
                model_id=model_id,
                name='p50_latency_ms',
                value=median(sorted_values),
                split=split,
                scope=scope,
            ),
            self._metric(
                run_id=run_id,
                task=task,
                model_id=model_id,
                name='p95_latency_ms',
                value=self._percentile(values=sorted_values, percentile=0.95),
                split=split,
                scope=scope,
            ),
            self._metric(
                run_id=run_id,
                task=task,
                model_id=model_id,
                name='total_latency_ms',
                value=total_latency,
                split=split,
                scope=scope,
            ),
        ]

    def calculate_classification_latency(
        self,
        run_id: str,
        model_id: str,
        predictions: list[ClassificationPredictionRecord],
    ) -> list[MetricRecord]:
        dataframe = self._classification_dataframe(predictions=predictions)

        if dataframe.is_empty():
            return []

        metrics: list[MetricRecord] = []
        for split, input_source in self._classification_groups(dataframe=dataframe):
            group_predictions = [
                prediction
                for prediction in predictions
                if prediction.split == split and prediction.input_source == input_source
            ]
            metrics.extend(
                self.calculate_latency(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    predictions=group_predictions,
                    split=split,
                    scope=f'input_source={input_source}',
                ),
            )

        return metrics

    def _classification_dataframe(
        self,
        predictions: list[ClassificationPredictionRecord],
    ) -> pl.DataFrame:
        rows = [
            {
                'split': prediction.split,
                'input_source': prediction.input_source,
                'true_class_id': prediction.true_class_id,
                'true_class_name': prediction.true_class_name,
                'pred_class_id': prediction.pred_class_id,
                'pred_class_name': prediction.pred_class_name,
            }
            for prediction in predictions
        ]

        return pl.DataFrame(rows)

    def _classification_groups(self, dataframe: pl.DataFrame) -> list[tuple[str, str]]:
        group_rows = dataframe.select('split', 'input_source').unique().sort(['split', 'input_source']).to_dicts()

        return [
            (str(row['split']), str(row['input_source']))
            for row in group_rows
        ]

    def _calculate_classification_group_metrics(
        self,
        run_id: str,
        model_id: str,
        split: str,
        scope: str,
        dataframe: pl.DataFrame,
        requested_metrics: set[str],
    ) -> list[MetricRecord]:
        if dataframe.is_empty():
            return []

        rows = dataframe.to_dicts()
        class_ids = sorted(
            {
                int(row['true_class_id'])
                for row in rows
                if row['true_class_id'] is not None
            }
            | {
                int(row['pred_class_id'])
                for row in rows
                if row['pred_class_id'] is not None
            },
        )
        metrics: list[MetricRecord] = []

        if 'accuracy' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    name='accuracy',
                    value=self._accuracy(rows=rows),
                    split=split,
                    scope=scope,
                ),
            )

        precision_by_class = [self._precision(rows=rows, class_id=class_id) for class_id in class_ids]
        recall_by_class = [self._recall(rows=rows, class_id=class_id) for class_id in class_ids]
        f1_by_class = [
            self._f1(precision=precision, recall=recall)
            for precision, recall in zip(precision_by_class, recall_by_class)
        ]

        if 'macro_precision' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    name='macro_precision',
                    value=self._mean(values=precision_by_class),
                    split=split,
                    scope=scope,
                ),
            )

        if 'macro_recall' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    name='macro_recall',
                    value=self._mean(values=recall_by_class),
                    split=split,
                    scope=scope,
                ),
            )

        if 'macro_f1' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    name='macro_f1',
                    value=self._mean(values=f1_by_class),
                    split=split,
                    scope=scope,
                ),
            )

        if 'class_recall' in requested_metrics:
            metrics.extend(
                self._metric(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    name='class_recall',
                    value=self._recall(rows=rows, class_id=class_id),
                    split=split,
                    scope=f'{scope};class_id={class_id}',
                )
                for class_id in class_ids
            )

        if 'unknown_rate' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='classification',
                    model_id=model_id,
                    name='unknown_rate',
                    value=self._unknown_rate(rows=rows),
                    split=split,
                    scope=scope,
                ),
            )

        return metrics

    def _detection_splits(
        self,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
    ) -> list[str]:
        return sorted(
            {
                prediction.split
                for prediction in predictions
            }
            | {
                reference.split
                for reference in references
            },
        )

    def _detection_class_ids(
        self,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
    ) -> list[int]:
        return sorted(
            {
                int(prediction.class_id)
                for prediction in predictions
                if prediction.class_id is not None
            }
            | {
                int(reference.class_id)
                for reference in references
                if reference.class_id is not None
            },
        )

    def _match_detection_predictions(
        self,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
        iou_threshold: float,
    ) -> list[DetectionMatch]:
        sorted_predictions = sorted(
            predictions,
            key=lambda prediction: prediction.confidence or 0.0,
            reverse=True,
        )
        matched_reference_ids: set[str] = set()
        matches: list[DetectionMatch] = []

        for prediction in sorted_predictions:
            reference, iou = self._best_detection_reference(
                prediction=prediction,
                references=references,
                matched_reference_ids=matched_reference_ids,
            )
            is_true_positive = reference is not None and iou is not None and iou >= iou_threshold

            if is_true_positive:
                matched_reference_ids.add(reference.object_id)

            matches.append(
                DetectionMatch(
                    prediction=prediction,
                    reference=reference if is_true_positive else None,
                    iou=iou if is_true_positive else None,
                    is_true_positive=is_true_positive,
                ),
            )

        return matches

    def _best_detection_reference(
        self,
        prediction: DetectionPredictionRecord,
        references: list[ObjectRecord],
        matched_reference_ids: set[str],
    ) -> tuple[ObjectRecord | None, float | None]:
        if prediction.class_id is None:
            return None, None

        best_reference: ObjectRecord | None = None
        best_iou: float | None = None

        for reference in references:
            if reference.object_id in matched_reference_ids:
                continue
            if reference.sample_id != prediction.sample_id:
                continue
            if reference.class_id != prediction.class_id:
                continue

            prediction_bbox, reference_bbox = self._resolve_detection_bbox_pair(
                prediction=prediction,
                reference=reference,
            )
            iou = self._bbox_iou(prediction_bbox=prediction_bbox, reference_bbox=reference_bbox)
            if iou is None:
                continue
            if best_iou is None or iou > best_iou:
                best_reference = reference
                best_iou = iou

        return best_reference, best_iou

    def _calculate_detection_group_metrics(
        self,
        run_id: str,
        model_id: str,
        split: str,
        scope: str,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
        matches: list[DetectionMatch],
        requested_metrics: set[str],
        iou_threshold: float,
    ) -> list[MetricRecord]:
        true_positive_count = sum(1 for match in matches if match.is_true_positive)
        false_positive_count = len(predictions) - true_positive_count
        false_negative_count = len(references) - true_positive_count
        precision = self._safe_divide(numerator=true_positive_count, denominator=true_positive_count + false_positive_count)
        recall = self._safe_divide(numerator=true_positive_count, denominator=true_positive_count + false_negative_count)
        metrics: list[MetricRecord] = []

        if 'precision' in requested_metrics:
            metrics.append(self._metric(run_id=run_id, task='detection', model_id=model_id, name='precision', value=precision, split=split, scope=scope))
        if 'recall' in requested_metrics:
            metrics.append(self._metric(run_id=run_id, task='detection', model_id=model_id, name='recall', value=recall, split=split, scope=scope))
        if 'f1' in requested_metrics:
            metrics.append(self._metric(run_id=run_id, task='detection', model_id=model_id, name='f1', value=self._f1(precision=precision, recall=recall), split=split, scope=scope))
        if 'map50' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='detection',
                    model_id=model_id,
                    name='map50',
                    value=self._mean_average_precision(
                        predictions=predictions,
                        references=references,
                        iou_threshold=iou_threshold,
                    ),
                    split=split,
                    scope=scope,
                ),
            )
        if 'map50_95' in requested_metrics:
            metrics.append(
                self._metric(
                    run_id=run_id,
                    task='detection',
                    model_id=model_id,
                    name='map50_95',
                    value=self._mean_average_precision_range(
                        predictions=predictions,
                        references=references,
                    ),
                    split=split,
                    scope=scope,
                ),
            )
        if 'num_predictions' in requested_metrics:
            metrics.append(self._metric(run_id=run_id, task='detection', model_id=model_id, name='num_predictions', value=float(len(predictions)), split=split, scope=scope))
        if 'num_gt' in requested_metrics:
            metrics.append(self._metric(run_id=run_id, task='detection', model_id=model_id, name='num_gt', value=float(len(references)), split=split, scope=scope))

        return metrics

    def _calculate_detection_class_metrics(
        self,
        run_id: str,
        model_id: str,
        split: str,
        class_id: int,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
        matches: list[DetectionMatch],
        requested_metrics: set[str],
    ) -> list[MetricRecord]:
        if 'class_ap50' not in requested_metrics:
            return []

        return [
            self._metric(
                run_id=run_id,
                task='detection',
                model_id=model_id,
                name='class_ap50',
                value=self._average_precision(matches=matches, num_references=len(references)),
                split=split,
                scope=f'class_id={class_id}',
            ),
        ]

    def _matched_prediction_records(
        self,
        matches: list[DetectionMatch],
    ) -> list[DetectionPredictionRecord]:
        return [
            DetectionPredictionRecord(
                prediction_id=match.prediction.prediction_id,
                sample_id=match.prediction.sample_id,
                image_id=match.prediction.image_id,
                object_id=match.prediction.object_id,
                split=match.prediction.split,
                class_id=match.prediction.class_id,
                class_name=match.prediction.class_name,
                confidence=match.prediction.confidence,
                bbox_xyxy=match.prediction.bbox_xyxy,
                bbox_yolo=match.prediction.bbox_yolo,
                matched_object_id=match.reference.object_id if match.reference is not None else None,
                match_iou=match.iou,
                latency_ms=match.prediction.latency_ms,
                metadata=match.prediction.metadata,
            )
            for match in matches
        ]

    def _mean_average_precision(
        self,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
        iou_threshold: float,
    ) -> float:
        class_ids = self._detection_class_ids(predictions=predictions, references=references)
        if not class_ids:
            return 0.0

        average_precisions = []
        for class_id in class_ids:
            class_predictions = [
                prediction
                for prediction in predictions
                if prediction.class_id == class_id
            ]
            class_references = [
                reference
                for reference in references
                if reference.class_id == class_id
            ]
            class_matches = self._match_detection_predictions(
                predictions=class_predictions,
                references=class_references,
                iou_threshold=iou_threshold,
            )
            average_precisions.append(
                self._average_precision(matches=class_matches, num_references=len(class_references)),
            )

        return self._mean(values=average_precisions)

    def _mean_average_precision_range(
        self,
        predictions: list[DetectionPredictionRecord],
        references: list[ObjectRecord],
    ) -> float:
        thresholds = [threshold / 100 for threshold in range(50, 100, 5)]

        return self._mean(
            values=[
                self._mean_average_precision(
                    predictions=predictions,
                    references=references,
                    iou_threshold=threshold,
                )
                for threshold in thresholds
            ],
        )

    def _average_precision(
        self,
        matches: list[DetectionMatch],
        num_references: int,
    ) -> float:
        if num_references == 0:
            return 0.0

        true_positive_total = 0
        false_positive_total = 0
        precisions: list[float] = []
        recalls: list[float] = []

        for match in matches:
            if match.is_true_positive:
                true_positive_total += 1
            else:
                false_positive_total += 1

            precisions.append(
                self._safe_divide(
                    numerator=true_positive_total,
                    denominator=true_positive_total + false_positive_total,
                ),
            )
            recalls.append(true_positive_total / num_references)

        return self._area_under_precision_recall(precisions=precisions, recalls=recalls)

    def _area_under_precision_recall(
        self,
        precisions: list[float],
        recalls: list[float],
    ) -> float:
        if not precisions or not recalls:
            return 0.0

        mrec = [0.0, *recalls, 1.0]
        mpre = [0.0, *precisions, 0.0]

        for index in range(len(mpre) - 2, -1, -1):
            mpre[index] = max(mpre[index], mpre[index + 1])

        area = 0.0
        for index in range(1, len(mrec)):
            if mrec[index] != mrec[index - 1]:
                area += (mrec[index] - mrec[index - 1]) * mpre[index]

        return area

    def _bbox_iou(
        self,
        prediction_bbox: list[float] | None,
        reference_bbox: list[float] | None,
    ) -> float | None:
        if prediction_bbox is None or reference_bbox is None:
            return None
        if len(prediction_bbox) != 4 or len(reference_bbox) != 4:
            return None

        pred_x1, pred_y1, pred_x2, pred_y2 = prediction_bbox
        ref_x1, ref_y1, ref_x2, ref_y2 = reference_bbox
        inter_x1 = max(pred_x1, ref_x1)
        inter_y1 = max(pred_y1, ref_y1)
        inter_x2 = min(pred_x2, ref_x2)
        inter_y2 = min(pred_y2, ref_y2)
        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height
        prediction_area = max(0.0, pred_x2 - pred_x1) * max(0.0, pred_y2 - pred_y1)
        reference_area = max(0.0, ref_x2 - ref_x1) * max(0.0, ref_y2 - ref_y1)
        union_area = prediction_area + reference_area - intersection_area

        if union_area <= 0:
            return None

        return intersection_area / union_area

    def _resolve_detection_bbox_pair(
        self,
        prediction: DetectionPredictionRecord,
        reference: ObjectRecord,
    ) -> tuple[list[float] | None, list[float] | None]:
        if prediction.bbox_xyxy is not None and reference.bbox_xyxy is not None:
            return prediction.bbox_xyxy, reference.bbox_xyxy

        if prediction.bbox_yolo is not None and reference.bbox_yolo is not None:
            return (
                self._yolo_bbox_to_xyxy(bbox_yolo=prediction.bbox_yolo),
                self._yolo_bbox_to_xyxy(bbox_yolo=reference.bbox_yolo),
            )

        return None, None

    def _yolo_bbox_to_xyxy(
        self,
        bbox_yolo: list[float],
    ) -> list[float] | None:
        if len(bbox_yolo) != 4:
            return None

        x_center, y_center, width, height = bbox_yolo

        return [
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ]

    def _safe_divide(self, numerator: int | float, denominator: int | float) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _accuracy(self, rows: list[dict[str, object]]) -> float:
        if not rows:
            return 0.0

        correct_count = sum(1 for row in rows if row['true_class_id'] == row['pred_class_id'])

        return correct_count / len(rows)

    def _precision(self, rows: list[dict[str, object]], class_id: int) -> float:
        true_positive = sum(1 for row in rows if row['true_class_id'] == class_id and row['pred_class_id'] == class_id)
        false_positive = sum(1 for row in rows if row['true_class_id'] != class_id and row['pred_class_id'] == class_id)
        denominator = true_positive + false_positive

        if denominator == 0:
            return 0.0

        return true_positive / denominator

    def _recall(self, rows: list[dict[str, object]], class_id: int) -> float:
        true_positive = sum(1 for row in rows if row['true_class_id'] == class_id and row['pred_class_id'] == class_id)
        false_negative = sum(1 for row in rows if row['true_class_id'] == class_id and row['pred_class_id'] != class_id)
        denominator = true_positive + false_negative

        if denominator == 0:
            return 0.0

        return true_positive / denominator

    def _f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _unknown_rate(self, rows: list[dict[str, object]]) -> float:
        if not rows:
            return 0.0

        unknown_count = sum(1 for row in rows if row['pred_class_id'] is None)

        return unknown_count / len(rows)

    def _confusion_matrix(self, dataframe: pl.DataFrame) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {}

        for row in dataframe.to_dicts():
            true_key = str(row['true_class_name'] or row['true_class_id'])
            pred_key = str(row['pred_class_name'] or row['pred_class_id'] or 'unknown')
            matrix.setdefault(true_key, {})
            matrix[true_key][pred_key] = matrix[true_key].get(pred_key, 0) + 1

        return matrix

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0

        index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))

        return values[index]

    def _metric(
        self,
        run_id: str,
        task: str,
        model_id: str,
        name: str,
        value: float | None,
        split: str,
        scope: str,
    ) -> MetricRecord:
        return MetricRecord(
            run_id=run_id,
            task=task,
            model_id=model_id,
            metric_name=name,
            metric_value=value,
            split=split,
            scope=scope,
        )
