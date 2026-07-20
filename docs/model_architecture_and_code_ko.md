# IronFlow 모델 아키텍처와 코드 연결 정리

이 문서는 현재 `top10_balanced` 실험 조합이 어떤 모델 구조로 구성되어 있고, 실제 코드에서 모델이 어디서 로드/생성/학습되는지 확인하기 위한 정리이다.

핵심은 다음과 같다.

```text
대부분의 조합:
original image -> detection model -> bbox -> crop adapter -> crop image -> classification model -> class prediction

10번 조합:
original image -> YOLO26 detection -> bbox/crop -> SAM2 segmentation -> OpenCLIP embedding
```

즉, IronFlow가 YOLO, EfficientNet, ConvNeXt, SAM2 같은 모델 레이어를 직접 구현하는 구조는 아니다. IronFlow는 config로 실험 조합을 정의하고, adapter가 Ultralytics/Torchvision/timm/SAM2/OpenCLIP 같은 외부 런타임을 호출한다. 우리 코드가 직접 담당하는 부분은 실험 orchestration, 데이터 경로 해석, crop 연결, wrapper 실행, metric/artifact/checkpoint 수집이다.

## Top10 조합 구조

| # | Config | Detection | 중간 처리 | Classification / 후단 | 실제 adapter |
|---|---|---|---|---|---|
| 01 | `01_baseline_fast_yolo11n_mobilenetv3_small.yaml` | YOLO11n | detection bbox -> crop | MobileNetV3 Small | `ultralytics_yolo` + `detection_to_classification_crop` + `torchvision_classifier` |
| 02 | `02_balanced_yolo11n_efficientnet_b0.yaml` | YOLO11n | detection bbox -> crop | EfficientNet-B0 | `ultralytics_yolo` + `detection_to_classification_crop` + `torchvision_classifier` |
| 03 | `03_classifier_accuracy_yolo11n_convnext_v2_tiny.yaml` | YOLO11n | detection bbox -> crop | ConvNeXt V2 Tiny | `ultralytics_yolo` + `detection_to_classification_crop` + `timm_classifier` |
| 04 | `04_attention_yolo12n_efficientnet_b0.yaml` | YOLO12n | detection bbox -> crop | EfficientNet-B0 | `ultralytics_yolo` + `detection_to_classification_crop` + `torchvision_classifier` |
| 05 | `05_yolo26_only_detection_classification.yaml` | YOLO26n | detection bbox -> crop | YOLO26n-cls | `ultralytics_yolo` + `detection_to_classification_crop` + `ultralytics_yolo_classifier` |
| 06 | `06_detr_precision_rf_detr_convnext_v2_tiny.yaml` | RF-DETR | detection bbox -> crop | ConvNeXt V2 Tiny | `rf_detr_detection` + `detection_to_classification_crop` + `timm_classifier` |
| 07 | `07_detr_speed_d_fine_efficientnet_v2_s.yaml` | D-FINE | detection bbox -> crop | EfficientNetV2-S | `d_fine_detection` + `detection_to_classification_crop` + `torchvision_classifier` |
| 08 | `08_rtdetr_global_context_efficientnet_b0.yaml` | RT-DETR | detection bbox -> crop | EfficientNet-B0 | `rt_detr_detection` + `detection_to_classification_crop` + `torchvision_classifier` |
| 09 | `09_next_yolo_family_yolo26n_efficientnet_b0.yaml` | YOLO26n | detection bbox -> crop | EfficientNet-B0 | `ultralytics_yolo` + `detection_to_classification_crop` + `torchvision_classifier` |
| 10 | `10_embedding_review_yolo26_sam2_openclip.yaml` | YOLO26n | detection bbox/crop | SAM2 segmentation + OpenCLIP embedding | `ultralytics_yolo` + `detection_to_classification_crop` + `sam_promptable_segmentation` + `clip_embedding` |

## 공통 실행 흐름

config 파일은 `models.detection`, `models.classification`, `models.segmentation`, `models.embedding` 섹션으로 각 모델의 adapter와 model id를 지정한다.

실제 config 로딩 코드는 다음 위치에 있다.

```python
# src/ironflow_exp/configs/config_loader.py
def _build_models(self, data: dict[str, Any]) -> ModelGroupConfig:
    return ModelGroupConfig(
        detection=self._build_model(data=data.get('detection', {})),
        classification=self._build_model(data=data.get('classification', {})),
        segmentation=self._build_model(data=data.get('segmentation', {})),
        embedding=self._build_model(data=data.get('embedding', {})),
    )

def _build_model(self, data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        enabled=self._bool(value=data.get('enabled', False)),
        model_id=self._optional_str(value=data.get('model_id')),
        adapter=self._optional_str(value=data.get('adapter')),
        checkpoint=self._optional_str(value=data.get('checkpoint')),
        pretrained=self._bool(value=data.get('pretrained', True)),
        output_dim=self._optional_int(value=data.get('output_dim')),
        train=self._build_train(data=data.get('train', {})),
        predict=self._build_predict(data=data.get('predict', {})),
    )
```

즉, YAML의 `adapter` 값이 실제 실행 클래스를 고르는 열쇠이고, `model_id`는 해당 adapter가 외부 라이브러리에 넘길 모델 이름 또는 checkpoint 기준이 된다.

## YOLO detection 계열

대상 조합:

- 01, 02, 03: `yolo11n`
- 04: `yolo12n`
- 05, 09, 10: `yolo26n`

구조적으로는 다음 역할을 한다.

```text
original image -> YOLO detector -> bbox / object score / object class
```

IronFlow 내부에서는 YOLO 네트워크 레이어를 직접 만들지 않는다. `UltralyticsYoloTaskAdapter`가 Ultralytics의 `YOLO` 클래스를 import하고, config에서 온 모델 reference로 객체를 만든 뒤 `model.train(...)`을 호출한다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
def _load_yolo_runtime(self, context: TaskExecutionContext) -> type[Any]:
    ...
    from ultralytics import YOLO
    ...
    return YOLO

def _run_train(...):
    ...
    yolo_class = self._load_yolo_runtime(context=context)
    model = yolo_class(UltralyticsYoloDetectionAdapter(model_config=model_config)._model_reference())
    train_result = model.train(**self._train_kwargs(context=context, model_config=model_config, data_yaml=data_yaml))
```

학습 파라미터는 `epochs`, `image_size`, `batch_size`, `workers`, `patience`, `seed`, `learning_rate`, `device` 등을 Ultralytics 인자로 변환한다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
for param_name, yolo_name in [
    ('epochs', 'epochs'),
    ('image_size', 'imgsz'),
    ('batch_size', 'batch'),
    ('workers', 'workers'),
    ('patience', 'patience'),
    ('seed', 'seed'),
]:
    value = self._optional_int(params.get(param_name))
    if value is not None:
        kwargs[yolo_name] = value

learning_rate = self._optional_float(params.get('learning_rate'))
if learning_rate is not None:
    kwargs['lr0'] = learning_rate
```

## Crop adapter

대상 조합:

- 01~10 모두 detection 결과를 후단 입력으로 넘길 때 사용한다.

역할은 모델이 아니라 연결 단계이다.

```text
detection_predictions.json -> bbox 좌표 기준 crop 생성 -> classifier/segmentation/embedding 입력
```

따라서 `detection_to_classification_crop`은 학습 가능한 neural network가 아니라, detection 결과와 원본 이미지를 이용해 후단 모델 입력 이미지를 만드는 preprocessing adapter이다.

## Torchvision classifier 계열

대상 조합:

- 01: MobileNetV3 Small
- 02, 04, 08, 09: EfficientNet-B0
- 07: EfficientNetV2-S

구조적으로는 다음 역할을 한다.

```text
crop image -> Torchvision backbone -> replaced classifier head -> class prediction
```

IronFlow는 Torchvision의 pretrained model을 가져온 뒤, 마지막 classification head를 우리 class 수에 맞게 교체한다. 직접 MobileNet/EfficientNet block을 구현하지 않는다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
def _replace_classifier_head(self, *, model: object, class_count: int, torch: object) -> None:
    linear = torch.nn.Linear
    classifier = getattr(model, 'classifier', None)
    if classifier is not None and hasattr(classifier, '__len__') and len(classifier) > 0:
        last_layer = classifier[-1]
        in_features = getattr(last_layer, 'in_features', None)
        if isinstance(in_features, int):
            classifier[-1] = linear(in_features, class_count)
            return
    fc = getattr(model, 'fc', None)
    in_features = getattr(fc, 'in_features', None)
    if isinstance(in_features, int):
        model.fc = linear(in_features, class_count)
        return
    head = getattr(model, 'head', None)
    in_features = getattr(head, 'in_features', None)
    if isinstance(in_features, int):
        model.head = linear(in_features, class_count)
        return
```

학습 루프는 일반적인 supervised classification 구조이다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
def _classification_train_epoch(...):
    model.train()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## timm ConvNeXt V2 classifier

대상 조합:

- 03: YOLO11n + ConvNeXt V2 Tiny
- 06: RF-DETR + ConvNeXt V2 Tiny

구조적으로는 다음 역할을 한다.

```text
crop image -> timm ConvNeXt V2 Tiny -> reset classifier head -> class prediction
```

`timm_classifier`는 Torchvision classifier adapter를 상속하되, 모델 생성만 timm으로 바꾼다. class 수에 맞는 head 교체는 timm의 `reset_classifier`를 우선 사용한다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
class TimmClassifierTaskAdapter(TorchvisionClassifierTaskAdapter):
    adapter_key = 'timm_classifier'
    default_model_id = 'convnext_v2_tiny'

    def _build_model(self, *, model_config: ModelConfig) -> object:
        adapter = TimmClassificationAdapter(model_config=self._checkpoint_safe_build_config(model_config=model_config))
        timm = adapter._load_timm()
        reference = adapter._model_reference_contract()

        return timm.create_model(reference.timm_model_name, pretrained=reference.pretrained)

    def _replace_classifier_head(self, *, model: object, class_count: int, torch: object) -> None:
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=class_count)
            return

        super()._replace_classifier_head(model=model, class_count=class_count, torch=torch)
```

## YOLO classifier 계열

대상 조합:

- 05: YOLO26n detector + YOLO26n-cls classifier

이 조합은 “detector 하나로 모든 걸 끝내는 구조”가 아니다. 현재 config 기준으로는 detection과 classification이 분리되어 있다.

```text
original image -> YOLO26n detection -> bbox crop -> YOLO26n-cls classification -> class prediction
```

즉, 앞단 detection은 `yolo26n`, 후단 classification은 `yolo26n-cls`이다. 둘 다 Ultralytics 런타임을 쓰지만 task가 다르다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
class UltralyticsYoloClassifierTaskAdapter(TorchvisionClassifierTaskAdapter):
    adapter_key = 'ultralytics_yolo_classifier'
    default_model_id = 'yolo26n-cls'
    ...
    def _contract_metadata(...):
        ...
        return {
            ...
            'ultralytics_task': 'classify',
            ...
        }
```

## RF-DETR / D-FINE / RT-DETR detection 계열

대상 조합:

- 06: RF-DETR + ConvNeXt V2 Tiny
- 07: D-FINE + EfficientNetV2-S
- 08: RT-DETR + EfficientNet-B0

구조적으로는 detection model이 원본 이미지에서 bbox를 만들고, 뒤는 다른 조합과 동일하게 crop classification으로 이어진다.

```text
original image -> DETR-family detector -> bbox -> crop -> classifier
```

IronFlow 코드에서 이 계열은 `PlannedDetectionTaskAdapter` 기반 adapter로 연결되어 있다.

```python
# src/ironflow_exp/engine/tasks/model_task_adapters.py
class DFineDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'd_fine_detection'
    default_model_id = 'd_fine'

class RfDetrDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'rf_detr_detection'
    default_model_id = 'rf_detr'

class RtDetrDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'rt_detr_detection'
    default_model_id = 'rt_detr'
```

이 계열의 세부 neural network 구현은 IronFlow 안에 직접 구현되어 있지 않다. native wrapper 또는 외부 detector 런타임의 contract를 통해 실행되며, IronFlow는 wrapper 실행 요청, 입력 manifest, 출력 prediction artifact, metric 수집을 관리한다.

## SAM2 segmentation

대상 조합:

- 10: YOLO26n + SAM2 + OpenCLIP

SAM2는 detector bbox를 prompt로 받아 mask를 만든다.

```text
image + bbox prompt -> SAM2 predictor -> mask polygon
```

실제 wrapper 코드는 SAM2 predictor를 로드하고, bbox를 `predict(box=...)`에 넘긴다.

```python
# scripts/external_segmentation_wrappers/external_segmentation_contract.py
def _load_sam2_predictor(*, request: dict[str, Any]) -> Any:
    predictor_module = optional_import('sam2.sam2_image_predictor')
    predictor_class = predictor_module.SAM2ImagePredictor
    hf_model_id = request_param(request, 'sam2_hf_model_id') or request_param(request, 'sam2_model_id')
    if hf_model_id:
        return predictor_class.from_pretrained(str(hf_model_id))
    ...

def _run_native_sam2(...):
    ...
    predictor.set_image(np.asarray(image))
    masks, scores, _logits = predictor.predict(
        box=np.asarray(bbox, dtype=np.float32),
        multimask_output=False,
    )
```

## OpenCLIP embedding

대상 조합:

- 10: YOLO26n + SAM2 + OpenCLIP

OpenCLIP은 closed-set classifier가 아니라 image embedding을 만든다.

```text
crop/masked image -> OpenCLIP image encoder -> normalized embedding vector
```

실제 wrapper는 `open_clip.create_model_and_transforms(...)`로 모델과 preprocess를 만들고, `model.encode_image(...)`로 vector를 추출한다.

```python
# scripts/external_embedding_wrappers/external_embedding_contract.py
def _run_native_openclip(...):
    ...
    open_clip = optional_import('open_clip')
    model, _unused, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    ...
    image = preprocess(image_module.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    vector = model.encode_image(image)
    vector = vector / vector.norm(dim=-1, keepdim=True)
```

## Pretrained weight 준비 코드

pretrained weight는 실행 중 암묵적으로 다운로드되는 것을 기본적으로 막고, 명시적으로 준비된 checkpoint를 쓰는 방향으로 설계되어 있다. 관련 스크립트는 다음 위치에 있다.

```text
scripts/download_pretrained_weights.py
```

이 스크립트는 Ultralytics, Torchvision, timm 계열 weight를 `models/checkpoints/pretrained` 아래에 준비한다.

```python
# scripts/download_pretrained_weights.py
for model_id in yolo_ids:
    results.append(_download_yolo(model_id=model_id, output_root=output_root))
for model_id in YOLO_CLASSIFIER_MODEL_IDS:
    results.append(_download_yolo(model_id=model_id, output_root=output_root))
...
for model_id in TORCHVISION_MODEL_IDS:
    results.append(_download_torchvision(model_id=model_id, output_root=output_root))
...
for model_id in TIMM_MODEL_IDS:
    results.append(_download_timm(model_id=model_id, output_root=output_root))
```

## 모델 추가 시 실제로 건드리는 층

새 모델을 추가할 때는 아래 세 층을 구분해서 봐야 한다.

1. 실험 조합 YAML
   - `configs/engine/top10_balanced/*.yaml`
   - 어떤 detector/classifier/segmentation/embedding 조합을 쓸지 정의한다.

2. adapter 코드
   - `src/ironflow_exp/engine/tasks/model_task_adapters.py`
   - 기존 adapter가 지원하는 모델이면 YAML만 추가해도 된다.
   - 완전히 새 라이브러리나 새 task 방식이면 adapter 또는 wrapper 추가가 필요하다.

3. 외부 wrapper / pretrained 준비
   - `scripts/external_*_wrappers/*`
   - `scripts/download_pretrained_weights.py`
   - GPU 서버에서 실행 가능한 dependency, checkpoint, 실행 contract를 맞춘다.

따라서 “모델 조합 추가”와 “새 모델 라이브러리 지원 추가”는 난이도가 다르다. 이미 지원되는 adapter 범위 안에서 model id만 바꾸는 것은 비교적 가볍고, 새 런타임을 붙이는 경우는 adapter/wrapper/dependency/checkpoint 정책까지 같이 봐야 한다.
