import importlib
from typing import Any

from PIL import Image

from ironflow_exp.engine.tasks.base import TaskExecutionContext


class AlbumentationsTransformSchema:
    supported_transforms = frozenset({
        'Affine',
        'Blur',
        'HorizontalFlip',
        'RandomBrightnessContrast',
    })

    def transform_specs(self, *, context: TaskExecutionContext) -> list[dict[str, object]]:
        raw_specs = context.record.params.get('transforms')
        if raw_specs is None:
            return [{'name': 'HorizontalFlip', 'p': 1.0}]
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError('transforms must be a non-empty list')
        specs: list[dict[str, object]] = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                raise ValueError('each transform spec must be an object')
            name = str(spec.get('name') or '').strip()
            if not name:
                raise ValueError('each transform spec must include name')
            specs.append(dict(spec, name=name))

        return specs

    def validated_transform_kwargs(self, *, spec: dict[str, object]) -> dict[str, object]:
        name = str(spec['name'])
        if name not in self.supported_transforms:
            raise ValueError(f'unsupported albumentations transform: {name}')

        allowed_params = self._allowed_transform_params(name=name)
        unknown_params = sorted(set(spec) - {'name'} - allowed_params)
        if unknown_params:
            raise ValueError(f'{name} has unsupported params: {", ".join(unknown_params)}')

        kwargs: dict[str, object] = {}
        kwargs['p'] = self.probability(spec.get('p', 1.0), key=f'{name}.p')
        if name == 'Blur':
            if 'blur_limit' in spec:
                kwargs['blur_limit'] = self._blur_limit(spec['blur_limit'], key='Blur.blur_limit')
            return kwargs
        if name == 'HorizontalFlip':
            return kwargs
        if name == 'RandomBrightnessContrast':
            if 'brightness_limit' in spec:
                kwargs['brightness_limit'] = self._bounded_float_or_pair(
                    spec['brightness_limit'],
                    key='RandomBrightnessContrast.brightness_limit',
                    lower=-1.0,
                    upper=1.0,
                )
            if 'contrast_limit' in spec:
                kwargs['contrast_limit'] = self._bounded_float_or_pair(
                    spec['contrast_limit'],
                    key='RandomBrightnessContrast.contrast_limit',
                    lower=-1.0,
                    upper=1.0,
                )
            if 'brightness_by_max' in spec:
                kwargs['brightness_by_max'] = self.boolean(
                    spec['brightness_by_max'],
                    key='RandomBrightnessContrast.brightness_by_max',
                )
            if 'ensure_safe_range' in spec:
                kwargs['ensure_safe_range'] = self.boolean(
                    spec['ensure_safe_range'],
                    key='RandomBrightnessContrast.ensure_safe_range',
                )
            return kwargs
        if name == 'Affine':
            if 'scale' in spec:
                kwargs['scale'] = self._bounded_float_or_pair(
                    spec['scale'],
                    key='Affine.scale',
                    lower=0.0,
                    upper=4.0,
                )
            for key in ['translate_percent', 'rotate', 'shear']:
                if key in spec:
                    kwargs[key] = self._bounded_float_or_pair(
                        spec[key],
                        key=f'Affine.{key}',
                        lower=-360.0 if key in {'rotate', 'shear'} else -1.0,
                        upper=360.0 if key in {'rotate', 'shear'} else 1.0,
                    )
            for key in ['fit_output', 'keep_ratio']:
                if key in spec:
                    kwargs[key] = self.boolean(spec[key], key=f'Affine.{key}')
            if 'rotate_method' in spec:
                kwargs['rotate_method'] = self.one_of(
                    spec['rotate_method'],
                    key='Affine.rotate_method',
                    choices={'largest_box', 'ellipse'},
                )
            return kwargs

        raise ValueError(f'unsupported albumentations transform: {name}')

    def bbox_policy(self, *, context: TaskExecutionContext) -> dict[str, object]:
        raw_policy = context.record.params.get('bbox_policy')
        if raw_policy is None:
            raw_policy = {}
        if not isinstance(raw_policy, dict):
            raise ValueError('bbox_policy must be an object')

        allowed_keys = {
            'clip',
            'filter_invalid_bboxes',
            'min_area',
            'min_visibility',
            'min_width',
            'min_height',
            'max_accept_ratio',
        }
        unknown_keys = sorted(set(raw_policy) - allowed_keys)
        if unknown_keys:
            raise ValueError(f'bbox_policy has unsupported keys: {", ".join(unknown_keys)}')

        policy: dict[str, object] = {
            'clip': self.boolean(raw_policy.get('clip', True), key='bbox_policy.clip'),
            'filter_invalid_bboxes': self.boolean(
                raw_policy.get('filter_invalid_bboxes', True),
                key='bbox_policy.filter_invalid_bboxes',
            ),
            'min_area': self.non_negative_float(raw_policy.get('min_area', 0.0), key='bbox_policy.min_area'),
            'min_visibility': self.probability(raw_policy.get('min_visibility', 0.0), key='bbox_policy.min_visibility'),
            'min_width': self.non_negative_float(raw_policy.get('min_width', 0.0), key='bbox_policy.min_width'),
            'min_height': self.non_negative_float(raw_policy.get('min_height', 0.0), key='bbox_policy.min_height'),
        }
        if 'max_accept_ratio' in raw_policy:
            max_accept_ratio = raw_policy['max_accept_ratio']
            if max_accept_ratio is None:
                policy['max_accept_ratio'] = None
            else:
                value = self.float(value=max_accept_ratio, key='bbox_policy.max_accept_ratio')
                if value <= 0.0:
                    raise ValueError('bbox_policy.max_accept_ratio must be > 0')
                policy['max_accept_ratio'] = value

        return policy

    def base_seed(self, *, context: TaskExecutionContext) -> int | None:
        raw_seed = context.record.params.get('seed')
        if raw_seed is None:
            return None
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError('seed must be an integer')
        if not 0 <= raw_seed <= 2**32 - 1:
            raise ValueError('seed must be between 0 and 4294967295')

        return raw_seed

    def seed_for_record(self, *, context: TaskExecutionContext, index: int) -> int | None:
        base_seed = self.base_seed(context=context)
        if base_seed is None:
            return None

        return (base_seed + index) % 2**32

    def numpy_array(self, image: Image.Image) -> Any:
        numpy = importlib.import_module('numpy')

        return numpy.array(image)

    def _allowed_transform_params(self, *, name: str) -> set[str]:
        shared = {'p'}
        params_by_name = {
            'Affine': {
                'fit_output',
                'keep_ratio',
                'rotate',
                'rotate_method',
                'scale',
                'shear',
                'translate_percent',
            },
            'Blur': {'blur_limit'},
            'HorizontalFlip': set(),
            'RandomBrightnessContrast': {
                'brightness_by_max',
                'brightness_limit',
                'contrast_limit',
                'ensure_safe_range',
            },
        }

        return shared | params_by_name[name]

    def probability(self, value: object, *, key: str) -> float:
        probability = self.float(value=value, key=key)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f'{key} must be between 0 and 1')

        return probability

    def _bounded_float_or_pair(
        self,
        value: object,
        *,
        key: str,
        lower: float,
        upper: float,
    ) -> float | tuple[float, float]:
        if isinstance(value, list | tuple):
            if len(value) != 2:
                raise ValueError(f'{key} must be a number or a two-number list')
            pair = (
                self.float(value=value[0], key=f'{key}[0]'),
                self.float(value=value[1], key=f'{key}[1]'),
            )
            if pair[0] > pair[1]:
                raise ValueError(f'{key} lower bound must be <= upper bound')
            for item in pair:
                if not lower <= item <= upper:
                    raise ValueError(f'{key} values must be between {lower:g} and {upper:g}')
            return pair

        number = self.float(value=value, key=key)
        if not lower <= number <= upper:
            raise ValueError(f'{key} must be between {lower:g} and {upper:g}')

        return number

    def _blur_limit(self, value: object, *, key: str) -> int | tuple[int, int]:
        if isinstance(value, list | tuple):
            if len(value) != 2:
                raise ValueError(f'{key} must be an odd integer or a two-integer list')
            pair = (
                self._odd_blur_integer(value=value[0], key=f'{key}[0]'),
                self._odd_blur_integer(value=value[1], key=f'{key}[1]'),
            )
            if pair[0] > pair[1]:
                raise ValueError(f'{key} lower bound must be <= upper bound')
            return pair

        return self._odd_blur_integer(value=value, key=key)

    def _odd_blur_integer(self, *, value: object, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{key} must be an integer')
        if value < 3 or value % 2 == 0:
            raise ValueError(f'{key} must be an odd integer >= 3')

        return value

    def boolean(self, value: object, *, key: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be a boolean')

        return value

    def one_of(self, value: object, *, key: str, choices: set[str]) -> str:
        text = str(value)
        if text not in choices:
            raise ValueError(f'{key} must be one of: {", ".join(sorted(choices))}')

        return text

    def float(self, *, value: object, key: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f'{key} must be a number')

        return float(value)

    def non_negative_float(self, value: object, *, key: str) -> float:
        number = self.float(value=value, key=key)
        if number < 0.0:
            raise ValueError(f'{key} must be >= 0')

        return number
