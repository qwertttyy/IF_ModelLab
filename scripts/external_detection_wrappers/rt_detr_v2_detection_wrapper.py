"""RT-DETRv2 external detection wrapper entrypoint."""

from __future__ import annotations

from external_detection_contract import run_contract_wrapper


if __name__ == '__main__':
    raise SystemExit(run_contract_wrapper(model_family='rt_detr_v2', adapter_key='rt_detr_v2_detection'))
