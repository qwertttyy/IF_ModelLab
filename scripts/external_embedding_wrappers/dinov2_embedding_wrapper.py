"""DINOv2 embedding wrapper entrypoint."""

from __future__ import annotations

from external_embedding_contract import run_contract_wrapper


if __name__ == '__main__':
    raise SystemExit(run_contract_wrapper(model_family='dinov2', adapter_key='dinov2_embedding'))
