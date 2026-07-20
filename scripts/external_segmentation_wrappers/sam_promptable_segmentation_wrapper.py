"""SAM-family promptable segmentation wrapper entrypoint."""

from __future__ import annotations

from external_segmentation_contract import run_contract_wrapper


if __name__ == '__main__':
    raise SystemExit(run_contract_wrapper(model_family='sam_promptable', adapter_key='sam_promptable_segmentation'))
