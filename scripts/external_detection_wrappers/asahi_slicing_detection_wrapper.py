"""ASAHI slicing external detection wrapper entrypoint."""

from __future__ import annotations

from external_detection_contract import run_contract_wrapper


if __name__ == '__main__':
    raise SystemExit(run_contract_wrapper(model_family='asahi_slicing', adapter_key='asahi_slicing_detection'))
