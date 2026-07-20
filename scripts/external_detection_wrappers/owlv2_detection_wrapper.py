"""OWLv2 external detection wrapper entrypoint."""

from __future__ import annotations

from external_detection_contract import run_contract_wrapper


if __name__ == '__main__':
    raise SystemExit(run_contract_wrapper(model_family='owlv2', adapter_key='owlv2_open_vocab_detection'))
