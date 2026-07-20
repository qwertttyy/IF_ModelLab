import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.remote_task_adapter import main


if __name__ == '__main__':
    raise SystemExit(main())
