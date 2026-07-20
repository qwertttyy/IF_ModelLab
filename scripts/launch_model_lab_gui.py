from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.ui.tk_app import EngineTkApp


def main() -> None:
    root = tk.Tk()
    root.geometry('1100x760+80+80')
    root.attributes('-topmost', True)
    EngineTkApp(root=root, project_dir=PROJECT_ROOT)
    root.lift()
    root.focus_force()
    root.after(2000, lambda: root.attributes('-topmost', False))
    root.mainloop()


if __name__ == '__main__':
    main()
