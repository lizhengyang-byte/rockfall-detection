"""Execute rock_fall_001.ipynb code cells in one shared namespace with a tiny epoch override."""
import json, re, time, traceback
from pathlib import Path

nb = json.load(open("rock_fall_001.ipynb", encoding="utf-8"))
ns = {"__name__": "__main__"}

def run_cell(src, idx):
    # strip IPython magics
    lines = []
    for ln in src.split("\n"):
        if ln.lstrip().startswith("%matplotlib") or ln.lstrip().startswith("%"):
            continue
        lines.append(ln)
    code = "\n".join(lines)
    # tiny-config override: reduce epochs so training test is fast
    code = code.replace("epochs=60", "epochs=3")
    try:
        exec(compile(code, f"<cell{idx}>", "exec"), ns)
        return True, None
    except Exception:
        return False, traceback.format_exc()

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "\n".join(c["source"])
    t = time.time()
    ok, err = run_cell(src, i)
    dt = time.time() - t
    print(f"[cell {i:02d}] {'OK' if ok else 'FAIL'} ({dt:.1f}s)")
    if not ok:
        print("  first line:", src.splitlines()[0])
        print(err)
        break
