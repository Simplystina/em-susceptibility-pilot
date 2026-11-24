"""One GPU worker for the pilot. Started by emlib.launch_workers; not run by hand.

Exit codes: 0 finished its share, 3 stopped for the session deadline (re-run to
continue), anything else is an error (see the worker log).
"""
import argparse
import json
import sys
import traceback

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["anchor", "susc"], required=True)
p.add_argument("--cfg", required=True)
p.add_argument("--questions", required=True)
p.add_argument("--worker", type=int, default=0)
p.add_argument("--n_workers", type=int, default=1)
p.add_argument("--deadline", type=float, default=None)
a = p.parse_args()

import emlib as E  # imported after argument parsing so CUDA_VISIBLE_DEVICES is already set

print(f"worker {a.worker}/{a.n_workers} | {E.device_report()}", flush=True)
cfg = E.load_cfg(a.cfg)
questions = json.load(open(a.questions))
budget = E.Budget(a.deadline)
try:
    if a.stage == "anchor":
        E.anchor_gpu(cfg, questions, a.worker, a.n_workers, budget)
    else:
        E.susc_gpu(cfg, questions, a.worker, a.n_workers, budget)
    print("WORKER FINISHED", flush=True)
except E.OutOfTime as e:
    print(f"WORKER STOPPED FOR TIME: {e}", flush=True)
    sys.exit(3)
except Exception:
    traceback.print_exc()
    sys.exit(1)
