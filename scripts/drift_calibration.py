"""False-alarm and detection rates of the drift monitor on held-out samples (no artefact vs the injected artefact)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate_production import inject_batch_effect  # noqa: E402

from mcvd.data import task  # noqa: E402
from mcvd.drift import DriftMonitor  # noqa: E402
from mcvd.model import MethylClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(n_monitors: int = 5, per_monitor: int = 8):
    Z, y, probes, _ = task("generalized_vs_none")
    X = Z[:, :-2]
    ann = pd.read_parquet(ROOT / "data" / "processed" / "probes.parquet").set_index("probe")
    type2 = (ann.loc[probes, "Infinium_Design_Type"].astype(str) == "II").to_numpy()
    tr, stream = next(StratifiedShuffleSplit(1, test_size=0.4, random_state=0).split(Z, y))
    champ = MethylClassifier(k=2000, C=0.05).fit(Z[tr], y[tr])
    cols = champ.var_idx_[:50000]
    pos = np.searchsorted(cols, champ.var_idx_[champ.idx_])
    pos = pos[(pos < len(cols)) & (cols[np.minimum(pos, len(cols) - 1)] == champ.var_idx_[champ.idx_])]
    rows = []
    for m in range(n_monitors):
        mon = DriftMonitor(seed=m).fit(X[tr][:, cols], np.zeros(len(tr)), pos)
        rng = np.random.default_rng(100 + m)
        for _ in range(per_monitor):
            b = rng.choice(stream, 28, replace=False)
            for art in (False, True):
                Xb = inject_batch_effect(X[b], type2) if art else X[b]
                r = mon.check(Xb[:, cols], np.zeros(28))
                rows.append({"artifact": art, "mmd_alarm": r["batch_drift"], "hold": r["hold"]})
    df = pd.DataFrame(rows)
    out = {k: {"n": int(len(g)), "mmd_alarm_rate": float(g.mmd_alarm.mean()), "hold_rate": float(g.hold.mean())}
           for k, g in df.groupby(df.artifact.map({False: "clean", True: "artifact"}))}
    (ROOT / "reports" / "drift_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
