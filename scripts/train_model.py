"""Optional CPU training: a logistic-regression probe per label on frozen TorchXRayVision features.

    python -m scripts.train_model --dataset iu_xray [--weights densenet121-res224-nih] [--max-cases 2000]
    python -m scripts.train_model --dataset iu_xray --predict-test     # only after model selection is final

Protocol
- Features: DenseNet-121 penultimate layer (1024-d), frozen. Cached in models/linear_probe/<dataset>/features_<split>.npz.
- Training uses the train split only; the regularisation strength C is chosen on the val split (AUROC).
- The test split is not read during training or selection. --predict-test writes a predictions file
  that scripts.evaluate scores; run it once, after you have stopped changing the model.
- For IU X-Ray prefer --weights densenet121-res224-nih: the default 'all' weights were trained on Open-i,
  so their features have already seen these images.
Hardware: CPU is enough. Feature extraction is roughly 0.1-0.5 s per image on a laptop CPU (not benchmarked here).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app.core.config import get_settings
from app.services.evaluation_service import ground_truth, load_label_map
from app.services.manifest_service import ManifestStore

C_GRID = (0.01, 0.1, 1.0, 10.0)
MIN_POS = 10


def train_probe(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, seed: int = 0):
    """Fit one label. Returns (model, chosen C, val AUROC per C) or None if a split lacks both classes."""
    if ytr.min() == ytr.max() or yva.min() == yva.max() or ytr.sum() < MIN_POS:
        return None
    scores = {}
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        scores[c] = round(float(auc), 4)
        if best is None or auc > best[1]:
            best = (clf, auc, c)
    return best[0], best[2], scores


class FeatureExtractor:
    def __init__(self, weights: str, model_dir: Path):
        import torch
        import torchxrayvision as xrv

        from app.agents.image_agent import to_grayscale_array

        self.torch, self.xrv, self.to_gray = torch, xrv, to_grayscale_array
        self.model = xrv.models.DenseNet(weights=weights, cache_dir=str(model_dir)).eval()

    def __call__(self, path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            gray, max_value, _ = self.to_gray(im)
        img = self.xrv.datasets.normalize(gray, max_value)[None, ...]
        img = self.xrv.datasets.XRayResizer(224)(self.xrv.datasets.XRayCenterCrop()(img))
        with self.torch.no_grad():
            return self.model.features2(self.torch.from_numpy(img).float()[None, ...])[0].numpy()


def extract(store: ManifestStore, split: str, fx, cache: Path, max_cases: int | None):
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["X"], list(d["case_ids"])
    feats, ids = [], []
    recs = [r for r in store.records if r.split == split][:max_cases]
    for i, rec in enumerate(recs, 1):
        path = store.image_path(rec, 0)  # first image per case (frontal in most IU reports; not guaranteed)
        if path is None:
            continue
        feats.append(fx(path)); ids.append(rec.case_id)
        if i % 100 == 0:
            print(f"  {split}: {i}/{len(recs)}")
    X = np.stack(feats) if feats else np.zeros((0, 1024), np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, case_ids=np.array(ids))
    return X, ids


def main(argv=None) -> int:
    import joblib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", default="densenet121-res224-nih")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args(argv)
    s = get_settings()
    store = ManifestStore(s.manifest_dir_abs / f"{args.dataset}.jsonl", None, "cli")
    if store.error:
        print(f"Error: {store.error}", file=sys.stderr)
        return 1
    out_dir = s.model_path_abs.parent / "linear_probe" / args.dataset
    label_map = load_label_map(s.resolve(f"config/label_maps/{args.dataset}.yaml"))
    if not label_map:
        print("Error: a label map in config/label_maps/ is required.", file=sys.stderr)
        return 1
    try:
        fx = FeatureExtractor(args.weights, s.model_path_abs)
    except Exception as exc:
        print(f"Error: could not load {args.weights} ({exc}). Download with scripts.download_model --weights {args.weights}",
              file=sys.stderr)
        return 1
    labels = sorted(set(label_map.values()))

    def y_for(ids):
        rows = [ground_truth(store.by_id[c], label_map, set(labels), "exclude") for c in ids]
        return {lab: np.array([r.get(lab, -1) for r in rows]) for lab in labels}

    if args.predict_test:
        cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        Xte, ids = extract(store, "test", fx, out_dir / "features_test.npz", None)
        models = joblib.load(out_dir / "probe.joblib")
        out = s.evaluation_dir_abs / f"predictions_{args.dataset}_test_probe.jsonl"
        model_info = {"backend": "linear_probe", "name": f"Linear probe on {args.weights}", "version": cfg["trained_at"],
                      "labels": list(models), "training_datasets": cfg["feature_training_datasets"] + [args.dataset]}
        with out.open("w", encoding="utf-8") as fh:
            for i, cid in enumerate(ids):
                scores = {lab: round(float(m.predict_proba(Xte[i:i + 1])[0, 1]), 4) for lab, m in models.items()}
                fh.write(json.dumps({"case_id": cid, "image_index": 0, "dataset_source": args.dataset,
                                     "model": model_info, "scores": scores, "latency_ms": None, "error": None}) + "\n")
        print(f"Wrote {out}. Note: training_datasets includes {args.dataset} (train split), so scripts.evaluate "
              "will warn; the test split itself was never used for fitting.")
        return 0

    Xtr, tr_ids = extract(store, "train", fx, out_dir / "features_train.npz", args.max_cases)
    Xva, va_ids = extract(store, "val", fx, out_dir / "features_val.npz", args.max_cases)
    ytr, yva = y_for(tr_ids), y_for(va_ids)
    models, summary = {}, {}
    for lab in labels:
        mtr, mva = ytr[lab] >= 0, yva[lab] >= 0
        res = train_probe(Xtr[mtr], ytr[lab][mtr], Xva[mva], yva[lab][mva])
        if res is None:
            summary[lab] = "skipped: too few positives or a split lacks both classes"
            continue
        models[lab], c, scores = res
        summary[lab] = {"C": c, "val_auroc_by_C": scores, "train_pos": int(ytr[lab][mtr].sum()),
                        "val_pos": int(yva[lab][mva].sum())}
    from app.agents.model_adapters import TXRV_TRAINING_DATA
    joblib.dump(models, out_dir / "probe.joblib")
    cfg = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": args.dataset,
           "feature_extractor": args.weights, "feature_training_datasets": TXRV_TRAINING_DATA.get(args.weights, []),
           "preprocessing": "first image per case; normalize to [-1024,1024]; center crop; resize 224; features2",
           "label_map": label_map, "train_cases": len(tr_ids), "val_cases": len(va_ids), "labels": summary,
           "selection": "C chosen by validation AUROC; test split unused"}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {len(models)} probe(s) to {out_dir}. These validation AUROCs were used for selection and are "
          "optimistic; report test metrics from --predict-test + scripts.evaluate instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
