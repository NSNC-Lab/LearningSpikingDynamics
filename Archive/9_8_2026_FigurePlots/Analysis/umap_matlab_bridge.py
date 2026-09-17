"""UMAP embedding worker called by MATLAB; no classifiers or plots here."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
from scipy.io import loadmat, savemat
import umap


def main():
    source, destination = map(Path, sys.argv[1:3])
    jobs = loadmat(source, simplify_cells=True)["jobs"]
    if isinstance(jobs, dict):
        jobs = [jobs]
    cache = destination.parent / "umap_cache"
    cache.mkdir(exist_ok=True)
    output = {}
    for index, job in enumerate(jobs, 1):
        X = np.asarray(job["X"], dtype=np.float64)
        query = np.asarray(job["query"], dtype=np.float64)
        if query.size == 0:
            query = np.empty((0, X.shape[1]))
        y = np.asarray(job["y"], dtype=np.int32).reshape(-1)
        settings = dict(n_neighbors=int(job["neighbors"]),
                        min_dist=float(job["minDist"]),
                        target_weight=float(job["targetWeight"]),
                        random_state=int(job["seed"]),
                        transform_seed=int(job["seed"]),
                        n_epochs=500, n_components=2, n_jobs=1,
                        metric="euclidean")
        supervised = bool(job["supervised"])
        h = hashlib.sha256()
        for a in (X, query, y):
            h.update(str(a.shape).encode()); h.update(a.tobytes())
        h.update(json.dumps(settings, sort_keys=True).encode())
        h.update(str(supervised).encode())
        h.update(umap.__version__.encode())
        path = cache / (h.hexdigest() + ".npz")
        if path.exists():
            saved = np.load(path)
            train, test = saved["train"], saved["test"]
        else:
            model = umap.UMAP(**settings)
            train = model.fit_transform(X, y=y if supervised else None)
            test = model.transform(query) if len(query) else np.empty((0, 2))
            assert np.isfinite(train).all() and np.isfinite(test).all()
            np.savez_compressed(path, train=train, test=test)
        output[f"train_{index}"] = train
        output[f"test_{index}"] = test
        if index % 10 == 0 or index == len(jobs):
            print(f"UMAP embedding jobs {index}/{len(jobs)}", flush=True)
    savemat(destination, output, do_compression=True)
    versions = {p: importlib.metadata.version(p) for p in
                ["umap-learn", "numpy", "scipy", "scikit-learn", "numba", "pynndescent"]}
    (destination.parent / "umap_versions.json").write_text(json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()
