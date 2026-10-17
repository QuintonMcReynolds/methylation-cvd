"""Stream GEO GSE220622 (EPIC, 391 whole-blood samples) into analysis-ready arrays.

The processed matrix interleaves, per sample, an M-value column and a detection p-value column. One streaming pass:
  - M-values -> float32 memmap  data/processed/M.f32  (probes x samples) + probe and sample indexes
  - detection p > 0.01 counted per sample and per probe (QC), then discarded
Then the sample sheet (series matrix) and the probe annotation (Illumina manifest: chromosome, SNP overlap) are
written as parquet, and a manifest records source hashes and shapes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
M_FILE = RAW / "GSE220622_processed_M-value.tsv.gz"
DETP_FAIL = 0.01


def sha256(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def sample_sheet() -> pd.DataFrame:
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(RAW / "GSE220622_series_matrix.txt.gz", "rt") as f:
        for line in f:
            if line.startswith(("!Sample_geo_accession", "!Sample_title", "!Sample_characteristics_ch1")):
                k = line.split("\t")[0]
                rows.setdefault(k, []).append([x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]])
    df = pd.DataFrame({"gsm": rows["!Sample_geo_accession"][0],
                       "sample_id": [t.split("[")[-1].rstrip("]") for t in rows["!Sample_title"][0]]})
    for ch in rows["!Sample_characteristics_ch1"]:
        key = ch[0].split(":")[0].strip().replace(" ", "_")
        df[key] = [x.split(": ", 1)[1] for x in ch]
    df["age"] = df["age"].astype(int)
    df["sex"] = df.pop("gender").str.lower()
    df["sa_extent"] = df.pop("pesa_score")
    return df


def probe_annotation(probes: list[str]) -> pd.DataFrame:
    cols = ["IlmnID", "CHR", "MAPINFO", "UCSC_RefGene_Name", "Relation_to_UCSC_CpG_Island", "SNP_ID", "SNP_DISTANCE",
            "SNP_MinorAlleleFrequency", "Infinium_Design_Type"]
    ann = pd.read_csv(RAW / "GPL21145_MethylationEPIC_manifest.csv.gz", skiprows=7, usecols=cols, low_memory=False)
    ann = ann[ann.IlmnID.astype(str).str.startswith(("cg", "ch", "rs"))].drop_duplicates("IlmnID").set_index("IlmnID")
    return ann.reindex(probes).rename_axis("probe").reset_index()


def stream_matrix(block_rows: int = 20000):
    opts = pacsv.ReadOptions(block_size=1 << 26)
    parse = pacsv.ParseOptions(delimiter="\t")
    reader = pacsv.open_csv(M_FILE, read_options=opts, parse_options=parse)
    names = reader.schema.names
    samples = names[1::2]
    assert all(n == "Detection Pval" or n.startswith("Detection Pval") for n in names[2::2]), "unexpected column layout"
    n_s = len(samples)
    probes: list[str] = []
    m_path = OUT / "M.f32"
    chunks, sample_fail, probe_fail = [], np.zeros(n_s, np.int64), []
    for batch in reader:
        ids = batch.column(0).to_pylist()
        cols = batch.columns
        M = np.column_stack([np.asarray(cols[1 + 2 * j].to_numpy(zero_copy_only=False), dtype=np.float32) for j in range(n_s)])
        P = np.column_stack([np.asarray(cols[2 + 2 * j].to_numpy(zero_copy_only=False), dtype=np.float32) for j in range(n_s)])
        fail = P > DETP_FAIL
        sample_fail += fail.sum(0)
        probe_fail.append(fail.mean(1).astype(np.float32))
        probes += ids
        chunks.append(M)
        if sum(c.shape[0] for c in chunks) >= block_rows * 10:
            _flush(m_path, chunks)
            chunks = []
    _flush(m_path, chunks)
    return samples, probes, sample_fail, np.concatenate(probe_fail)


def _flush(path: Path, chunks: list[np.ndarray]):
    if chunks:
        with open(path, "ab") as f:
            np.concatenate(chunks).astype(np.float32).tofile(f)


def load_M() -> tuple[np.memmap, list[str], list[str]]:
    meta = json.loads((OUT / "manifest.json").read_text())
    probes = pd.read_parquet(OUT / "probes.parquet").probe.tolist()
    samples = pd.read_parquet(OUT / "samples.parquet").sample_id.tolist()
    M = np.memmap(OUT / "M.f32", dtype=np.float32, mode="r", shape=tuple(meta["M_shape"]))
    return M, probes, samples


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "M.f32").unlink(missing_ok=True)
    samples, probes, sample_fail, probe_fail = stream_matrix()
    sheet = sample_sheet().set_index("sample_id").reindex(samples).reset_index()
    sheet["detp_fail_frac"] = sample_fail / len(probes)
    sheet.to_parquet(OUT / "samples.parquet", index=False)
    ann = probe_annotation(probes)
    ann["detp_fail_frac"] = probe_fail
    ann.to_parquet(OUT / "probes.parquet", index=False)
    man = {"source": "GEO GSE220622 (PESA study, Illumina EPIC, whole blood)", "M_shape": [len(probes), len(samples)],
           "dtype": "float32", "detection_p_threshold": DETP_FAIL,
           "source_sha256": {p.name: sha256(p) for p in sorted(RAW.glob("*.gz")) if p.stat().st_size < 3e9}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "source_sha256"}), sheet.sa_extent.value_counts().to_dict())


if __name__ == "__main__":
    main()
