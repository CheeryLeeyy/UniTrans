#!/usr/bin/env python3
import os
import re
import argparse
from glob import glob
from typing import Dict, Tuple, Optional

import csv
import yaml

METRICS = ["ap_30", "ap_50", "ap_70"]


def list_subdirs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    return [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def noise_to_key(noise: float) -> str:
    # unify to one decimal by default, matches your filenames like noise0.1
    return f"{float(noise):.1f}"


def find_yaml(folder: str, ego: str, noise: float) -> Optional[str]:
    """
    Precisely match ego (avoid m3 matching m36) and match noise value.
    Filename examples:
      eval_..._noise0.1_..._egom3.yaml
    """
    files = glob(os.path.join(folder, "*.yaml")) + glob(os.path.join(folder, "*.yml"))
    if not files:
        return None

    ego_pat = re.compile(rf"ego{re.escape(ego)}(?!\d)", re.IGNORECASE)

    # Parse noise number from filename and compare rounded(3)
    target = round(float(noise), 3)

    cand = []
    for p in files:
        base = os.path.basename(p)
        if not ego_pat.search(base):
            continue
        m = re.search(r"noise([0-9]*\.?[0-9]+)", base, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            nv = round(float(m.group(1)), 3)
        except ValueError:
            continue
        if nv == target:
            cand.append(p)

    if not cand:
        return None

    # if multiple, take newest
    cand.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cand[0]


def _extract_metrics_from_text(text: str) -> Dict[str, float]:
    out = {m: 0.0 for m in METRICS}
    for m in METRICS:
        r = re.search(rf"{re.escape(m)}\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
        if r:
            out[m] = float(r.group(1))
    return out


def read_metrics_from_yaml(yaml_path: str) -> Dict[str, float]:
    with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    try:
        data = yaml.safe_load(text)
    except Exception:
        return _extract_metrics_from_text(text)

    if isinstance(data, dict):
        ok = True
        out = {}
        for m in METRICS:
            v = data.get(m, None)
            if isinstance(v, (int, float)):
                out[m] = float(v)
            else:
                ok = False
        if ok:
            return out

    return _extract_metrics_from_text(text)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_one_csv(out_csv: str, methods: Dict[str, Dict[str, float]], noise_keys):
    """
    methods[method_name][noise_key] = value
    若某个 method 在所有 noise 上都为 0，则不写入该行
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + list(noise_keys))

        for method_name in sorted(methods.keys()):
            vals = [float(methods[method_name].get(nk, 0.0)) for nk in noise_keys]

            # 全 0 行直接跳过
            if all(v == 0.0 for v in vals):
                continue

            row = [method_name] + [f"{v:.4f}" for v in vals]
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--ego_list", type=str, default="m3,m4,m7,m36")
    parser.add_argument("--noise_list", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--out_dir", type=str, default="./csv_by_ego_ap")
    args = parser.parse_args()

    ego_list = parse_str_list(args.ego_list)
    noise_list = parse_float_list(args.noise_list)
    noise_keys = [noise_to_key(n) for n in noise_list]

    root_dir = os.path.abspath(args.root_dir)
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    subdirs = list_subdirs(root_dir)

    # data[(ego, metric)][method][noise_key] = value
    data: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for ego in ego_list:
        for metric in METRICS:
            data[(ego, metric)] = {}

    for sub in subdirs:
        method = os.path.basename(sub)
        # initialize per ego/metric maps for this method
        for ego in ego_list:
            for metric in METRICS:
                data[(ego, metric)].setdefault(method, {})

        for ego in ego_list:
            for noise, nk in zip(noise_list, noise_keys):
                ypath = find_yaml(sub, ego, noise)
                if not ypath:
                    # missing yaml -> keep default (0)
                    continue
                mvals = read_metrics_from_yaml(ypath)
                for metric in METRICS:
                    data[(ego, metric)][method][nk] = mvals.get(metric, 0.0)

    # write CSVs: one per ego+ap
    for ego in ego_list:
        for metric in METRICS:
            out_csv = os.path.join(out_dir, f"{ego}_{metric}.csv")
            write_one_csv(out_csv, data[(ego, metric)], noise_keys)

    print(f"Saved CSVs to: {out_dir}")
    print("Files:")
    for ego in ego_list:
        for metric in METRICS:
            print(f"  {ego}_{metric}.csv")


if __name__ == "__main__":
    main()