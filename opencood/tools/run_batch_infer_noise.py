#!/usr/bin/env python3
import os
import argparse
import asyncio
from datetime import datetime
from glob import glob
import re

# ----------------------------
# Defaults
# ----------------------------
EGO_LIST_DEFAULT = ["m3", "m4", "m7", "m36"]
NOISE_LIST_DEFAULT = [0.1, 0.2, 0.3, 0.4, 0.5]


def list_subdirs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    subdirs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            subdirs.append(p)
    return subdirs


def _safe_tag(s: str):
    return s.replace("/", "_").replace(" ", "_")


def build_cmd(model_dir: str, ego: str, noise: float, script_path: str, range_str: str, use_cav: str, save_feat_interval: int):
    return " \\\n    ".join([
        f"python {script_path}",
        f"--range {range_str}",
        f"--use_cav '{use_cav}'",
        f"--ego_modality {ego}",
        f"--save_feat_interval {save_feat_interval}",
        f"--model_dir {model_dir}",
        f"--noise {noise}",
    ])


def has_checkpoint(model_dir: str) -> bool:
    """
    Consider the folder runnable if it contains any checkpoint file:
      - net_epoch_bestval_at*.pth, or
      - *epoch*.pth (including net_epoch*.pth)
    """
    model_dir = os.path.abspath(model_dir)
    if glob(os.path.join(model_dir, "net_epoch_bestval_at*.pth")):
        return True
    if glob(os.path.join(model_dir, "*epoch*.pth")):
        return True
    return False


def result_yaml_exists(model_dir: str, ego: str, noise: float) -> bool:
    """
    Skip rule:
    If there exists a yaml/yml under model_dir that corresponds to (ego, noise),
    then skip this task.

    - Precise ego match: egom3 should not match egom36, using (?!\\d)
    - Noise match: look for 'noise{value}' with a tolerant numeric pattern.
      Example filenames in your repo: ..._noise0.0_..._egom3.yaml
    """
    model_dir = os.path.abspath(model_dir)
    files = glob(os.path.join(model_dir, "*.yaml")) + glob(os.path.join(model_dir, "*.yml"))
    if not files:
        return False

    ego_pat = re.compile(rf"ego{re.escape(ego)}(?!\d)", re.IGNORECASE)

    # Build a tolerant noise string match:
    # noise=0.1 -> match "noise0.1" or "noise0.10" etc.
    # We'll accept any number that parses to the same rounded(3) value.
    target = round(float(noise), 3)

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
            return True

    return False


async def run_one(cmd: str, env: dict, log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[CMD]\n{cmd}\n\n")
        f.flush()
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=f,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        rc = await proc.wait()
        f.write(f"\n[RET] {rc}\n")
        return rc


async def worker(queue: asyncio.Queue, gpu_id: str, sem: asyncio.Semaphore, log_dir: str):
    log_dir = os.path.abspath(log_dir)
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        cmd, tag = item
        async with sem:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"gpu{gpu_id}_{ts}_{_safe_tag(tag)}.log")

            print(f"[START] GPU={gpu_id}  {tag}")
            rc = await run_one(cmd, env, log_path)
            print(f"[DONE ] GPU={gpu_id}  rc={rc}  {tag}  log={log_path}")

        queue.task_done()


async def main_async(args):
    args.log_dir = os.path.abspath(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("No GPUs provided, e.g. --gpus 0,1")

    # Parse ego_list: accept either JSON-like 'm3,m4' or python list passed as string
    ego_list = [x.strip() for x in args.ego_list.split(",") if x.strip()]
    noise_list = [float(x) for x in args.noise_list.split(",") if x.strip()]

    subdirs = list_subdirs(args.root_dir)

    tasks = []
    skipped_no_ckpt = 0
    skipped_has_yaml = 0

    for model_dir in subdirs:
        model_base = os.path.basename(model_dir)

        if args.require_ckpt and not has_checkpoint(model_dir):
            skipped_no_ckpt += 1
            print(f"[SKIP-FOLDER-NO-CKPT] {model_base}")
            continue

        for ego in ego_list:
            for noise in noise_list:
                if args.skip_if_yaml_exists and result_yaml_exists(model_dir, ego, noise):
                    skipped_has_yaml += 1
                    continue

                tag = f"{model_base}__ego_{ego}__noise_{noise}"
                cmd = build_cmd(
                    model_dir=model_dir,
                    ego=ego,
                    noise=noise,
                    script_path=args.script_path,
                    range_str=args.range,
                    use_cav=args.use_cav,
                    save_feat_interval=args.save_feat_interval,
                )
                tasks.append((cmd, tag))

    print(f"Model dirs: {len(subdirs)}")
    print(f"Egos: {ego_list}")
    print(f"Noises: {noise_list}")
    print(f"GPUs: {gpus}, per-GPU concurrency: {args.per_gpu}")
    print(f"Skipped folders(no ckpt): {skipped_no_ckpt}")
    print(f"Skipped tasks(yaml exists): {skipped_has_yaml}")
    print(f"Jobs to run: {len(tasks)}")
    print(f"Logs: {args.log_dir}")

    if not tasks:
        return 0

    q = asyncio.Queue()
    for cmd, tag in tasks:
        await q.put((cmd, tag))

    workers = []
    for gpu_id in gpus:
        sem = asyncio.Semaphore(args.per_gpu)
        for _ in range(args.per_gpu):
            workers.append(asyncio.create_task(worker(q, gpu_id, sem, args.log_dir)))

    for _ in workers:
        await q.put(None)

    await q.join()
    await asyncio.gather(*workers)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--script_path", type=str, default="opencood/tools/inference_heter_experiments.py")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--per_gpu", type=int, default=3, help="Concurrent jobs per GPU")

    parser.add_argument("--ego_list", type=str, default=",".join(EGO_LIST_DEFAULT),
                        help='Comma-separated egos, e.g. "m3,m4,m7,m36"')
    parser.add_argument("--noise_list", type=str, default=",".join(str(x) for x in NOISE_LIST_DEFAULT),
                        help='Comma-separated noises, e.g. "0.1,0.2,0.3"')

    parser.add_argument("--range", type=str, default="102.4,51.2")
    parser.add_argument("--use_cav", type=str, default="[5]")
    parser.add_argument("--save_feat_interval", type=int, default=400)

    parser.add_argument("--log_dir", type=str, default="opencood/logs/batch_logs_noise")
    parser.add_argument("--require_ckpt", action="store_true", default=True)
    parser.add_argument("--skip_if_yaml_exists", action="store_true", default=True)

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())