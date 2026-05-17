import os
import sys
import argparse
import asyncio
from datetime import datetime
from glob import glob
import re

EGO_LIST_DEFAULT = ["m3", "m4", "m7", "m13", "m17", "m31", "m36", "m39"]


def result_yaml_exists(model_dir: str, ego: str) -> bool:
    """
    Check whether result yaml for this ego already exists under model_dir.
    Precise match: egom3 should not match egom36/egom39.
    """
    model_dir = os.path.abspath(model_dir)
    files = glob(os.path.join(model_dir, "*.yaml")) + glob(os.path.join(model_dir, "*.yml"))
    if not files:
        return False

    # match "egom3" but not "egom36"
    pat = re.compile(rf"ego{re.escape(ego)}(?!\d)", re.IGNORECASE)
    for p in files:
        if pat.search(os.path.basename(p)):
            return True
    return False


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

def build_cmd(
    model_dir: str,
    ego_modality: str,
    script_path: str,
    range_str: str,
    use_cav: str,
    save_feat_interval: int,
    extra_args: str,
):
    parts = [
        f"python {script_path}",
        f"--range {range_str}",
        f"--use_cav '{use_cav}'",
        f"--ego_modality {ego_modality}",
        f"--save_feat_interval {save_feat_interval}",
        f"--model_dir {model_dir}",
    ]
    if extra_args.strip():
        parts.append(extra_args.strip())
    return " \\\n    ".join(parts)

def _safe_tag(s: str):
    return s.replace("/", "_").replace(" ", "_")

def tail_contains_ret0(path: str, n_lines: int = 30) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n_lines:
                step = block if size >= block else size
                size -= step
                f.seek(size, os.SEEK_SET)
                data = f.read(step) + data
            text = data.decode("utf-8", errors="ignore")
        return "[RET] 0" in text.splitlines()[-n_lines:]
    except Exception:
        return False

def task_succeeded(log_dir: str, tag: str) -> bool:
    """
    找包含 safe_tag(tag) 的最近 log，且末尾有 [RET] 0 则认为成功
    注意：日志文件名形如 gpu0_时间戳_{tag}.log，因此必须用 *{tag}*.log
    """
    log_dir = os.path.abspath(log_dir)
    # pattern = os.path.join(log_dir, f"*{_safe_tag(tag)}*.log")
    pattern = os.path.join(log_dir, f"*_{_safe_tag(tag)}.log")
    cand = glob(pattern)
    if not cand:
        return False
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return tail_contains_ret0(cand[0], n_lines=30)

def get_model_path_from_dir(model_dir: str) -> str:
    """
    Returns the checkpoint path if found, otherwise raises RuntimeError.
    Priority:
      1) net_epoch_bestval_at*.pth (expect exactly 1 if exists)
      2) the largest epoch among *epoch*.pth, returning net_epoch{max}.pth
    """
    def find_last_checkpoint(save_dir: str) -> str:
        file_list = glob(os.path.join(save_dir, '*epoch*.pth'))
        if not file_list:
            raise RuntimeError(f"No checkpoint found under {save_dir}")

        epochs_exist = []
        for file_ in file_list:
            result = re.findall(r".*epoch(.*).pth.*", file_)
            if result:
                # result[0] might be like "69" or "69_best" (rare); keep strict int
                try:
                    epochs_exist.append(int(result[0]))
                except ValueError:
                    pass

        if not epochs_exist:
            raise RuntimeError(f"No parsable epoch checkpoints under {save_dir}")

        max_epoch = max(epochs_exist)
        return os.path.join(save_dir, f'net_epoch{max_epoch}.pth')

    file_list = glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))
    if len(file_list):
        if len(file_list) != 1:
            raise RuntimeError(f"Expected exactly 1 net_epoch_bestval_at*.pth, got {len(file_list)}: {file_list}")
        model_path = file_list[0]
    else:
        model_path = find_last_checkpoint(model_dir)

    print(f"[get_model_path_from_dir] find {model_path}")
    return model_path

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

async def worker(task_queue: asyncio.Queue, gpu_id: str, sem: asyncio.Semaphore, log_dir: str):
    log_dir = os.path.abspath(log_dir)
    while True:
        item = await task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        cmd, tag = item
        async with sem:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_tag = _safe_tag(tag)
            log_path = os.path.join(log_dir, f"gpu{gpu_id}_{ts}_{safe_tag}.log")

            print(f"[START] GPU={gpu_id}  {tag}")
            rc = await run_one(cmd, env, log_path)
            print(f"[DONE ] GPU={gpu_id}  rc={rc}  {tag}  log={log_path}")

        task_queue.task_done()

async def main_async(args):
    # Ensure absolute log_dir and create if missing
    args.log_dir = os.path.abspath(args.log_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    subdirs = list_subdirs(args.root_dir)
    if not subdirs:
        print(f"No sub-directories found under: {args.root_dir}")
        return 0

    ego_list = args.ego_list.split(",") if args.ego_list else EGO_LIST_DEFAULT
    ego_list = [x.strip() for x in ego_list if x.strip()]

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("No GPUs provided. Example: --gpus 0,1")

    tasks = []
    skipped_tasks = 0
    skipped_folders_no_ckpt = 0

    for model_dir in subdirs:
        model_base = os.path.basename(model_dir)

        # New: check checkpoint existence before scheduling any tasks for this folder
        try:
            _ = get_model_path_from_dir(model_dir)
        except Exception as e:
            skipped_folders_no_ckpt += 1
            print(f"[SKIP-FOLDER-NO-CKPT] {model_base}  reason={e}")
            continue

        for ego in ego_list:
            tag = f"{model_base}__ego_{ego}"

            # Skip if result yaml already exists in this model_dir
            if args.skip_task_if_success and result_yaml_exists(model_dir, ego):
                skipped_tasks += 1
                print(f"[SKIP-TASK-YAML] {model_base} ego={ego}")
                continue

            cmd = build_cmd(
                model_dir=model_dir,
                ego_modality=ego,
                script_path=args.script_path,
                range_str=args.range,
                use_cav=args.use_cav,
                save_feat_interval=args.save_feat_interval,
                extra_args=args.extra_args,
            )
            tasks.append((cmd, tag))

    print(f"Found {len(subdirs)} model dirs.")
    print(f"Ego modalities: {ego_list}")
    print(f"GPUs: {gpus}, per-GPU concurrency: {args.per_gpu}")
    print(f"Skipped folders (no ckpt): {skipped_folders_no_ckpt}")
    print(f"Skipped tasks (already success): {skipped_tasks}")
    print(f"Total jobs to run: {len(tasks)}")
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
    parser.add_argument("--root_dir", type=str, default="opencood/logs/test/baseline")
    parser.add_argument("--script_path", type=str, default="opencood/tools/inference_heter_experiments.py")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--per_gpu", type=int, default=2)
    # parser.add_argument("--ego_list", type=str, default="")
    parser.add_argument("--ego_list", type=str, default=",".join(EGO_LIST_DEFAULT))
    parser.add_argument("--range", type=str, default="102.4,51.2")
    parser.add_argument("--use_cav", type=str, default="[5]")
    parser.add_argument("--save_feat_interval", type=int, default=40)
    parser.add_argument("--log_dir", type=str, default="opencood/logs/batch_logs")
    parser.add_argument("--extra_args", type=str, default="")

    # Only keep task-level skip (your intended behavior)
    parser.add_argument("--skip_task_if_success", action="store_true", default=True, help="Skip task if result yaml for this ego already exists under the model_dir",)

    args = parser.parse_args()
    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        rc = 130
    return rc

if __name__ == "__main__":
    sys.exit(main())