"""共享路径与 run 管理：脚本1写、脚本2读，统一中间数据传递。"""
import hashlib
import json
import pickle
from datetime import datetime
from fractions import Fraction
from pathlib import Path

# ============ 项目路径（基于本文件位置，与"在哪运行"无关）============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = DATA_DIR / "runs"

for _d in (DATA_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MATRIX_FILENAME = "matrix_elements.pkl"      # run 目录内固定名
META_FILENAME = "matrix_elements.meta.json"  # 参数随数据走
MANIFEST_NAME = "manifest.json"              # 全部历史 run
LATEST_NAME = "latest.json"                  # 最近一次 run


# ============ 脚本1 用：生成 run 目录并保存 ============
def _normalize_q(q):
    """把 Fraction / float / int 统一成标准字符串，避免 q 写法不同导致误判。"""
    if isinstance(q, Fraction):
        return f"{q.numerator}/{q.denominator}"
    f = float(q)
    if abs(f - round(f)) < 1e-10:
        return str(int(round(f)))
    fr = Fraction(f).limit_denominator(100)
    return f"{fr.numerator}/{fr.denominator}"


def compute_run_id(q, l_max, k_2D, threshold):
    """由参数生成 8 位唯一 ID（归一化后哈希，float 精度不会误判）。"""
    key = {
        "q": _normalize_q(q),
        "l_max": round(float(l_max), 10),
        "k_2D": round(float(k_2D), 10),
        "threshold": float(threshold),
    }
    s = json.dumps(key, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def make_run_dir(q, l_max, k_2D, threshold):
    """创建并返回本次 run 的目录（同名参数复用，不同参数并存）。"""
    q_str = _normalize_q(q).replace("/", "_")
    run_id = compute_run_id(q, l_max, k_2D, threshold)
    run_dir = RUNS_DIR / f"q{q_str}_lmax{l_max}_k{k_2D}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id, q_str


def save_matrix_elements(q, l_max, k_2D, threshold, payload, num_states):
    """保存矩阵元 + 元数据，并更新索引。返回 (run_dir, meta)。"""
    run_dir, run_id, q_str = make_run_dir(q, l_max, k_2D, threshold)
    matrix_path = run_dir / MATRIX_FILENAME
    meta_path = run_dir / META_FILENAME

    # ① 大文件：矩阵元（目录内固定名，脚本2 不用猜）
    with open(matrix_path, "wb") as f:
        pickle.dump(payload, f)

    # ② 小文件：参数元数据
    meta = {
        "run_id": run_id,
        "q": float(q),
        "q_str": q_str,
        "l_max": float(l_max),
        "k_2D": k_2D,
        "threshold": float(threshold),
        "num_states": num_states,
        "matrix_file": MATRIX_FILENAME,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ③ 更新索引
    _update_manifest(run_dir, run_id, meta)
    return run_dir, meta


def _update_manifest(run_dir, run_id, meta):
    entry = {
        "run_id": run_id,
        "dir_name": run_dir.name,
        "created": meta["created"],
        "q": meta["q"], "q_str": meta["q_str"],
        "l_max": meta["l_max"], "k_2D": meta["k_2D"],
        "threshold": meta["threshold"],
        "num_states": meta["num_states"],
    }
    manifest_path = RUNS_DIR / MANIFEST_NAME
    runs = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            runs = json.load(f)
    runs = [r for r in runs if r["run_id"] != run_id]   # 同参数重跑则替换旧记录
    runs.append(entry)
    runs.sort(key=lambda r: r["created"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)
    with open(RUNS_DIR / LATEST_NAME, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


# ============ 脚本2 用：定位 & 选择 run ============
def load_manifest():
    manifest_path = RUNS_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"没有任何已保存的 run（{manifest_path} 不存在），请先运行脚本1"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    if not runs:
        raise FileNotFoundError("manifest 为空，请先运行脚本1")
    return runs


def resolve_run(run_id=None):
    """按 run_id 定位；run_id 为空则最近一次。返回 (run_dir, meta)。"""
    runs = load_manifest()
    if run_id is None:
        entry = runs[-1]
    else:
        entry = next((r for r in runs if r["run_id"].startswith(run_id)), None)
        if entry is None:
            raise FileNotFoundError(
                f"找不到 run_id={run_id}，可用：{[r['run_id'] for r in runs]}"
            )
    run_dir = RUNS_DIR / entry["dir_name"]
    with open(run_dir / META_FILENAME, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return run_dir, meta


def choose_run_interactive():
    """交互式菜单：列出所有 run，回车用最近一次。返回 run_id。"""
    runs = load_manifest()
    print("\n可用的矩阵元数据：")
    print("-" * 72)
    for i, r in enumerate(runs):
        mark = "  ← 最近" if i == len(runs) - 1 else ""
        print(f"  [{i}] q={r['q_str']:>5}  l_max={r['l_max']:<5} "
              f"k={r['k_2D']}  态数={r['num_states']:>6}  "
              f"{r['created']}  {r['run_id']}{mark}")
    print("-" * 72)
    print(f"  直接回车 = 使用最近一次（{runs[-1]['run_id']}）")

    while True:
        sel = input("请选择编号：").strip()
        if sel == "":
            return runs[-1]["run_id"]
        if sel.isdigit() and 0 <= int(sel) < len(runs):
            return runs[int(sel)]["run_id"]
        print(f"  无效输入，请输入 0-{len(runs) - 1} 或直接回车")