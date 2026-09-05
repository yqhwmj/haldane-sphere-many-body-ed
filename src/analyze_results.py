import argparse
import pickle
import math
import os
import sys
import glob

def format_fractional(x, tol=1e-9):
    """
    将数值格式化为更易读的字符串：
    - 近似整数：返回整数
    - 近似半整数：返回 n/2
    - 否则：返回小数（保留6位或科学计数）
    """
    if x is None:
        return "None"
    # 尝试整数 
    ri = round(x)
    if abs(x - ri) < tol:
        return str(int(ri))
    # 尝试半整数
    r2 = round(2.0 * x)
    if abs(2.0 * x - r2) < tol:
        return f"{int(r2)}/2"
    # 常规小数
    ax = abs(x)
    if ax != 0 and (ax < 1e-4 or ax > 1e4):
        return f"{x:.6e}" 
    return f"{x:.6f}"

def format_state_tuple(state):
    """
    将 (l, m) 转为字符串 "(l_str, m_str)"
    state 可以是 (float, float) 或 (Fraction, Fraction)
    """
    if state is None or len(state) != 2:
        return "(?, ?)"
    l, m = state
    # 支持 Fraction 或 float
    try:
        lf = float(l)
    except Exception:
        lf = l
    try:
        mf = float(m)
    except Exception:
        mf = m
    return f"({format_fractional(lf)}, {format_fractional(mf)})"

def normalize_vdict(data):
    """
    统一不同 pkl 结构为：
    - 返回 vitems: 列表[(i,j,k,l,value)]
    - 返回 index_to_state: dict{i: (l,m)}
    - 返回 parameters: dict
    支持：
    - 2.py 的 'V_pairs'，键为 ((i,j),(k,l))
    - matrix_element_calculator.py 的 'matrix_elements'，键为 (i,j,k,l)
    """
    parameters = data.get('parameters', {})
    index_to_state = data.get('index_to_state', {})
    # 兼容 index_to_state 若为列表 
    if isinstance(index_to_state, list):
        index_to_state = {i: index_to_state[i] for i in range(len(index_to_state))}

    vitems = []
    if 'V_pairs' in data and isinstance(data['V_pairs'], dict):
        # 2.py 格式：键为 ((i,j),(k,l))
        for key, val in data['V_pairs'].items():
            try:
                (i, j), (k, l) = key
                vitems.append((int(i), int(j), int(k), int(l), val))
            except Exception:
                # 跳过异常键
                continue
    elif 'matrix_elements' in data and isinstance(data['matrix_elements'], dict):
        # 旧版格式：键为 (i,j,k,l)
        for key, val in data['matrix_elements'].items():
            try:
                i, j, k, l = key
                vitems.append((int(i), int(j), int(k), int(l), val))
            except Exception:
                continue
    else:
        raise ValueError("未在 pkl 中找到可识别的矩阵元数据（期望键 'V_pairs' 或 'matrix_elements'）。")

    return vitems, index_to_state, parameters

def value_to_str(val, imag_tol=1e-12):
    """
    将数值矩阵元格式化为字符串。
    - 若为实数或虚部很小：输出实部
    - 否则：输出 a + bi
    """
    try:
        vr = float(val)
    # 纯实数
        return f"{vr:.10e}"
    except Exception:
    # 可能是复数
        try:
            vr = float(val.real)
            vi = float(val.imag)
            if abs(vi) < imag_tol:
                return f"{vr:.10e}"
            return f"{vr:.10e} + {vi:.10e}i"
        except Exception:
            return str(val)
        
def compute_m(index_to_state, idx):
    """
    返回索引对应的 m 值（float），若不可用则返回 None
    """
    st = index_to_state.get(idx)
    if not st or len(st) != 2:
        return None
    try:
        return float(st[1])
    except Exception:
        return None
    
def main():
    parser = argparse.ArgumentParser(description="Convert V pkl (from 2.py or matrix_element_calculator.py) to readable txt.")
    parser.add_argument('--input', type=str, required=True, help="input pkl path")
    parser.add_argument('--output', type=str, required=True, help="output txt path")
    parser.add_argument('--min-abs', type=float, default=0.0, help="only write entries with |V| >= min_abs")
    parser.add_argument('--top-n', type=int, default=0, help="limit number of entries written (0 means all)")
    parser.add_argument('--sort', type=str, default='abs', choices=['abs', 'value', 'none'], help="sorting mode: abs, value, or none")
    parser.add_argument('--bucket-summary', action='store_true', help="write per-M bucket summary before entries")
    args = parser.parse_args()

    # 读取 pkl
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    vitems, index_to_state, parameters = normalize_vdict(data)

    # 过滤
    filtered = []
    for (i, j, k, l, v) in vitems:
        try:
            av = abs(v) if not isinstance(v, complex) else abs(v)
        except Exception:
            # 尝试 float
            try:
                av = abs(float(v))
            except Exception:
                av = 0.0
        if av >= args.min_abs:
            filtered.append((i, j, k, l, v, av))

    # 排序
    if args.sort == 'abs':
        filtered.sort(key=lambda t: t[5], reverse=True)
    elif args.sort == 'value':
        # 实部排序
        def real_part(x):
            v = x[4]
            try:
                return float(v.real) if isinstance(v, complex) else float(v)
            except Exception:
                return 0.0
        filtered.sort(key=real_part, reverse=True)
    # else 'none' 保持原顺序

    # 限制 top-n
    if args.top_n > 0:
        filtered = filtered[:args.top_n]

    # 写入 txt
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as out:
        out.write("Two-body interaction matrix elements (converted from pkl)\n")
        # 参数区
        if parameters:
            out.write("Parameters:\n")
            for k, v in parameters.items():
                out.write(f"  {k}: {v}\n")
        out.write(f"Total entries in pkl: {len(vitems)}\n")
        out.write(f"Entries after filtering: {len(filtered)} (min_abs={args.min_abs}, top_n={args.top_n if args.top_n>0 else 'all'})\n")

        # M 桶摘要（可选）
        if args.bucket_summary:
            buckets = {}
            for (i, j, k, l, v, av) in filtered:
                mi = compute_m(index_to_state, i)
                mj = compute_m(index_to_state, j)
                mk = compute_m(index_to_state, k)
                ml = compute_m(index_to_state, l)
                if mi is None or mj is None or mk is None or ml is None:
                    Mkey = "unknown"
                else:
                    Mij = mi + mj
                    Mkl = mk + ml
                    # 守恒检查
                    cons = "ok" if abs(Mij - Mkl) < 1e-9 else "viol"
                    Mkey = f"{format_fractional(Mij)} ({cons})"
                buckets.setdefault(Mkey, {"count": 0, "sum_abs": 0.0})
                buckets[Mkey]["count"] += 1
                buckets[Mkey]["sum_abs"] += av
            out.write("Bucket summary by M = m_i + m_j:\n")
            for Mkey, stats in sorted(buckets.items(), key=lambda kv: kv[1]["count"], reverse=True):
                out.write(f"  M={Mkey}: count={stats['count']}, sum|V|={stats['sum_abs']:.6e}\n")

        # 明细区域
        out.write("Entries:\n")
        out.write("Format: (i,j;k,l) | (li,mi)+(lj,mj) -> (lk,mk)+(ll,ml) | M=m_i+m_j | V\n")
        for (i, j, k, l, v, av) in filtered:
            si = index_to_state.get(i)
            sj = index_to_state.get(j)
            sk = index_to_state.get(k)
            sl = index_to_state.get(l)
            si_str = format_state_tuple(si)
            sj_str = format_state_tuple(sj)
            sk_str = format_state_tuple(sk)
            sl_str = format_state_tuple(sl)
            mi = compute_m(index_to_state, i)
            mj = compute_m(index_to_state, j)
            Mij = None if (mi is None or mj is None) else (mi + mj)
            M_str = format_fractional(Mij) if Mij is not None else "unknown"
            v_str = value_to_str(v)
            out.write(f"({i},{j};{k},{l}) | {si_str}+{sj_str} -> {sk_str}+{sl_str} | M={M_str} | {v_str}\n")

    print(f"Wrote txt to: {args.output}")

if __name__ == "__main__":
    main()

    #D:/软件/python.exe d:/文档/python/diaco1/analyze_results.py --input "D:\文档\python\V_pairs_k1.0.pkl" --output "D:\文档\python\diaco1\V_pairs_k1.0.txt" --bucket-summary