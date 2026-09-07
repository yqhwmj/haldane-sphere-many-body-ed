import numpy as np
import math
import pickle
import time
import warnings
from collections import OrderedDict
from scipy.special import gammaln, factorial
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import sys
from fractions import Fraction
from sympy import symbols, diff, lambdify
from scipy.integrate import quad
import os
from pathlib import Path

# 项目根目录 = src/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 各个目录
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# 自动创建（不存在就建，存在就跳过）
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')

def compute_normalization(l, m, q):
    """
    根据文档中的公式计算归一化因子 M_{lm}
    M_{lm} = [(-1)^n / 2^l * n!] * sqrt((2l+1)/(4π) * (l-m)!*(l+m)! / ((l-q)!*(l+q)!)
    其中 n = l + m
    """
    l_float = float(l)
    m_float = float(m)
    q_float = float(q)
    
    if l_float < abs(q_float) or abs(m_float) > l_float:
        return 0.0
    
    try:
        n = l_float + m_float
        
        # 计算 (-1)^n / 2^l * n!
        sign_factor = (-1)**n
        denominator = (2**l_float) * math.factorial(int(n))
        prefactor = sign_factor / denominator
        
        # 计算 sqrt((2l+1)/(4π) * (l-m)!*(l+m)! / ((l-q)!*(l+q)!)
        numerator = (2*l_float + 1) * math.factorial(int(l_float - m_float)) * math.factorial(int(l_float + m_float))
        denominator_frac = 4 * math.pi * math.factorial(int(l_float - q_float)) * math.factorial(int(l_float + q_float))
        sqrt_factor = math.sqrt(numerator / denominator_frac)
        
        M_lm = prefactor * sqrt_factor
        return M_lm
    except Exception as e:
        print(f"归一化计算错误: l={l}, m={m}, q={q}, 错误: {e}")
        return 0.0

def compute_analytical_V(q, l1, m1, l2, m2, l3, m3, l4, m4, k_2D):
    """
    计算相互作用矩阵元 V_{l1m1l2m2, l3m3l4m4}
    """
    # 检查角动量投影守恒
    if abs(m1 + m2 - m3 - m4) > 1e-10:
        return 0.0
    
    # 计算归一化因子
    M1 = compute_normalization(l1, m1, q)
    M2 = compute_normalization(l2, m2, q)
    M3 = compute_normalization(l3, m3, q)
    M4 = compute_normalization(l4, m4, q)
    if M1 == 0 or M2 == 0 or M3 == 0 or M4 == 0:
        return 0.0
    
    # 计算参数
    n1 = l1 + m1
    n2 = l2 + m2
    n3 = l3 + m3
    n4 = l4 + m4
    m_t = (m1 + m2 + m3 + m4) / 2  # m_t = Σ_i m_i / 2

    # 使用SymPy进行符号计算
    x = symbols('x')
    
    # 计算导数乘积项 ∏_i d^{n_i}/dx^{n_i} [(1-x)^{l_i-q} (1+x)^{l_i+q}]
    derivative_product = 1
    for l_i, n_i in zip([l1, l2, l3, l4], [n1, n2, n3, n4]):
        base_expr = (1 - x)**(l_i - q) * (1 + x)**(l_i + q)
        deriv_expr = diff(base_expr, x, n_i)
        derivative_product *= deriv_expr
    
    # 乘以权重因子 (1-x)^{2q+m_t} (1+x)^{-2q+m_t}
    weight_factor = (1 - x)**(2*q + m_t) * (1 + x)**(-2*q + m_t)
    integrand_expr = weight_factor * derivative_product
    
    # 将符号表达式转换为数值函数
    integrand_func = lambdify(x, integrand_expr, 'numpy')
    
    # 数值积分从-1到1
    try:
        integral, error = quad(integrand_func, -1, 1, points=[-1, 1], limit=100, epsabs=1e-10, epsrel=1e-10)
    except Exception as e:
        print(f"积分错误: l1={l1}, m1={m1}, l2={l2}, m2={m2}, l3={l3}, m3={m3}, l4={l4}, 错误: {e}")
        return 0.0
    
    # 组合因子：2π k_2D δ_{m1+m2,m3+m4} M1 M2 M3 M4 * 积分
    V_val = 2 * math.pi * k_2D * M1 * M2 * M3 * M4 * integral
    return V_val

def compute_chunk(args_chunk, k_2D, valid_states, q, threshold):
    """
    处理任务块 - 使用解析表达式计算矩阵元
    """
    results = {}
    
    for i, j, k, l in args_chunk:
        l1, m1 = valid_states[i]
        l2, m2 = valid_states[j]
        l3, m3 = valid_states[k]
        l4, m4 = valid_states[l]
        
        V_val = compute_analytical_V(q, l1, m1, l2, m2, l3, m3, l4, m4, k_2D)
        
        if abs(V_val) > threshold:
            results[(i, j, k, l)] = V_val
            
    return results

class StateIndexMapper:
    """
    态映射系统：管理量子态与索引的映射关系
    """
    def __init__(self, q, l_max):
        self.q = q
        self.l_max = l_max
        self.l_min = abs(q)
        self._total_states = self._calculate_total_states()
        self._valid_states = self._generate_valid_states()
        
    def _calculate_total_states(self):
        """计算总态数：Σ_{l=l_min}^{l_max} (2l+1)"""
        total = 0
        l = self.l_min
        while l <= self.l_max:
            total += int(2*l + 1)
            l += 1
        return total
        
    def _generate_valid_states(self):
        """生成所有有效量子态列表"""
        states = []
        l = self.l_min
        while l <= self.l_max:
            m = -l
            while m <= l:
                states.append((l, m))
                m += 1
            l += 1
        return states
        
    def get_index(self, l, m):
        """通过公式计算索引：(l, m) → index"""
        if not (self.l_min <= l <= self.l_max and -l <= m <= l):
            return None    
        
        # 计算在l之前的态总数
        states_before = 0
        l_prime = self.l_min
        while l_prime < l:
            states_before += int(2*l_prime + 1)
            l_prime += 1
        
        # 当前l内的偏移
        offset = int(m + l)
        return states_before + offset
        
    def get_state(self, index):
        """通过公式计算量子态：index → (l, m)"""
        if index < 0 or index >= self._total_states:
            return None
        
        l = self.l_min
        remaining_index = index
        while l <= self.l_max:
            states_in_l = int(2*l + 1)
            if remaining_index < states_in_l:
                m = remaining_index - l
                return (l, m)
            remaining_index -= states_in_l
            l += 1
        
        return None
        
    def total_states(self):
        """返回总态数"""
        return self._total_states
        
    def get_valid_states(self):
        """获取所有有效态列表"""
        return self._valid_states

def format_state(state):
    """格式化量子态为(l, m)字符串"""
    l, m = state
    # 半整数特殊处理
    if abs(l - round(l)) < 1e-6:
        l_str = f"{int(l)}" if l.is_integer() else f"{int(2*l)}/2"
    else:
        l_str = f"{l:.2f}"
    
    if abs(m - round(m)) < 1e-6:
        m_str = f"{int(m)}" if m.is_integer() else f"{int(2*m)}/2"
    else:
        m_str = f"{m:.2f}"
    
    return f"({l_str}, {m_str})"

def main():
    # ==================== 参数设置 ====================
    q = Fraction(2, 2)      # 磁单极强度
    l_max = 1          # 最大角量子数
    k_2D = 1              # 二维相互作用强度
    threshold = 1e-10       # 矩阵元阈值：绝对值小于此值的矩阵元将被忽略
    
    chunk_size = 100        # 并行计算任务块大小
    num_workers = 40       # 并行工作进程数
    
    print("开始计算磁单极谐函数相互作用矩阵元...")
    print(f"参数设置: q={q}, l_max={l_max}, k_2D={k_2D}")
    print(f"矩阵元阈值: {threshold} (绝对值小于此值的矩阵元将被忽略)")
    start_time = time.time()
    
    # ==================== 初始化态映射系统 ====================
    mapper = StateIndexMapper(q, l_max)
    num_states = mapper.total_states()
    valid_states = mapper.get_valid_states()
    print(f"创建态映射器完成: 总态数 = {num_states}")
    
    # ==================== 生成计算任务 ====================
    print("\n生成计算任务（筛选角动量守恒的组合）...")
    task_chunks = []
    total_combinations = 0
    valid_combinations = 0
    
    # 筛选满足角动量投影守恒的组合
    for i in range(num_states):
        _, m1 = valid_states[i]
        for j in range(num_states):
            _, m2 = valid_states[j]
            for k in range(num_states):
                _, m3 = valid_states[k]
                for l_idx in range(num_states):
                    _, m4 = valid_states[l_idx]
                    total_combinations += 1
                    
                    # 检查角动量投影守恒
                    if abs(m1 + m2 - m3 - m4) < 1e-10:
                        task_chunks.append((i, j, k, l_idx))
                        valid_combinations += 1
    
    print(f"总组合数: {total_combinations}")
    print(f"角动量守恒组合数: {valid_combinations}")
    
    # ==================== 并行计算矩阵元 ====================
    print("开始并行计算矩阵元（使用解析表达式）...")
    comp_start = time.time()
    
    V_matrix_dict = {}
    completed_chunks = 0
    total_chunks = (len(task_chunks) + chunk_size - 1) // chunk_size
    
    # 使用进程池并行计算
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(task_chunks), chunk_size):
            chunk = task_chunks[i:i+chunk_size]
            futures.append(executor.submit(
                compute_chunk, chunk, k_2D, valid_states, q, threshold
            ))
        
        # 收集结果
        for future in futures:
            V_matrix_dict.update(future.result())
            completed_chunks += 1
            print(f"已完成 {completed_chunks}/{total_chunks} 任务块")
    
    comp_time = time.time() - comp_start
    valid_matrix_elements = len(V_matrix_dict)
    
    # ==================== 保存结果 ====================
    q_str = str(q).replace('/', '_')
    output_filename = f"V_matrix_q{q_str}_lmax{l_max}_k{k_2D}.pkl"
    output_path = DATA_DIR / output_filename

    with open(output_path, 'wb') as f:
        pickle.dump({
            'parameters': {
                'q': float(q),
                'l_max': l_max,
                'k_2D': k_2D,
                'threshold': threshold
            },
            'matrix_elements': V_matrix_dict,
            'index_to_state': {i: valid_states[i] for i in range(num_states)},
            'statistics': {
                'total_combinations': total_combinations,
                'momentum_conserved': valid_combinations,
                'computed_elements': valid_matrix_elements,
                'filtered_out': valid_combinations - valid_matrix_elements,
                'filter_ratio': (valid_combinations - valid_matrix_elements) / valid_combinations if valid_combinations > 0 else 0,
                'total_time': time.time() - start_time,
                'computation_time': comp_time
            }
        }, f)

    print(f"矩阵元已保存至：{output_path}")
    # ==================== 输出总结 ====================
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n计算完成! 结果摘要:")
    print("=" * 60)
    print(f"总组合数: {total_combinations}")
    print(f"角动量守恒组合数: {valid_combinations}")
    print(f"有效矩阵元数 (|V| > {threshold}): {valid_matrix_elements}")
    print(f"被过滤矩阵元数: {valid_combinations - valid_matrix_elements}")
    print(f"过滤比例: {(valid_combinations - valid_matrix_elements) / valid_combinations * 100:.2f}%")
    print(f"矩阵元计算时间: {comp_time:.2f}s")
    print(f"总运行时间: {elapsed_time:.2f}s")
    print(f"\n结果已保存至: {output_filename}")
    
    # 显示矩阵元详细列表
    if valid_matrix_elements > 0:
        print("\n矩阵元详细列表 (前5个有效矩阵元):")
        print("=" * 80)
        print(f"{'索引组合':<14} | {'量子态组合 (初态 → 末态)':<45} | {'矩阵元值':>15}")
        print("-" * 80)
        
        displayed_count = 0
        for key, value in list(V_matrix_dict.items())[:10]:
            states = [valid_states[k] for k in key]
            initial_states = states[2:]
            final_states = states[:2]
            
            initial_str = " + ".join([format_state(s) for s in initial_states])
            final_str = " + ".join([format_state(s) for s in final_states])
            state_str = f"{initial_str} → {final_str}"
            
            # 只显示实部
            val_str = f"{value.real:.6f}"
            print(f"{str(key):<14} | {state_str:<45} | {val_str:>15}")
            displayed_count += 1
        
        if valid_matrix_elements > 5:
            print(f"... (还有 {valid_matrix_elements - 5} 个矩阵元未显示)")
        print("=" * 80)
    else:
        print("\n警告: 未找到任何有效矩阵元!")

if __name__ == "__main__":
    main()