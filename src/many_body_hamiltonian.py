import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pickle
from collections import defaultdict
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys
from fractions import Fraction
import os
import traceback
import shutil
import os
import shutil
from pathlib import Path

# ====== 路径配置（与脚本1完全一致：基于文件自身位置推算）======
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"        # 脚本1 生成的矩阵元
RESULTS_DIR = PROJECT_ROOT / "results"  # 脚本输出的能谱、哈密顿量

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REAL = 1 # 修改相互作用强度
G_basis_states = None
G_state_to_index = None

def get_real_scale():
    return REAL

def set_real_scale(x: float):
    global REAL
    REAL = float(x)

def _init_worker_context(fock_pack):
    """
    每个子进程启动时调用一次，把大对象注入到子进程全局变量，避免每个任务重复序列化。
    """
    global G_basis_states, G_state_to_index
    G_basis_states = fock_pack['basis_states']
    G_state_to_index = fock_pack['state_to_index']

def _find_state_transitions_worker(i, j, k, l):
    """
    在子进程中使用全局上下文(G_basis_states/G_state_to_index),
    返回满足 i,j -> k,l 的 (initial_idx, final_idx) 列表。
    """
    state_pairs = []
    n_basis = len(G_basis_states)
    for initial_idx in range(n_basis):
        initial_occ = list(G_basis_states[initial_idx])
        
        # 出射条件
        if i == j:
            if initial_occ[i] < 2:
                continue
        else:
            if initial_occ[i] < 1 or initial_occ[j] < 1:
                continue
        
        # 应用跃迁: i,j -> k,l^+
        final_occ = list(initial_occ)
        final_occ[i] -= 1
        final_occ[j] -= 1
        final_occ[k] += 1
        final_occ[l] += 1
        if min(final_occ) < 0:
            continue
        
        final_idx = G_state_to_index.get(tuple(final_occ), -1)
        if final_idx != -1:
            state_pairs.append((initial_idx, final_idx))
    
    return state_pairs

def _boson_coeff_worker(initial_idx, final_idx, i, j, k, l):
    """
    玻色子统计因子(按“先湮灭、后产生”的次序)。
    使用全局的 G_basis_states。
    """
    initial_occ = list(G_basis_states[initial_idx])
    factor = 1.0
    intermediate_occ = initial_occ[:]
    # 湮灭
    if i == j:
        if intermediate_occ[i] >= 2:
            factor *= np.sqrt(intermediate_occ[i] * (intermediate_occ[i] - 1))
            intermediate_occ[i] -= 2
        else:
            return 0.0
    else:
        if intermediate_occ[i] >= 1 and intermediate_occ[j] >= 1:
            factor *= np.sqrt(intermediate_occ[i] * intermediate_occ[j])
            intermediate_occ[i] -= 1
            intermediate_occ[j] -= 1
        else:
            return 0.0
    # 产生
    if k == l:
        factor *= np.sqrt((intermediate_occ[k] + 1) * (intermediate_occ[k] + 2))
    else:
        factor *= np.sqrt((intermediate_occ[k] + 1) * (intermediate_occ[l] + 1))
    return factor

def _process_chunk_worker(chunk, interaction_scale=1.0, tmp_dir=None, drop_tol=1e-12):
    """
    子进程处理一个小块 chunk。生成局部稀疏矩阵并保存到 tmp_dir，返回文件路径。
    仅接收少量数据: [((i,j,k,l), V), ...]; 大上下文通过全局变量访问。
    """
    n_basis = len(G_basis_states) # 子进程内可见
    rows, cols, data = [], [], []
    for (i, j, k, l), V_val in chunk:
        # 0.5 因子用于去除 (i,j)<->(j,i) 等对称的重复计数
        V_real = 0.5 * interaction_scale * float(np.real(V_val))
        if abs(V_real) < drop_tol:
            continue

        for initial_idx, final_idx in _find_state_transitions_worker(i, j, k, l):
            coeff = _boson_coeff_worker(initial_idx, final_idx, i, j, k, l)
            if abs(coeff) < drop_tol:
                continue
            me = V_real * coeff
            if abs(me) > drop_tol:
                rows.append(final_idx)
                cols.append(initial_idx)
                data.append(me)

    # 生成局部块（合并重复项）
    if rows:
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        data = np.asarray(data, dtype=np.float64)
        coo = sp.coo_matrix((data, (rows, cols)), shape=(n_basis, n_basis))
        coo.sum_duplicates()
        block = coo.tocsr()
    else:
        block = sp.csr_matrix((n_basis, n_basis))

    # 确保临时目录存在
    tmp_dir = Path(tmp_dir) if tmp_dir else Path.cwd()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一文件名并保存
    fname = f"block_{os.getpid()}_{time.time_ns()}.npz"
    path = os.path.join(tmp_dir, fname)
    sp.save_npz(path, block)
    return path




class FockSpace:
    """Fock空间生成与管理类"""
    
    def __init__(self, n_particles, single_particle_states):
        """
        初始化Fock空间
        
        参数:
        n_particles: 粒子数
        single_particle_states: 单粒子态列表，每个态为(l, m)元组
        """
        self.n_particles = n_particles
        self.single_particle_states = single_particle_states
        self.n_orbitals = len(single_particle_states)
        self.basis_states = []
        self.state_to_index = {}
        self._generate_basis()

    def _generate_basis(self):
        """生成Fock空间基矢（占有数表示）"""
        from itertools import combinations_with_replacement
        
        orbital_indices = list(range(self.n_orbitals))
        all_combinations = combinations_with_replacement(orbital_indices, self.n_particles)
        
        for comb in all_combinations:
            occupation = [0] * self.n_orbitals    # 初始化全零数组
            for orb_idx in comb:
                occupation[orb_idx] += 1    # 在对应轨道上增加粒子数
            
            if sum(occupation) == self.n_particles:
                self.basis_states.append(tuple(occupation))
        
        for idx, state in enumerate(self.basis_states):
            self.state_to_index[state] = idx
        
        print(f"生成Fock空间基矢: {len(self.basis_states)}个态")
    
    def get_basis_size(self):
        """返回Fock空间维数"""
        return len(self.basis_states)
    
    def get_state_index(self, occupation):
        """返回给定占有数态的索引"""
        return self.state_to_index.get(tuple(occupation), -1)
    
    def get_occupation_numbers(self, state_idx):
        """返回索引对应态的占有数"""
        if 0 <= state_idx < len(self.basis_states):
            return list(self.basis_states[state_idx])
        return None
    
    def print_basis_info(self, max_display=10):
        """打印基矢信息"""
        print(f"\nFock空间信息:")
        print(f"- 粒子数: {self.n_particles}")
        print(f"- 单粒子轨道数: {self.n_orbitals}")
        print(f"- Fock空间维数: {self.get_basis_size()}")
        
        print(f"\n前{max_display}个基矢:")
        for i in range(min(max_display, len(self.basis_states))):
                occ = self.basis_states[i]
                orbital_info = []
                for orb_idx, n in enumerate(occ):
                    if n > 0:
                        l, m = self.single_particle_states[orb_idx]
                        orbital_info.append(f"轨道{orb_idx}(l={l},m={m}):{n}个粒子")
                print(f"态 {i}: {occ} -> {', '.join(orbital_info)}")

class ManyBodyHamiltonian:
    """多体哈密顿量构建器"""
    
    def __init__(self, n_particles, matrix_data_path, num_workers=None,
        output_dir=None, custom_filename=None,
        nnz_sparse_threshold=500000):
        """
        初始化多体哈密顿量构建器
        
        参数:
        n_particles: 粒子数
        matrix_data_path: 预计算矩阵元数据文件路径
        num_workers: 并行工作进程数
        """
        self.n_particles = n_particles
        self.matrix_data_path = matrix_data_path
        self.num_workers = num_workers if num_workers else mp.cpu_count()
        self.matrix_data = None
        self.V_matrix_dict = None
        self.index_to_state = None
        self.single_particle_states = None
        self.single_particle_energies = None
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.custom_filename = custom_filename
        self.nnz_sparse_threshold = nnz_sparse_threshold

        os.makedirs(self.output_dir, exist_ok=True)

        self.load_matrix_data()
        
        self.single_particle_states = [self.index_to_state[i] for i in range(len(self.index_to_state))]
        
        self.fock_space = FockSpace(n_particles, self.single_particle_states)
        
        self.calculate_single_particle_energies()
        
        self.H0 = None
        self.H_int = None
        self.H_full = None
        
        print(f"初始化完成，使用 {self.num_workers} 个并行工作进程")
    


    def load_matrix_data(self):
        """加载预计算的矩阵元数据"""
        try:
            with open(self.matrix_data_path, 'rb') as f:
                self.matrix_data = pickle.load(f)
                
            self.V_matrix_dict = self.matrix_data['matrix_elements']
            self.index_to_state = self.matrix_data['index_to_state']
            
            print(f"成功加载矩阵元数据:")
            print(f"  磁单极强度 q = {self.matrix_data['parameters']['q']}")
            print(f"  最大角量子数 l_max = {self.matrix_data['parameters']['l_max']}")
            print(f"  相互作用强度 k_2D = {self.matrix_data['parameters']['k_2D']}")
            print(f"  有效矩阵元数 = {len(self.V_matrix_dict)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到矩阵元数据文件: {self.matrix_data_path}")
        except Exception as e:
            raise RuntimeError(f"加载数据文件失败: {str(e)}")
    


    def generate_filename(self, file_type="hamiltonian", suffix=""):
        """生成包含参数信息的文件名"""
        
        
        if self.matrix_data is None:
            self.load_matrix_data()  
        
        # 从矩阵数据中获取参数
        parameters = self.matrix_data['parameters']
        q = parameters.get('q', 0)
        l_max = parameters.get('l_max', 0)
        k_2D = parameters.get('k_2D', 0)
        raw_k = float(parameters.get('k_2D', 1.0))
        k_2D = raw_k * get_real_scale()


        # 构建参数化的基础文件名
        if self.custom_filename:
            base_name = f"{self.custom_filename}_N{self.n_particles}_q{q}_lmax{l_max}_k{k_2D}"
        else:
            base_name = f"N{self.n_particles}_q{q}_lmax{l_max}_k{k_2D}"
        
        # 清理文件名中的特殊字符
        base_name = base_name.replace('.', 'p')
        base_name = base_name.replace(' ', '_')
        
        # 添加文件类型和后缀
        filename = f"{base_name}_{file_type}"
        if suffix:
            filename += f"_{suffix}"
        
        # 添加扩展名
        if file_type == "hamiltonian":
            filename += ".npz"
        elif file_type in ["eigenvalues", "eigenvectors", "occupation"]:
            filename += ".pkl"
        elif file_type == "summary":
            filename += ".txt"
        
        # 组合完整路径
        return self.output_dir / filename

    def calculate_single_particle_energies(self):
        """计算单粒子能级"""
        self.single_particle_energies = {}
        q = self.matrix_data['parameters']['q']
        
        for idx, state in self.index_to_state.items():
            l, m = state
            energy = l * (l + 1) - q**2
            self.single_particle_energies[state] = energy
        
        print(f"计算单粒子能级完成: {len(self.single_particle_energies)}个能级")
    
    def build_non_interacting_hamiltonian(self):
        """构建非相互作用部分哈密顿量（对角元）"""
        print("构建非相互作用哈密顿量...")
        n_basis = self.fock_space.get_basis_size()
        rows, cols, data = [], [], []
        
        for state_idx in range(n_basis):
            occupation = self.fock_space.get_occupation_numbers(state_idx)
            energy = 0.0 
            
            for orb_idx, n in enumerate(occupation):
                if n > 0:
                    state = self.single_particle_states[orb_idx]
                    energy += n * self.single_particle_energies.get(state, 0.0)
            
            rows.append(state_idx)
            cols.append(state_idx)
            data.append(energy)
        
        self.H0 = sp.csr_matrix((data, (rows, cols)), shape=(n_basis, n_basis))
        print(f"非相互作用哈密顿量构建完成，非零元: {self.H0.nnz}")
        return self.H0
    
    def _calculate_matrix_element_coefficient(self, initial_idx, final_idx, i, j, k, l):
        """
        计算矩阵元系数（玻色子统计因子）
        已添加边界检查确保物理正确性
        """
        initial_occ = self.fock_space.get_occupation_numbers(initial_idx)
        final_occ = self.fock_space.get_occupation_numbers(final_idx)
        
        factor = 1.0
        
        intermediate_occ = initial_occ.copy()

        if i == j:
            if intermediate_occ[i] >= 2:
                factor *= np.sqrt(intermediate_occ[i] * (intermediate_occ[i] - 1))
                intermediate_occ[i] -= 2
            else:
                return 0.0
        else:
            if intermediate_occ[i] >= 1 and intermediate_occ[j] >= 1:
                factor *= np.sqrt(intermediate_occ[i] * intermediate_occ[j])
                intermediate_occ[i] -= 1  
                intermediate_occ[j] -= 1  
            else:
                return 0.0
        

        if k == l:
            factor *= np.sqrt((intermediate_occ[k] + 1) * (intermediate_occ[k] + 2))
        else:
            factor *= np.sqrt((intermediate_occ[k] + 1) * (intermediate_occ[l] + 1))
        
        return factor

    
    def build_interaction_hamiltonian(self, chunk_size=100, drop_tol=1e-12):
        """构建相互作用部分哈密顿量（模块级 worker + initializer 注入上下文）"""
        print(f"构建相互作用哈密顿量（并行处理，块大小: {chunk_size}）...")
        start_time = time.time()
        
        n_basis = self.fock_space.get_basis_size()
        
        # 为本次构建创建临时目录
        tmp_dir = self.output_dir / f"tmp_blocks_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # 取出预计算矩阵元
        matrix_items = list(self.V_matrix_dict.items())
        if not matrix_items:
            print("警告: 矩阵元数据为空，返回零矩阵 H_int")
            self.H_int = sp.csr_matrix((n_basis, n_basis))
            return self.H_int
        
        chunks = [matrix_items[i:i + chunk_size] for i in range(0, len(matrix_items), chunk_size)]
        print(f"将 {len(matrix_items)} 个矩阵元分成 {len(chunks)} 个块进行并行处理")
        
        # 有效相互作用强度：文件参数 × 外部扫描比例
        raw_scale = float(self.matrix_data['parameters'].get('k_2D', 1.0))
        interaction_scale = raw_scale * get_real_scale()
        
        # 打包一次上下文
        fock_pack = {
            'basis_states': self.fock_space.basis_states,
            'state_to_index': self.fock_space.state_to_index
        }
        
        # 累加器
        H_acc = sp.csr_matrix((n_basis, n_basis))
        
        try:
            if len(chunks) == 1 or (self.num_workers is not None and self.num_workers <= 1):
                # 串行分支
                _init_worker_context(fock_pack)
                path = _process_chunk_worker(chunks[0], interaction_scale, tmp_dir, drop_tol)
                block = sp.load_npz(path).tocsr()
                H_acc = H_acc + block
                try:
                    os.remove(path)
                except OSError:
                    pass
            else:
                max_workers = self.num_workers if self.num_workers else mp.cpu_count()
                max_workers = max(1, min(max_workers, len(chunks)))
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker_context,
                    initargs=(fock_pack,)
                ) as executor:
                    futures = {
                        executor.submit(_process_chunk_worker, c, interaction_scale, tmp_dir, drop_tol): idx
                        for idx, c in enumerate(chunks)
                    }
                    completed = 0
                    report_step = max(1, len(chunks) // 10)
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            path = fut.result()
                        except Exception as e:
                            print(f"处理块 {idx} 异常，改为主进程串行。异常: {e}")
                            traceback.print_exc()
                            _init_worker_context(fock_pack)
                            path = _process_chunk_worker(chunks[idx], interaction_scale, tmp_dir, drop_tol)
                        
                        block = sp.load_npz(path).tocsr()
                        H_acc = H_acc + block
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                        
                        completed += 1
                        if completed % report_step == 0 or completed == len(chunks):
                            print(f"处理进度: {completed}/{len(chunks)} 块")
        except Exception as e:
            print(f"进程池初始化/调度异常，整体改为串行。异常: {e}")
            traceback.print_exc()
            _init_worker_context(fock_pack)
            for c in chunks:
                path = _process_chunk_worker(c, interaction_scale, tmp_dir, drop_tol)
                block = sp.load_npz(path).tocsr()
                H_acc = H_acc + block
                try:
                    os.remove(path)
                except OSError:
                    pass
        finally:
            # 清理临时目录（如为空或即使有残留也尝试删除）
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        
        self.H_int = H_acc
        
        elapsed = time.time() - start_time
        rate = len(matrix_items) / elapsed if elapsed > 0 else float('inf')
        print("相互作用哈密顿量构建完成!")
        print(f" 非零元: {self.H_int.nnz}")
        print(f" 处理时间: {elapsed:.2f} 秒")
        print(f" 处理速率: {rate:.2f} 矩阵元/秒")
        return self.H_int


    
    def build_full_hamiltonian(self):
        """构建完整哈密顿量"""
        if self.H0 is None:
            self.build_non_interacting_hamiltonian()
        if self.H_int is None:
            self.build_interaction_hamiltonian()
        
        self.H_full = self.H0 + self.H_int
        
        print(f"完整哈密顿量构建完成!")
        print(f"  矩阵维度: {self.H_full.shape}")
        print(f"  非零元素数: {self.H_full.nnz}")
        print(f"  稀疏度: {self.H_full.nnz / (self.fock_space.get_basis_size()**2) * 100:.6f}%")
        
        return self.H_full
    
    def diagonalize(self, k=10, degeneracy_tolerance=1e-8, relative_tolerance=0.001):
        """
        对角化哈密顿量（使用稀疏矩阵算法）
        使用双重容差标准（绝对容差和相对容差）判断简并
        
        参数:
        k: 计算的本征值数量
        degeneracy_tolerance: 绝对容差
        relative_tolerance: 相对容差
        """
        if self.H_full is None:
            self.build_full_hamiltonian()
        
        # 读取参数并判定是否 LLL（l_max == q）
        params = self.matrix_data.get('parameters', {})
        q = params.get('q', 0.0)
        l_max = params.get('l_max', 0.0)
        is_lll = abs(l_max - q) < 1e-12  # 浮点容差

        N = self.H_full.shape[0]
        nnz_count = self.H_full.nnz if sp.issparse(self.H_full) else int(np.count_nonzero(self.H_full))
        force_sparse_due_to_nnz = nnz_count > getattr(self, 'nnz_sparse_threshold', 500000)

        if is_lll and not force_sparse_due_to_nnz:
            print(f"开始对角化（LLL 且 nnz={nnz_count} <= 阈值 {self.nnz_sparse_threshold}，使用稠密矩阵，完整谱）...")
            H_dense = self.H_full.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
        else:
            reason = "LLL 但 nnz 超阈值" if is_lll else "非 LLL"
            k_use = min(k, N - 1) # eigsh 需要 k < N
            print(f"开始对角化（{reason}，使用稀疏算法，目标前 {k_use} 个本征值；nnz={nnz_count}，阈值={self.nnz_sparse_threshold}）...")
            eigenvalues, eigenvectors = spla.eigsh(self.H_full, k=k_use, which='SA', return_eigenvectors=True)

        # 排序并继续你原来的后处理（简并判定、打印等）
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        

        print(f"对角化完成，找到{len(eigenvalues_sorted)}个本征值")
        print(f"基态能量: {eigenvalues_sorted[0]:.6f}")
        
        # 前十个本征值具体数值
        num_to_print = min(20, len(eigenvalues_sorted))
        print(f"\n前{num_to_print}本征值具体数值:")
        for i in range(num_to_print):
            print(f"  本征值 {i}: {eigenvalues_sorted[i]:.12f}")

        # 使用双重容差标准判断简并
        ground_energy = eigenvalues_sorted[0]
        first_excited_index = 1
        
        # 跳过所有简并的基态，找到第一激发态
        while first_excited_index < len(eigenvalues_sorted):
            energy_diff = abs(eigenvalues_sorted[first_excited_index] - ground_energy)
            
            # 计算相对差异（避免除零）
            if abs(ground_energy) < 1e-12:
                relative_diff = energy_diff
            else: 
                relative_diff = energy_diff / abs(ground_energy)
            
            # 满足任一容差即视为简并
            if energy_diff > degeneracy_tolerance or relative_diff > relative_tolerance:
                break
                
            first_excited_index += 1
        
        # 处理所有能级都简并的特殊情况
        if first_excited_index >= len(eigenvalues_sorted):
            first_excited_energy = ground_energy
            gap = 0.0
            print("警告: 所有计算能级都简并，系统可能为gapless")
        else:
            first_excited_energy = eigenvalues_sorted[first_excited_index]
            gap = first_excited_energy - ground_energy
        
        print(f"第一激发态能量: {first_excited_energy:.6f}")
        print(f"能隙: {gap:.6f}")
        
        # 计算简并度
        degeneracy = first_excited_index  # 简并基态的数量
        if degeneracy > 1:
            print(f"基态简并: {degeneracy}")
        else:
            print("基态非简并")
        
        # 创建去简并化的本征值数组（也使用双重容差标准）
        unique_energies = []
        current_energy = eigenvalues_sorted[0]
        unique_energies.append(current_energy)
        
        for i in range(1, len(eigenvalues_sorted)):
            energy_diff = abs(eigenvalues_sorted[i] - current_energy)
            
            if abs(current_energy) < 1e-12:
                relative_diff = energy_diff
            else:
                relative_diff = energy_diff / abs(current_energy)
            
            if energy_diff > degeneracy_tolerance or relative_diff > relative_tolerance:
                current_energy = eigenvalues_sorted[i]
                unique_energies.append(current_energy)
        
        print(f"去简并化能级数: {len(unique_energies)}")
        
        return eigenvalues_sorted, eigenvectors_sorted, unique_energies
    
    def compute_orbital_occupations(self, state_vector):
        """计算单粒子轨道的期望占有数"""
        n_orbitals = self.fock_space.n_orbitals
        orbital_occupations = np.zeros(n_orbitals)
        
        for state_idx, amplitude in enumerate(state_vector):
            prob = np.abs(amplitude)**2     #Fock态出现的概率
            occupation_numbers = self.fock_space.get_occupation_numbers(state_idx)     #Fock态的具体占有数分布
            
            #对每个轨道，累加所有Fock态的贡献
            for orb_idx in range(n_orbitals):
                orbital_occupations[orb_idx] += occupation_numbers[orb_idx] * prob
        
        return orbital_occupations

    def identify_dominant_fock_states(self, state_vector, top_k=5):
        """识别对量子态贡献最大的前k个Fock态"""
        components = []
        
        for state_idx, amplitude in enumerate(state_vector):
            weight = np.abs(amplitude)**2
            occupation = self.fock_space.get_occupation_numbers(state_idx)
            
            components.append({
                'weight': weight,
                'amplitude': amplitude,
                'occupation': occupation,
                'state_idx': state_idx
            })
        
        # 按权重降序排列
        components.sort(key=lambda x: x['weight'], reverse=True)
        
        print(f"\n主要Fock态贡献 (前{top_k}个):")
        print("=" * 60)
        for i, comp in enumerate(components[:top_k]):
            occ_str = self._format_occupation(comp['occupation'])
            print(f"{i+1}. 权重: {comp['weight']:.4f} | 振幅: {comp['amplitude']:.4f}")
            print(f"   占有数分布: {occ_str}")
            print(f"   Fock态索引: {comp['state_idx']}")
            print("-" * 40)
        
        return components[:top_k]

    def _compute_participation_ratio(self, state_vector):
        """计算参与率：1/∑|c_i|⁴，度量量子态复杂性"""
        weights = np.abs(state_vector)**2
        participation = 1.0 / np.sum(weights**2)  # ∑|c_i|⁴ = ∑(weights²)
        return participation

    def _format_occupation(self, occupation):
        """格式化占有数分布为字符串"""
        parts = []
        for orb_idx, n in enumerate(occupation):
            if n > 0:
                l, m = self.single_particle_states[orb_idx]
                parts.append(f"{n}@({l},{m})")
        return "[" + " ".join(parts) + "]"

    def save_hamiltonian(self, filename=None, suffix=""):
        """保存哈密顿量到文件，支持自定义文件名"""
        if self.H_full is None:
            self.build_full_hamiltonian()
        
        if filename is None:
            filename = self.generate_filename("hamiltonian", suffix)
        
        sp.save_npz(filename, self.H_full)
        print(f"哈密顿量已保存至: {filename}")
        return filename

    def save_eigen_data(self, eigenvalues, eigenvectors, suffix=""):
        """保存本征数据，包含完整的参数信息"""
        
        
        if self.matrix_data is None:
            self.load_matrix_data()  
        
        filename = self.generate_filename("eigenvalues", suffix)
        raw_k = float(self.matrix_data['parameters']['k_2D'])
        # 构建包含完整参数信息的数据结构
        data = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'basis_states': self.fock_space.basis_states, 
            'single_particle_states': self.single_particle_states,
            'simulation_parameters': {
                
           
                'k_2D': raw_k * get_real_scale(), 
                'n_particles': self.n_particles,
                'q': self.matrix_data['parameters']['q'],
                'Nphi': int(2 * self.matrix_data['parameters']['q']), 
                'l_max': self.matrix_data['parameters']['l_max'], 
                'matrix_data_path': self.matrix_data_path,
                'n_orbitals': self.fock_space.n_orbitals, 
                'basis_size': self.fock_space.get_basis_size() 
            },
            'timestamp': time.time(),
            'timestamp_readable': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_version': '1.1' 
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"本征数据已保存至: {filename}")
        return filename

    def save_occupation_data(self, occupation_data, suffix=""):
        """保存占有数数据"""
        filename = self.generate_filename("occupation", suffix)
        
        with open(filename, 'wb') as f:
            pickle.dump(occupation_data, f)
        
        print(f"占有数数据已保存至: {filename}")
        return filename

    def save_analysis_summary(self, analysis_results, suffix=""):
        """保存分析总结"""
        filename = self.generate_filename("summary", suffix)
        raw_k = float(self.matrix_data['parameters'].get('k_2D', 1.0))
        effective_k = raw_k * get_real_scale()

        with open(filename, 'w') as f:
            f.write("多体系统分析总结\n")
            f.write("=" * 50 + "\n")
            f.write(f"粒子数: {self.n_particles}\n")
            f.write(f"磁单极强度 q: {self.matrix_data['parameters']['q']}\n")
            f.write(f"最大角量子数 l_max: {self.matrix_data['parameters']['l_max']}\n")
            f.write(f"相互作用强度 k_2D: {effective_k}\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in analysis_results.items(): 
                f.write(f"{key}: {value}\n")
        
        print(f"分析总结已保存至: {filename}") 
        return filename 

def enhanced_example_usage():
    
    n_particles = 2
    matrix_data_path = DATA_DIR / "V_matrix_q1_lmax1_k1.pkl"
    num_workers = min(os.cpu_count() or 1, 16)
    
    output_directory = RESULTS_DIR   # 自定义保存目录
    custom_filename = ""  # 自定义文件名前缀

    print("=" * 60)
    print("多体哈密顿量构建模块") 
    print("=" * 60)
    
    # 初始化多体哈密顿量构建器
    mb_hamiltonian = ManyBodyHamiltonian(n_particles, matrix_data_path, num_workers,
        output_dir=output_directory,
        custom_filename=custom_filename,
        nnz_sparse_threshold=500000)
    
    # 显示Fock空间信息
    mb_hamiltonian.fock_space.print_basis_info(max_display=5)
    
    # 构建完整哈密顿量
    H_full = mb_hamiltonian.build_full_hamiltonian()
    
    # 对角化并获取本征值和本征向量
    eigenvalues, eigenvectors, unique_energies = mb_hamiltonian.diagonalize(
        k=20, 
        degeneracy_tolerance=0.001,  # 绝对容差
        relative_tolerance=0.001     # 相对容差
    )
    
    # 保存所有数据（使用参数化文件名）
    hamiltonian_file = mb_hamiltonian.save_hamiltonian(suffix="full")
    eigen_file = mb_hamiltonian.save_eigen_data(eigenvalues, eigenvectors)

    analysis_results = {
        "基态能量": eigenvalues[0],
        "能隙": eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0,
        "简并度": len([e for e in eigenvalues if abs(e - eigenvalues[0]) < 0.001]),
        "总态数": len(eigenvalues)
    }
    
    summary_file = mb_hamiltonian.save_analysis_summary(analysis_results)



    # 详细输出本征值信息
    print(f"\n【本征值详细分析】")
    print("-" * 40)
    print(f"计算得到的本征值数量: {len(eigenvalues)}")
    print(f"去简并化后的能级数量: {len(unique_energies)}")

    # 基态能量分析
    ground_energy = unique_energies[0]
    first_excited_energy = unique_energies[1] if len(unique_energies) > 1 else ground_energy
    gap = first_excited_energy - ground_energy
    
    print(f"\n【基态能量分析】")
    print("-" * 40)
    print(f"基态能量: {ground_energy:.8f}")
    print(f"第一激发态能量: {first_excited_energy:.8f}")
    print(f"能隙: {gap:.8f}")
    
    # ========== 简并基态分析 ==========
    print(f"\n【简并基态占有数分析】")
    print("-" * 40)
    
    # 1. 找出所有简并基态
    ground_energy = eigenvalues[0]
    degenerate_indices = [0]  # 基态总是包含在内
    degeneracy_tolerance = 1e-12
    
    for i in range(1, len(eigenvalues)):
        energy_diff = abs(eigenvalues[i] - ground_energy)
        if energy_diff <= degeneracy_tolerance:
            degenerate_indices.append(i)
        else:
            break  # 遇到第一个非简并态就停止
    
    print(f"发现 {len(degenerate_indices)} 个简并基态")
    print(f"简并基态索引: {degenerate_indices}")
    
    # 2. 计算平均占有数分布
    n_orbitals = mb_hamiltonian.fock_space.n_orbitals     #确定单粒子轨道的总数
    avg_occupation = np.zeros(n_orbitals)                 #创建全零数组，用于累加各态的占有数
    individual_occupations = []                           #存储每个简并态的独立占有数分布
    
    for idx in degenerate_indices:
        state_vector = eigenvectors[:, idx]
        occupation = mb_hamiltonian.compute_orbital_occupations(state_vector)
        individual_occupations.append(occupation)
        avg_occupation += occupation
    
    avg_occupation /= len(degenerate_indices)
    
    # 3. 输出结果
    print(f"简并基态平均占有数分布 ({len(degenerate_indices)}个态平均):")
    significant_count = 0
    for orb_idx, occ in enumerate(avg_occupation):
        if occ > 0.001:
            l, m = mb_hamiltonian.single_particle_states[orb_idx]
            print(f"  轨道{orb_idx} (l={l}, m={m}): {occ:.4f}")
            significant_count += 1
    
    if significant_count == 0:
        print("  (无显著占据轨道)")
    
    # ========== 多态分析部分 ==========
    print(f"\n【多低能态占有数分析 】")
    print("=" * 70)
    
    num_states_to_analyze = 1
    
    # 分析每个低能态
    for state_idx in range(num_states_to_analyze):
        print(f"\n{'#'*60}")
        print(f"分析第 {state_idx} 个低态 (能量: {eigenvalues[state_idx]:.8f})")
        if state_idx == 0:
            print("【基态分析】")
        else:
            energy_gap = eigenvalues[state_idx] - eigenvalues[0]
            print(f"【第{state_idx}激发态】能隙: {energy_gap:.6f}")
        print(f"{'#'*60}")
        
        # 获取该态的波函数
        state_vector = eigenvectors[:, state_idx]
        
        # 分析该态的占有数分布
        occupation_distribution = mb_hamiltonian.compute_orbital_occupations(state_vector)
        
        print(f"单粒子轨道期望占有数分布:")
        significant_count = 0
        for orb_idx, occ in enumerate(occupation_distribution): 
            if occ > 0.001:  # 只显示显著占据的轨道
                l, m = mb_hamiltonian.single_particle_states[orb_idx]
                print(f"  轨道{orb_idx} (l={l}, m={m}): {occ:.4f}")
                significant_count += 1
        
        if significant_count == 0:
            print("  (无显著占据轨道)")
         
        # 分析该态的主要Fock成分
        dominant_states = mb_hamiltonian.identify_dominant_fock_states(
            state_vector, top_k=3
        )
        
        # 计算参与率
        participation_ratio = mb_hamiltonian._compute_participation_ratio(state_vector)
        print(f"参与率: {participation_ratio:.2f} ")            #度量态复杂性
     
  
     
    print("\n完整分析完成!")

if __name__ == "__main__":
    mp.freeze_support()
    enhanced_example_usage()