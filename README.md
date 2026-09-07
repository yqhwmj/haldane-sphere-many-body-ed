# Haldane Sphere Many-Body Exact Diagonalization

本项目用于研究 Haldane 球面上相互作用玻色子的多体量子体系，包含两体相互作用矩阵元计算、玻色 Fock 空间生成、多体哈
密顿量构建、稀疏/稠密对角化及低能态分析。

## Project status

当前版本 `v0.2.1`。相比 `v0.1.0` 的初始归档状态，本版本完成了三项基础设施改进：

- **路径可移植**：所有路径基于 `Path(__file__)` 推导为项目相对路径，不再依赖特定的工作目录或操作系统。
- **中间数据传递统一**：新增 `src/runio.py`，用 run 目录 + 元数据文件替代此前人工对齐的文件名约定。
- **运行参数可配置**：矩阵元计算参数改为终端交互式输入或命令行传入，不再写死在源码中。

物理计算部分未作改动，默认参数下结果与 `v0.1.0` 一致。已知的遗留问题仍记录在 `KNOWN_ISSUES.md`。

## Workflow

1. `src/matrix_element_calculator.py`：计算磁单极谐函数的两体相互作用矩阵元，并按参数自动归档为一个 run。
2. `src/many_body_hamiltonian.py`：交互式选择 run，读取矩阵元与参数，生成玻色 Fock 基，构建和对角化多体哈密顿量，
   并输出能谱、占有数与分析结果。

```text
physical parameters (interactive prompt or CLI flags)
 ↓
matrix_element_calculator.py
 ↓
data/runs/<params>__<run_id>/matrix_elements.pkl
 + matrix_elements.meta.json  (carries q, l_max, k_2D, ...)
 ↓
many_body_hamiltonian.py  (selects a run from a menu)
 ↓
results/q<q>_lmax<l_max>_k<k>_n<n_particles>/
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── KNOWN_ISSUES.md
├── src/
│   ├── matrix_element_calculator.py
│   ├── many_body_hamiltonian.py
│   └── runio.py                 # shared path + run management
├── data/
│   └── runs/
│       ├── manifest.json        # index of all historical runs
│       ├── latest.json          # pointer to the most recent run
│       └── q<q>_lmax<l>_k<k>__<run_id>/
│           ├── matrix_elements.pkl
│           └── matrix_elements.meta.json
└── results/
    └── q<q>_lmax<l>_k<k>_n<n>/
```

## Run-based data handoff

两个脚本不再靠文件名约定耦合，而是通过 run 目录传递数据：

- 每组参数（`q`、`l_max`、`k_2D`、`threshold`）生成唯一的 8 位 `run_id`，对应一个独立目录
  `data/runs/<params>__<run_id>/`。不同参数的结果并存，不会互相覆盖。
- run 目录内矩阵元使用**固定文件名** `matrix_elements.pkl`，脚本 2 无需重建文件名。
- 参数随数据保存在 `matrix_elements.meta.json` 中，脚本 2 直接读取，不存在两端不同步的可能。
- `manifest.json` 记录所有历史 run，`latest.json` 指向最近一次。

`n_particles` 不影响矩阵元，因此**不在** `run_id` 中。同一组矩阵元可以用不同粒子数反复计算，无需重跑最耗时的
矩阵元步骤。

## Environment

- Python 3
- NumPy
- SciPy
- SymPy

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

> 注意：矩阵元求导部分在 SymPy ≥ 1.13 下会因求导阶数为浮点而报错。建议在 `requirements.txt` 中将 SymPy 限制为
> `sympy>=1.12,<1.13`，或确保求导阶数已转为整数。

## Usage

### 1. 计算矩阵元

交互式设置参数（推荐）：

```bash
python src/matrix_element_calculator.py
```

终端会依次提示 `q`、`l_max`、`k_2D`、`threshold`、`chunk_size`、`num_workers`，直接回车使用默认值。半整数可写作
`3/2` 或 `1.5`。

命令行直接指定：

```bash
python src/matrix_element_calculator.py -q 5/2 -l 5/2 -w 8
python src/matrix_element_calculator.py -y          # 跳过交互，全部使用默认值
```

### 2. 构建并分析哈密顿量

```bash
python src/many_body_hamiltonian.py
```

终端先询问粒子数（单个值如 `4`，或逗号分隔的批量值如 `3,4,5`），随后列出所有可用 run 供选择：

```
可用的矩阵元数据：
  [0] q=  3_2  l_max=1.5  k=1  态数=  1234  2026-09-05T21:30:00  a3f9c2b1
  [1] q=  5_2  l_max=2.5  k=1  态数=  2871  2026-09-05T22:15:00  7d1e44aa  ← 最近
  直接回车 = 使用最近一次（7d1e44aa）
```

也可跳过交互：

```bash
python src/many_body_hamiltonian.py -n 5              # 指定粒子数
python src/many_body_hamiltonian.py a3f9c2b1 -n 5     # 指定 run 与粒子数
python src/many_body_hamiltonian.py --batch 3,4,5,6   # 批量扫粒子数
```

输出目录由输入参数自动命名（如 `results/q3_2_lmax1.5_k1_n4/`），不同配置的结果不会互相覆盖。

> 交互式输入需要真正的终端。在 VS Code 中请使用集成终端，或设置 `"code-runner.runInTerminal": true`。

Generated `.pkl`, `.npz` 和 result files are excluded from Git by default because they may be
machine-specific or large.

## Main methods

- monopole-harmonic state indexing;
- analytical and numerical evaluation of interaction matrix elements;
- bosonic occupation-number basis generation;
- sparse many-body Hamiltonian assembly;
- dense or sparse eigensolver selection;
- ground-state degeneracy, energy-gap, orbital-occupation and dominant-Fock-state analysis;
- multiprocessing for matrix-element and Hamiltonian construction;
- project-relative path resolution and run-based intermediate data management.

## Roadmap

- [x] Replace machine-specific paths with project-relative paths.
- [x] Connect the generated matrix-element filename to the Hamiltonian input automatically.
- [x] Move run parameters to a configuration file or command-line interface.
- [ ] Add small reproducible examples and automated tests.
- [ ] Pin verified dependency versions.
- [ ] Add benchmark results and physics validation documentation.

## Citation

If this repository supports a publication, citation information will be added after the
corresponding work is publicly available.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
