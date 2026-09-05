# Haldane Sphere Many-Body Exact Diagonalization
本项目用于研究 Haldane 球面上相互作用玻色子的多体量子体系，包含两体相互作用矩阵元计算、玻色 Fock 空间生成、多体哈
密顿量构建、稀疏/稠密对角化及低能态分析。
## Project status
当前版本 `v0.1.0` 是科研代码的初始归档版本。为保留原始计算状态，本版本暂未统一硬编码路径、输入输出文件名和运行参
数。复现前请阅读 `KNOWN_ISSUES.md`。后续改进将通过独立分支和版本提交完成。
## Workflow
1. `src/matrix_element_calculator.py`：计算磁单极谐函数的两体相互作用矩阵元，并保存为 Pickle 数据文件。
2. `src/many_body_hamiltonian.py`：读取矩阵元数据，生成玻色 Fock 基，构建和对角化多体哈密顿量，并输出能谱、
占有数与分析结果。
```text
physical parameters
 ↓
matrix_element_calculator.py
 ↓
interaction matrix elements (.pkl)
 ↓
many_body_hamiltonian.py
 ↓
Hamiltonian and eigensystem results
```
## Repository structure
```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── KNOWN_ISSUES.md
├── src/
│ ├── matrix_element_calculator.py
│ └── many_body_hamiltonian.py
├── data/
└── results/
```
## Main methods
- monopole-harmonic state indexing;
- analytical and numerical evaluation of interaction matrix elements;
- bosonic occupation-number basis generation;
- sparse many-body Hamiltonian assembly;
- dense or sparse eigensolver selection;
- ground-state degeneracy, energy-gap, orbital-occupation and dominant-Fock-state analysis;
- multiprocessing for matrix-element and Hamiltonian construction.
## Environment
- Python 3
- NumPy
第 11 页
GitHub 项目整合与上传实操手册
- SciPy
- SymPy
Install dependencies:
```bash
python -m pip install -r requirements.txt
```
## Usage
The current snapshot preserves the original research configuration. Before executing the scripts
on another computer, check the parameter block, data-file path, output path and worker count 
recorded in the source files and `KNOWN_ISSUES.md`.
The intended execution order is:
```bash
python src/matrix_element_calculator.py
python src/many_body_hamiltonian.py
```
Generated `.pkl`, `.npz` and result files are excluded from Git by default because they may be 
machine-specific or large.
## Roadmap
- [ ] Replace machine-specific paths with project-relative paths.
- [ ] Connect the generated matrix-element filename to the Hamiltonian input automatically.
- [ ] Move run parameters to a configuration file or command-line interface.
- [ ] Add small reproducible examples and automated tests.
- [ ] Pin verified dependency versions.
- [ ] Add benchmark results and physics validation documentation.
## Citation
If this repository supports a publication, citation information will be added after the 
corresponding work is publicly available.
## License
No open-source license is granted in the initial research snapshot. A license will be selected 
after confirming publication and collaboration requirements.