# hjxdl（导入名 `hdl`）

**作者：胡建星（Jianxing Hu）** ｜ 邮箱：j.hu@pku.edu.cn ｜ 仓库：https://github.com/huluxiaohuowa/hdl

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg?style=flat-square)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

`hjxdl` 是一个把日常科研代码沉淀下来的工具库，三块能力：

1. **化学信息学**：基于 RDKit 的分子绘制、标准化、骨架抽取、三维构象对齐、形状指纹、PDB 配体抽取、分子指纹与分子图特征化。
2. **神经网络**：基于 PyTorch / PyTorch Geometric 的图神经网络层与模型（GIN、GCN、手性图网络、图 Transformer、归一化流），配套数据集、DataLoader、训练器、预测器、归因解释与主动学习。
3. **应用工具**：大模型（LLM）调用封装、多模态与视频场景识别、数据库/网页/天气查询、Jupyter Notebook 快捷函数。

包内所有源码文件顶部都带有作者信息头（`# 作者：胡建星（Jianxing Hu）`），化学与神经网络相关模块的函数、类与核心算法逻辑都带有中文功能注释。

---

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `hdl/args/` | 损失函数与训练参数定义 |
| `hdl/controllers/al/` | 主动学习（active learning）控制器、任务调度、反馈回写 |
| `hdl/controllers/explain/` | 模型归因解释：Shapley 值、子图归因（subgraphx） |
| `hdl/controllers/predictors/` | 推理封装：GIN 预测器、反应预测器、通用 Torch 预测器 |
| `hdl/controllers/train/` | 训练流程：Trainer 基类、迭代式 Trainer、GINet 训练、反应模型训练 |
| `hdl/data/dataset/` | 数据集：指纹数据集、分子图数据集（GIN/MoleculeNet/手性）、反应序列数据集、数据切分与采样 |
| `hdl/data/to_mols.py` | SMILES / 结构文件与 RDKit 分子对象互转 |
| `hdl/datasets/` | 随包数据文件（城市编码与城市向量、词表、特征定义等） |
| `hdl/features/fp/` | 分子指纹（molecular fingerprint）特征生成器注册表 |
| `hdl/features/graph/` | 分子图特征化：原子特征、键特征 one-hot 编码 |
| `hdl/features/utils/` | 已算好特征的落盘/读回（`.npz`、`.npy`、`.csv`、`.pkl`、`.sdf`）与 SMILES 逐 token 切分正则 |
| `hdl/include/`、`hdl/pytorch/`、`hdl/kernel/`、`hdl/ops/` | 自定义 CUDA 算子：核函数、C++ 绑定、Python 侧封装 |
| `hdl/jupyfuncs/` | Jupyter 快捷函数集（`chem`、`dl`、`llm`、`path`、`show`、`network`、`dbtools`、`utils`） |
| `hdl/layers/general/` | 通用层：线性层、高斯过程（Gaussian Process）层 |
| `hdl/layers/graph/` | 图神经网络层：GIN、GCN、手性图卷积、四面体立体化学编码、图 Transformer |
| `hdl/layers/sequential/` | 序列模型层子包，目前只有空的包入口文件，实现在别的目录 |
| `hdl/metric_loss/` | 损失函数与评估指标：NT-Xent 对比学习损失、多标签损失、分类/回归指标 |
| `hdl/models/` | 模型定义与注册表（`model_dict`、`optim_dict`） |
| `hdl/optims/` | 自定义优化器（NAdam） |
| `hdl/utils/` | 通用工具：`chemical_tools`、`database_tools`、`llm`、`vis_tools`、`weather`、`schedulers`、`decorators`、`general`、`desc` |

---

## 安装

从 PyPI 安装：

```bash
pip install hjxdl
```

```python
import hdl
print(hdl.version)
```

本地开发安装（可编辑模式）：

```bash
git clone https://github.com/huluxiaohuowa/hdl.git
cd hdl
pip install -e .
```

### 依赖说明（重要）

`requirements.txt` 只声明了轻量通用依赖：`beautifulsoup4`、`openai`、`tqdm`、`geopy`、`pytz`、`duckduckgo_search[lxml]`、`opencv-python`、`redis[hiredis]`、`psycopg[binary]`、`Pillow`、`open_clip_torch`、`natsort`、`matplotlib`。

**`requirements.txt` 只覆盖了一小部分。** 下面这张表是对全部 `.py` 做 import 扫描得到的实际第三方依赖（`numpy` 被 25 个文件用到、`torch` 43 个、`rdkit` 15 个、`pandas` 13 个，四者都没有写进 `requirements.txt`），按需安装即可：

| 功能范围 | 实际需要的第三方包 |
| --- | --- |
| 基础数值与表格 | `numpy`、`scipy`、`pandas`、`scikit-learn`（`sklearn`） |
| 化学信息学 | `rdkit`；`molvs`（结构标准化）、`rxnfp`（反应指纹）、`py3Dmol` 与 `ipywidgets`、`IPython`（Notebook 渲染）、`cirpy` 与 `pubchempy`（化合物查询） |
| 三维构象与 PDB | `prody`、`pypdb`、`pyshapeit`、`multiprocess`，以及外部程序 PyMOL |
| 特征生成器 | `descriptastorus`（`rdkit_2d` / `rdkit_2d_normalized`；缺失时代码会注册成抛 `ImportError` 的占位实现） |
| 深度学习与图网络 | `torch`、`torch_geometric`、`torch_scatter`、`torch_sparse`、`transformers`、`einops`、`rotary_embedding_torch`、`gpytorch`（高斯过程层）、`networkx`（子图归因） |
| 大模型调用 | `openai`（已声明）、`instructor`、`tiktoken`、`PyYAML`（`yaml`）、`llama-cpp-python`、`gradio`；向量模型侧 `FlagEmbedding` / `BCEmbedding` / `sentence-transformers` |
| 文档与图像抽取 | `fitz`（PyMuPDF）、`pdfplumber`、`pytesseract`、`ltp`、`spire` |
| 其它工具 | `requests`、`psutil`、`timezonefinder`、`seaborn`、`typing_extensions`、`pkg_resources`（来自 `setuptools`） |

`hdl/datasets/city_embs.npy` 通过 Git LFS 管理（见 `.gitattributes`），克隆时请带上 `--recursive` 或先安装 `git-lfs`，否则该文件只是一个指针文本。

---

## 化学信息学功能

### 分子绘制与 Notebook 交互 —— `hdl/jupyfuncs/chem/mol.py`

- `draw_mol(mol)` / `moltosvg(mol, molSize, kekulize)`：把 RDKit 分子对象渲染成 SVG 显示。
- `drawmol_with_hi(mol, ...)`：在分子图上高亮指定原子/子结构。
- `show_atom_number(mol, label='atomNote')`：把原子序号标注到图上，便于核对特征索引。
- `draw_rxn(...)`：反应式绘制。
- `draw_mols_surfs(...)`：多分子表面（connolly surface）叠加绘制。
- `do_decomp(mols, cores, options)` / `show_decomp(...)`：R 基团分解（RGroupDecomposition），用于骨架-取代基拆解。
- `show_pharmacophore(...)`：药效团（pharmacophore）特征可视化。
- `get_ids_folds(id_list, num_folds, need_shuffle)`：交叉验证的 ID 折切分。
- `mol_without_indices(...)`：去掉分子缓存索引信息，避免序列化后特征错位。

### 分子标准化 —— `hdl/jupyfuncs/chem/norm.py`

`Normalization` / `Normalizer` 两个类负责分子规范化流程（去盐、碎片处理、电荷与同位素处理、重复单元归并等），按代码实际行为顺序执行。

### 母核骨架 —— `hdl/jupyfuncs/chem/scaffold.py`

`create_sn(...)` 生成分子的骨架编号（scaffold），用于骨架多样性分析与 `scaffold split` 数据集切分。

### 三维构象与分子形状 —— `hdl/jupyfuncs/chem/shape.py`

`get_mols_from_smi` 读取 SMILES，`gen_configs` 生成多构象，`get_aligned_mol` / `get_aligned_mol_mp` 做探针分子与库分子的三维叠合（含多进程版本），`get_aligned_sdf` 输出叠合后的 SDF，`show_alignment` 调 PyMOL 展示，`pymol_running` 检测 PyMOL 是否在运行。

### PDB 结构配体抽取 —— `hdl/jupyfuncs/chem/pdb_ext.py`

`get_pdb_components(pdb_id)` 拉取 PDB 条目组分，`process_ligand(ligand, res_name)` 处理配体，`write_pdb` / `write_sdf` 落盘，`main(pdb_name)` 是整体入口。

### 分子指纹特征生成 —— `hdl/features/fp/features_generators.py`

带注册表的指纹生成器工厂：

- `register_features_generator(name)`：装饰器，把函数登记进生成器表。
- `get_features_generator(name)`：按名字取生成器。
- `get_available_features_generators()`：列出所有可用生成器名。
- 已注册的生成器名：`morgan`（二值 Morgan/ECFP，`morgan_binary_features_generator`）、`morgan_count`（计数版）、`maccs`（167 位 MACCS 键）、`rdkit_2d` 与 `rdkit_2d_normalized`（依赖 `descriptastorus`，未安装时注册的是抛 `ImportError` 的占位实现）、`e3fp`、`whales`、`selfies`。
  实测 `get_available_features_generators()` 返回 8 个名字：`e3fp, maccs, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized, selfies, whales`；文件末尾的 `custom_features_generator` 不在这个列表里。
- 注意：`e3fp`、`whales`、`selfies` 三个当前是**占位实现**，只写了 docstring 与文献链接，函数体 `return NotImplemented`，实际调用拿不到特征。

输入既可以是 SMILES 字符串也可以是 RDKit 分子对象，返回一维 `numpy` 数组。

### 分子图特征化 —— `hdl/features/graph/featurization.py`

把分子转成图神经网络可用的原子特征与键特征（one-hot / 类别索引），包含原子种类、手性标签、键类型、共轭、成环、键方向等编码；`hdl/layers/graph/*` 里的嵌入维度常量与此处的特征类别数对应。

### 结构文件与化合物查询 —— `hdl/utils/chemical_tools/`、`hdl/jupyfuncs/dbtools/`

- `sdf.py`：SDF 多分子文件读写与逐条解析。
- `query_info.py` / `hdl/jupyfuncs/dbtools/query_info.py`：按化合物名称查询，`query_from_cir`（CIR）与 `query_from_pubchem`（PubChem）分别走不同数据源。

---

## 神经网络功能

### 层 —— `hdl/layers/`

| 文件 | 内容 |
| --- | --- |
| `graph/gin.py` | `GINEConv(MessagePassing)`：图同构网络（GIN）边感知卷积，含键类型与键方向嵌入（`num_bond_type=5`、`num_bond_direction=3`、`num_atom_type=119`、`num_chirality_tag=3`） |
| `graph/gcn.py` | 图卷积网络（GCN）层 |
| `graph/chiral_graph.py` | 手性（chirality）感知的图卷积层 |
| `graph/tetra.py` | 四面体立体化学编码层，用于真实三维构象下的方向性消息传递 |
| `graph/transformer.py` | 图/序列 Transformer 注意力层 |
| `general/linear.py` | 线性层族与多种激活、归一化组合 |
| `general/gp.py` | 高斯过程（Gaussian Process）层 |

### 模型 —— `hdl/models/`

| 文件 | 内容 |
| --- | --- |
| `ginet.py` | `GINet`（分子性质预测主干，参数 `num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean'`）与 `GINMLPR`（GIN + 多层感知机读出） |
| `chiral_gnn.py` | `GNN`：手性图神经网络 |
| `fast_transformer.py` | `MultiTaskMultiClassBlock`、`MuMcHardBlock`：高效注意力（linear attention）多任务多分类模型 |
| `rxn.py` | 反应/序列模型 |
| `norm_flows.py` | 可微归一化流（normalizing flow），含耦合层与对数雅可比行列式（log det Jacobian）计算 |
| `linear.py` | 线性基线模型 |
| `model_dict.py` | 模型名 → 类的注册表：`rxn_trans`、`rxn_trans_hard`、`mmiter_linear`、`chiral_gnn`、`ginet`、`ginmlpr` |
| `optim_dict.py` | 优化器名 → 优化器的注册表：`adam`、`adadelta`、`sgd`、`rmsprop`、`nadam` |
| `utils.py` | 模型侧公共工具 |

### 数据管线 —— `hdl/data/`

- `dataset/base_dataset.py`：数据集基类。
- `dataset/graph/gin.py`：`MoleculeDataset`（从 SMILES/CSV 读取分子，`__getitem__` 返回图对象）与 `MoleculeDatasetWrapper`（按 `batch_size`、`num_workers`、`valid_size`、`data_path` 直接产出 train/valid/test DataLoader）。
- `dataset/graph/molnet.py`：`MoleculeNet` 数据集，含 `process`、`collate`、`num_classes` 等接口。
- `dataset/graph/chiral.py`：手性图数据集。
- `dataset/fp/fp_dataset.py`：指纹向量数据集。
- `dataset/seq/rxn_dataset.py`：反应 SMILES 序列数据集，配合词表做 tokenization。
- `dataset/loaders/collate_funcs/`：`fp.py`（指纹批张量堆叠）与 `rxn.py`（序列 padding 与 mask 构造）的 batch 合并函数。
- `dataset/loaders/spliter.py`：`split_data(smis, labels, split_type='random', sizes=(0.8, 0.2, 0.0), seed=999, num_folds=1, balanced=True)` 做训练/验证/测试切分，支持随机与按骨架（scaffold）等策略。
- `dataset/loaders/general.py`、`chiral_graph.py`：DataLoader 构建。
- `dataset/samplers/chiral.py`：手性相关采样器。
- `dataset/utils.py`：`read_smiles` 等读取工具。

### 训练、推理、解释 —— `hdl/controllers/`

- `train/trainer_base.py`：Trainer 基类，封装优化器、损失、指标累积、检查点（checkpoint）保存。
- `train/trainer_iterative.py`：迭代式训练器（逐 step 训练与验证）。
- `train/train_ginet.py`、`train/rxn_train.py`：针对 GINet 与反应模型的完整训练脚本入口。
- `predictors/torch_predictor.py` / `gin_predictor.py` / `rxn_predictor.py`：推理封装，统一 `model.eval()` + `no_grad` 的预测流程与结果后处理。
- `explain/shapley.py`：基于联盟（coalition）采样的夏普利值（Shapley value）特征/子结构归因。
- `explain/subgraphx.py`：子图归因，找出对预测贡献最大的分子子图。
- `al/al.py`、`al/dispatcher.py`、`al/feedback.py`：主动学习闭环，用不确定性/多样性采集函数（acquisition function）挑样本并回写反馈。

### 损失、指标、优化器、调度 —— `hdl/metric_loss/`、`hdl/optims/`、`hdl/utils/schedulers/`、`hdl/args/`

- `metric_loss/nt_xent.py`：归一化温度缩放交叉熵损失（NT-Xent），用于对比学习，含正负样本对构造、相似度矩阵与对角掩码、温度系数 `tau`。
- `metric_loss/multi_label.py`：多标签/多分类序列损失，处理忽略索引（ignore index）与逐 token/逐序列聚合。
- `metric_loss/loss.py`：损失注册与选择。
- `metric_loss/metric.py`：分类与回归评估指标计算与累积统计。
- `optims/nadam.py`：NAdam 优化器（带动量延迟与偏差修正）。
- `utils/schedulers/norm_lr.py`：学习率调度器。
- `args/loss_args.py`：损失相关命令行/配置参数定义。

### 自定义 CUDA 算子 —— `hdl/kernel/`、`hdl/include/`、`hdl/pytorch/`、`hdl/ops/`

`add2_kernel.cu` 实现逐元素相加核函数（含 `blockIdx`/`threadIdx` 索引与 `if (i < n)` 边界保护），`add2.h` 声明接口，`hdl/pytorch/add2_ops.cpp` 做 PyTorch 扩展绑定，`hdl/ops/utils.py` 提供 Python 侧加载与调用。

---

## 大模型与应用工具

- `hdl/utils/llm/llm_wrapper.py`：`OpenAIWrapper(client_conf=None, client_conf_dir=None, load_conf=True)`，按配置目录里的多个模型配置分别建客户端，提供 `add_client`、`load_clients`、`get_resp`、`invoke`、`stream`、`embedding`。兼容任何 OpenAI 协议网关（含自建 vLLM、Groq 等）。
- `hdl/utils/llm/chat.py`：`OpenAI_M`（工具调用与思维链 Markdown 解析）、`MMChatter`（多轮会话）、`object_detect`（目标检测式视觉问答）、`parse_fn_markdown` / `parse_cot_markdown` / `run_tool_with_kwargs`。
- `hdl/utils/llm/embs.py`：`BEEmbedder`、`HFEmbedder`、`get_n_tokens`，文本向量化与 token 计数。
- `hdl/utils/llm/vis.py`：图像与 Base64 互转（`to_img`、`to_base64`、`imgurl_to_base64`、`imgfile_to_base64`、`imgbase64_to_pilimg`、`pilimg_to_base64`）以及 `draw_and_plot_boxes_from_json` 按 JSON 框坐标画图。
- `hdl/utils/llm/visrag.py` + `hdl/utils/llm/extract.py`：图文检索增强（RAG）：PDF 入库 `add_pdf_gradio`、检索 `retrieve_gradio`、看图回答 `answer_question`、点赞点踩反馈 `upvote` / `downvote`、`DocExtractor` 文档抽取。
- `hdl/utils/llm/ollama.py`、`llama_chat.py`、`chatgr.py`：本地 Ollama / GGUF 模型与 Gradio 会话演示。
- `hdl/utils/decorators/llm.py`：`measure_stream_performance` 流式吞吐计时、`run_llm_stream` 流式调用封装。
- `hdl/utils/vis_tools/scene_detect.py`：视频场景切分（`SceneDetector`、`detect_scenes_cli`、`extract_frames_with_cv`、`describe_image`、`fill_descriptions`），用多模态模型给分帧生成描述。
- `hdl/utils/database_tools/`：`connect_by_infofile`、`conn_redis`、`web_search_text`、`fetch_baidu_results`、`wolfram_alpha_calculate`、`get_datetime_by_cityname`。
- `hdl/utils/weather/weather.py`：`get_weather(city)` 城市天气查询，配合 `hdl/datasets/city_code.json` 与 `city_embs.npy` 做城市名匹配。
- `hdl/jupyfuncs/path/glob.py`：`in_jupyter`、`in_docker`、`get_files`、`recursive_glob`、`makedirs`、`chunkify_file`、`parallel_apply_line_by_line_chunk` 等文件与运行环境工具。
- `hdl/jupyfuncs/show/plot.py`：`accuracies_heat`、`get_metrics_curves`、`get_means_vars`、`get_metrics_bars`，多任务指标热力图与曲线。
- `hdl/jupyfuncs/dl/`：`tensor.py`（张量操作）、`fp.py`（`get_maccs_fp` / `get_morgan_fp` / `get_rdnorm_fp` / `get_fp` 按名取指纹）、`uncs.py`（不确定性采样度量：最小置信度 `least_conf_unc`、间隔 `margin_conf_unc`、比值 `ratio_conf_unc`、熵 `entropy_unc`，以及按名字取用 `get_prob_unc`，供主动学习采集函数使用）、`cp.py`（共形预测（conformal prediction）分类器 `CpClassfier`，用校准集概率做置信度校准）、`model_utils.py`（模型参数与权重工具）、`dataframe.py`、`list.py`（`list_diff` 交并差集）。
- `hdl/jupyfuncs/network/proxy.py`：`get_proxies` 代理配置。

---

## 快速上手

> 示例 1 已在本仓库当前代码上实测跑通；示例 2-4 需要上面依赖表里的可选包（本机未装 `IPython`、`torch_geometric`、`torch_scatter` 时分别会在 `hdl.jupyfuncs.chem.mol` 与 `hdl.models.model_dict` 的 import 处报 `ModuleNotFoundError`）。

### 1. 生成分子指纹（只依赖 RDKit + numpy）

```python
from hdl.features.fp.features_generators import get_features_generator

morgan = get_features_generator('morgan')
print(morgan('CC(=O)Oc1ccccc1C(=O)O').shape)   # 阿司匹林 -> (2048,)

maccs = get_features_generator('maccs')
vec = maccs('CC(=O)Oc1ccccc1C(=O)O')
print(vec.shape, int(vec.sum()))               # -> (167,) 21
```

上面这段是实测跑通的（RDKit 会打印一条 `DEPRECATION WARNING: please use MorganGenerator`，来自 `GetMorganFingerprintAsBitVect` 的新旧 API 过渡，不影响结果）。

### 2. 在 Notebook 里画分子

```python
from rdkit import Chem
from hdl.jupyfuncs.chem.mol import draw_mol

draw_mol(Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O'))
```

### 3. 搭一条 GIN 分子性质预测流水线（依赖 torch + torch_geometric）

```python
from hdl.data.dataset.graph.gin import MoleculeDatasetWrapper
from hdl.models.model_dict import model_dict
from hdl.models.optim_dict import optim_dict

wrapper = MoleculeDatasetWrapper(
    batch_size=50, num_workers=4, valid_size=0.1,
    data_path='data.csv', file_type='smi', y_col_name='label',
)
train_loader, valid_loader = wrapper.get_data_loaders()

model = model_dict['ginet'](num_layer=5, emb_dim=300, feat_dim=512)
optimizer = optim_dict['nadam'](model.parameters())
```

训练循环与检查点逻辑见 `hdl/controllers/train/trainer_base.py`、`train_ginet.py`；推理封装见 `hdl/controllers/predictors/gin_predictor.py`。

注意 `get_data_loaders()` 只把 `data_path` 传给数据集（`file_type`、`y_col_name` 等配置走默认值），因此它产出的加载器不带标签；带标签列/多 SMILES 列的加载器要用 `get_test_loader()`。

### 4. 调用大模型

```python
from hdl.utils.llm.llm_wrapper import OpenAIWrapper

llm = OpenAIWrapper(client_conf_dir='/path/to/model_conf.yaml')
resp = llm.get_resp('用一句话解释 MACCS 指纹')
```

---

## 版本与发布

- 包版本由 `setuptools_scm` 从 git tag 推导（`pyproject.toml` 里 `tag_regex = "^(\d+\.\d+\.\d+)$"`），并写入 `hdl/_version.py`；仓库里该文件是空占位，构建时才生成。
- `version.txt` 记录当前发布版本号，供发布脚本读取。
- `update_main.sh`：读取 `version.txt` → 末位版本号 +1 → 写回 → 提交 → 打 tag → 推送 `main` 与 tag。
- `.github/workflows/python-publish.yml`：监听到任意 tag 推送后，用 `python -m build --wheel` 构建并发布到 PyPI（检出时启用 Git LFS 与完整历史）。

---

## 已知问题（提交前实测确认）

以下内容是仓库现状，本次改动（加作者头、加中文注释、写 README）没有修改任何代码逻辑，因此这些问题依然存在：

1. **两个文件在 HEAD 就有语法错误，无法编译**：
   - `hdl/controllers/train/train.py` —— 约第 50 行有一段没有加引号的英文说明文字（`As result we get something like this:`），像是本该写成注释或 docstring 的内容漏了引号。
   - `hdl/utils/general/runners.py` —— 约第 26-34 行括号不匹配（`']' does not match opening parenthesis '('`）。
   两处都在本次改动之前就存在（已用 `git show HEAD:<file>` 逐字节确认），修复需要判断作者本意，未擅自改动。
2. `hdl/jupyfuncs/path/glob.py` 的 `get_dataset_file()` 里读的是 `pkg_resources.path('jupyfuncs.datasets', filename)`，而实际包路径是 `hdl.datasets`（`hdl/jupyfuncs/datasets/` 不存在），该函数在当前布局下会抛 `ValueError`。
3. `hdl/kernel/test` 是一个已编译的 x86-64 ELF 可执行文件被提交进了仓库，属于构建产物。
4. `path/to/utils/general/runners.py` 是 `hdl/utils/general/runners.py` 的同名副本，疑似误提交，且会在 `pip install` 时被 `find_packages()` 之外的方式带到源码目录里。
5. `hdl/datasets/las.tsv` 与 `hdl/datasets/route_template.json` 在当前仓库的 Python 代码中没有任何引用。
6. `hdl/datasets/vocab.txt` 与 `hdl/features/vocab.txt` 已随包发布，但 `hdl/data/dataset/seq/rxn_dataset.py` 和 `collate_funcs/rxn.py` 里硬编码的是相对路径 `models/transformers/bert_ft/vocab.txt`，两者对不上。
7. 许可证不一致：`LICENSE` 文件正文是 Anti 996 License 1.0（草稿版），但 `setup.py` 的 `classifiers` 声明的是 MIT License。
8. `hdl/features/fp/features_generators.py` 里 `e3fp`、`whales`、`selfies` 三个特征生成器是占位实现（函数体 `return NotImplemented`），已注册进表但没有真实输出。
9. `requirements.txt` 缺少大量实际依赖：`numpy`、`torch`、`rdkit`、`pandas` 这四个被数十个文件直接 import 的包都没有声明，因此 `pip install hjxdl` 之后化学与神经网络相关模块无法直接导入，需要照上面的依赖表自行安装。

---

## 作者与许可证

作者：胡建星（Jianxing Hu），邮箱 j.hu@pku.edu.cn，仓库地址 https://github.com/huluxiaohuowa/hdl 。

许可证见 `LICENSE`（Anti 996 License，另见上文已知问题第 7 条关于 `setup.py` 声明不一致的说明）。
