# 🔐 Neural Initialization + SAT Solver Pipeline for SIMON-12/32/64

本项目包含一个两阶段的自动化脚本，旨在结合 **神经网络预测模型** 和 **SIMON 密码分析求解器**，实现基于 ANF 表示的 SAT 实例的初始化与求解过程。

---

## 📁 项目结构说明

```plaintext
├── compute_time.py              # 主执行脚本
├── prediction.py                # 神经网络预测器脚本
├── simon_anf                    # 本地编译好的SIMON ANF求解器（二进制可执行文件）
├── simon-12-32-64-final         # anf文件夹下存储ANF格式的SAT实例，cnf文件夹下存储cnf格式的SAT实例，plain_cipher文件夹下存储明密文对的原始TXT文件
```

---

## ⚙️ 脚本功能说明

`compute_time.py` 执行两个主要阶段：

### 🧠 第 1 阶段：初始化比特串预测

使用神经网络模型CryptoANFNet对每个 SAT 实例预测一个 32 位密钥初始化比特串，用于加速求解过程。

该预测由如下命令调用：

```python
predict_cmd = (
    f"python prediction.py assignment {cnf_path} {CHECKPOINT_PATH} "
    f"--graph anf --seed 123 --model neurosat --test_splits sat"
)
```

- `assignment`: 表示预测任务类型为“变量赋值”
- `{cnf_path}`: 输入的SAT样例的 ANF 表示文件路径
- `{CHECKPOINT_PATH}`: 训练好的神经网络模型路径
- `--graph anf`: 指明图结构使用 ANF 格式构造
- `--model neurosat`: 指定使用的模型类型（例如 NeuroSAT）
- `--test_splits sat`: 指定测试集划分，这里默认是sat

输出应包含一行 32 位比特串，表示模型预测的初始密钥。

### ⚙️ 第 2 阶段：使用初始化值调用求解器

使用 `simon_anf` 工具进行求解：

```bash
脚本路径 <path_to_txt_file> <predicted_key_string>
```

- `<path_to_txt_file>`：包含明文和密文对的 `.txt` 文件路径
- `<predicted_key_string>`：神经网络预测出的 32 位密钥比特串

该二进制程序将根据明密文对和预测的密钥初始化值，执行 ANF 求解过程。

---

## 🔧 配置区说明

```python
TXT_FOLDER = "<明密文对存储文件夹路径>"
CNF_FOLDER = "<ANF SAT实例存储文件夹路径>"
CHECKPOINT_PATH = "<神经网络模型路径>"
```

请根据具体数据集和模型路径，修改上述三项以匹配你的本地文件系统。 注意，plain_cipher文件夹下和ANF样例文件夹下命名要对应，如 0.txt 和 0.cnf

---

## 📌 输入格式说明

- `*.txt`：包含明文和密文的明密文对文件（用于实际求解）
- `*.cnf`：ANF 生成的 SAT 实例，需与 `*.txt` 同名，仅后缀不同（用于预测）

---

## 🚀 如何运行

确保你已准备好 CNF 文件、TXT 文件与训练好的模型文件：

```bash
python compute_time.py
```

---
## 📝 输出结果

- 预测阶段将输出每个文件的密钥比特串预测结果
- 求解阶段输出每个文件的求解耗时
- 脚本末尾报告平均求解耗时（不包括预测时间）

---

