# Heuristic Enhancement Solver Evaluation

This repository provides a unified testing script `test_solver.py` for evaluating various SAT solvers on cryptographic benchmark datasets.

## ⚙️ 1. Prerequisites

Before running the test script, make sure that:

1. You have compiled the solvers you intend to test according to their respective `README.md` files.
    Example for Kissat:

   ```
   cd kissat
   ./configure
   make
   ```

   The compiled binary should appear at `kissat/build/kissat`.

2. Your dataset folders (e.g., `Cipher/`, `Simon/`, etc.) contain `.cnf` files.

------

## 🚀 2. Running the Evaluation Script

The main evaluation script is **`test_solver.py`**.

You can specify:

- which solver to run (`SOLVER_NUM`)
- which dataset to test (`EXPERIMENT_NUM`)
- and the corresponding execution path.

### Example usage

```
python test_solver.py
```

The script will:

- Automatically locate `.cnf` files under the selected dataset path
- Run each instance using the specified solver
- Log execution time to a log file under `./logs/<solver_name>/`

### Parameters inside the script

```
EXPERIMENT_NUM = 7   # Select dataset index (0–10)
SOLVER_NUM = 0       # Select solver index (0–11)
execution_path = './kissat/build/kissat'  # Path to solver executable
```

### Solver options

```
solver_options = [
    'kissat', 'MaplePainless', 'CDCL-Crypto', 'Maplesat',
    'glucose-4.1-bmm', 'maplecomsps_lrb_vsids_18-bmm',
    'maplelcmdistchronobt-bmm', 'maplesat-bmm', 'glucose-4.1',
    'maplecomsps_lrb_vsids_18', 'maplelcmdistchronobt', 'cadical'
]
```

### Dataset options

```
data_options = [
    'cipher-8', 'cipher-9', 'cipher-10', 'cipher-11', 'cipher-12',
    'simon-10-32-64', 'simon-11-32-64', 'simon-12-32-64',
    'md4-20rounds', 'sha1-21rounds', 'sha256-18rounds'
]
```

Logs are automatically generated in:

```
./logs/<solver_name>/<dataset_name>.log
```

------

## 📝 4. Logging and Timeout

Each run is logged with the following information:

```
<cnf_filename> took <execution_time> seconds
```

- Timeout for each instance: **5000 seconds**
- Logs are saved under the folder: `./logs/<solver_name>/`

------

## 💡 5. Notes

- Parallel execution: The script uses `ThreadPoolExecutor` with `max_workers=5` for concurrent execution.

- You can modify the number of threads based on your device performance.

- Make sure your solvers can run with shell commands like:

  ```
  ./kissat/build/kissat <input.cnf>
  ```

------

## 🧩 Example Output

Example log content (`logs/kissat/simon-12-32-64.log`):

```
2025-10-21 14:22:31,123 - INFO - Start Running
2025-10-21 14:22:32,584 - INFO - ../Simon/simon-12-32-64/0.cnf took xxx seconds
2025-10-21 14:22:40,777 - INFO - ../Simon/simon-12-32-64/1.cnf took xxx seconds
...
```

---

## 🧠 Notes

For neural-based solvers, additional preprocessing steps are required.

1. **Modify Hardcoded Paths**
    In the original scripts of GraphQSat, NeuroBack, and neuro-cadical, change all hardcoded file paths to match your environment.

2. **(Optional) Huawei Ascend NPU Migration**

   If running on a Huawei Ascend device, you can migrate PyTorch CUDA calls to NPU using Huawei’s migration script:

   ```
   ${ASCEND_TOOLKIT_HOME}/latest/tools/ms_fmk_transplt/pytorch_gpu2npu.sh \
     -i <path_to_original_script> \
     -o <path_to_output_migrated_script> \
     -v <pytorch_version>
   ```

   This converts all `torch.cuda` calls into `torch.npu` and helps adapt the solver to NPU devices.
    Note: You may still need to resolve dependency or compilation issues manually.

3. **Running on non-NPU devices**

   You can also execute the same code on GPU/CPU after removing or adjusting the NPU-specific calls.

