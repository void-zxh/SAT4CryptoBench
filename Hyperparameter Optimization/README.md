# 🚀 EasyNAS Quick Run Guide

## 1️⃣ Run Command

```bash
export PYTHONPATH=$PYTHONPATH:<path/to/EasyNAS>
cd <path/to/EasyNAS>
python app/run_engine.py --cfg <path/to/hpo.yaml>
```

> `<path/to/hpo.yaml>` points to your Hyper-Parameter Optimization (HPO) configuration file.

---

## 2️⃣ Solver Benchmark Command

In the YAML configuration file, the solver benchmarking command is defined under the `bash_cmd` field:

```yaml
engine:
  submodule_name: engines.BashEngine
  args:
    bash_cmd: "cd <path/to/EasyNAS> && bash test_cnf.sh"
    eval_names: ['time']
```

**Explanation:**
- `bash_cmd` specifies the command used to run the solver test script.  
  → The actual benchmarking script is:  
  `test_cnf.sh`
- EasyNAS automatically calls this script during each search iteration to measure solver performance.
- The evaluation metric (`eval_names: ['time']`) records the solver’s runtime as the optimization target.

---

## 3️⃣ Output and Logs

After execution:
- All logs and search results are saved under the `root_path` directory defined in your YAML file.  
- Each trial automatically benchmarks the solver by running `test_cnf.sh`.  
- The optimization process will maximize or minimize the `time` metric depending on your configuration.

---

✅ **Example Usage**

```bash
python app/run_engine.py --cfg configs/hpo_kissat.yaml
```

This will start the EasyNAS optimization process and benchmark your solver using the `test_cnf.sh` script.

# HEBO Quick Start

## Run Command

```bash
python hebo_optim_<solver>.py
```
<solver> corresponds to the solver name.

For example:

```bash
python hebo_optim_kissat.py
python hebo_optim_maplesat.py
python hebo_optim_cryptosat.py
```
