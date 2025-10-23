import os
import time
import pandas as pd
import numpy as np
import subprocess
import logging
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import concurrent.futures
from typing import List, Tuple


sat_samples = '/home/ma-user/SatBenchmark/test_data/simon/simon-12-32-64-final/sat'
N_cnf = 5
N_group = 4
N_iteration = 10
log_path = './logs/cryptominisat/'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, 'hebo-simon-12.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

np.random.seed(42)



def run_single_sample(args: Tuple[str, int, int, int, str]) -> float:
    sample, gluehist, rstfirst, confbtwsimp, sat_samples = args
    sample_path = os.path.join(sat_samples, sample)
    cmd = [
        '/home/ma-user/SatBenchmark/optimize/cryptominisat/build/cryptominisat5',
        '--gluehist', str(gluehist),
        '--rstfirst', str(rstfirst),
        '--confbtwsimp', str(confbtwsimp),
        sample_path
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5000)
        end_time = time.time()
        process_time = end_time - start_time

        if result.returncode != 10:
            logging.info(f"{sample} run error")
        else:
            logging.info(f"{sample} took {process_time} seconds")
        return process_time
    except subprocess.TimeoutExpired:
        logging.info(f"{sample} took 5000 seconds")
        return 5000.0


def run_cryptominisat(gluehist: int, rstfirst: int, confbtwsimp: int, sat_samples: str, n: int):
    params = (gluehist, rstfirst, confbtwsimp)
    sat_files = os.listdir(sat_samples)
    selected_sat_samples = np.random.choice(sat_files, n).tolist()

    args = [(sample, gluehist, rstfirst, confbtwsimp, sat_samples) for sample in selected_sat_samples]

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n, os.cpu_count())) as executor:
        times = list(executor.map(run_single_sample, args))
    
    return np.mean(times), params


def obj(params: pd.DataFrame, sat_samples: str, n: int) -> np.ndarray:
    times = []
    for _, row in params.iterrows():
        gluehist = int(row['gluehist'])
        rstfirst = int(row['rstfirst'])
        confbtwsimp = int(row['confbtwsimp'])

        logging.info(f"Testing params: gluehist={gluehist}, rstfirst={rstfirst}, confbtwsimp={confbtwsimp}")

        avg_time, params = run_cryptominisat(gluehist, rstfirst, confbtwsimp, sat_samples, n)

        if avg_time is not None:
            times.append(avg_time)
            logging.info(f"Average time: {avg_time:.2f} seconds for params: gluehist={gluehist}, rstfirst={rstfirst}, confbtwsimp={confbtwsimp}")
        else:
            logging.info("Error occurred, skipping this set of parameters")
    return np.array(times).reshape(-1, 1)



if __name__ == '__main__':
    space = DesignSpace().parse([
        {'name': 'gluehist', 'type': 'int', 'lb': 1, 'ub': 500},
        {'name': 'rstfirst', 'type': 'int', 'lb': 50, 'ub': 2000},
        {'name': 'confbtwsimp', 'type': 'int', 'lb': 5000, 'ub': 100000}
    ])

    opt = HEBO(space)

    initial_params = pd.DataFrame({
        'gluehist': [50],
        'rstfirst': [100],
        'confbtwsimp': [40000]
    })
    initial_results = obj(initial_params, sat_samples, N_cnf)
    opt.observe(initial_params, initial_results)
    
    for i in range(N_iteration):
        rec = opt.suggest(n_suggestions=N_group)
        results = obj(rec, sat_samples, N_cnf)
        opt.observe(rec, results)
