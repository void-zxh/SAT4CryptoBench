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


sat_samples = "../test_data/8rounds/sat"
N_cnf = 5
N_group = 4
N_iteration = 10
log_path = '../logs/kissat-rel-4.1.0/'           ######
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, 'hebo-test-8.log'),           ######
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filemode='w'
)

np.random.seed(42)


# Run a single cnf file
def run_single_sample(args: Tuple[str, int, int, float, str]) -> float:
    sample, restartint, reduceint, decay, sat_samples = args
    sample_path = os.path.join(sat_samples, sample)
    cmd = [
        '../SAT_Solvers/kissat-sc2021/build/kissat',          ######
        f'--restartint={int(restartint)}',
        f'--reduceint={int(reduceint)}',
        f'--decay={decay}',
        '--time=5000',
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


# Run a single group of cnf files
def run_kissat(restartint: int, reduceint: int, decay: float, sat_samples: str, n: int):
    # Select n random samples from the sat_samples directory
    params = (restartint, reduceint, decay)
    sat_files = os.listdir(sat_samples)
    selected_sat_samples = np.random.choice(sat_files, n).tolist()
    
    # Prepare arguments for parallel execution
    args = [(sample, restartint, reduceint, decay, sat_samples) 
            for sample in selected_sat_samples]
    
    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n, os.cpu_count())) as executor:
        times = list(executor.map(run_single_sample, args))
    
    return np.mean(times), params


# Run one iteration of the optimization
def obj(params: pd.DataFrame, sat_samples: str, n: int) -> np.ndarray:
    times = []
    for _, row in params.iterrows():
        restartint = int(row['restartint'])
        reduceint = int(row['reduceint'])
        decay = row['decay']
        
        logging.info(f"Testing params: restartint={restartint}, reduceint={reduceint}, decay={decay}")
        
        # 直接调用run_kissat函数，不使用线程池
        avg_time, params = run_kissat(restartint, reduceint, decay, sat_samples, n)
        
        if avg_time is not None:
            times.append(avg_time)
            logging.info(f"Average time: {avg_time:.2f} seconds for params: restartint={params[0]}, reduceint={params[1]}, decay={params[2]}")
        else:
            logging.info("Error occurred, skipping")
                
    return np.array(times).reshape(-1, 1)



if __name__ == '__main__':

    # Define the design space
    space = DesignSpace().parse([
        {'name': 'restartint', 'type': 'int', 'lb': 1, 'ub': 10000},
        {'name': 'reduceint',  'type': 'int', 'lb': 2, 'ub': 100000},
        {'name': 'decay',      'type': 'int', 'lb': 1, 'ub': 200}  # decay now as an int
    ])
    opt = HEBO(space)

    # Start from the Initial Observation
    initial_params = pd.DataFrame({
        'restartint': [1, 1000, 50],
        'reduceint': [1000, 1646, 3259],
        'decay': [50, 1568, 1]
    })
    initial_results = obj(initial_params, sat_samples, N_cnf)
    opt.observe(initial_params, initial_results)
    
    for i in range(N_iteration):
        rec = opt.suggest(n_suggestions=N_group)
        results = obj(rec, sat_samples, N_cnf)
        opt.observe(rec, results)

        # best_index = opt.y.argmin()
        # best_params = opt.X.iloc[best_index]
        # best_time = opt.y[best_index]