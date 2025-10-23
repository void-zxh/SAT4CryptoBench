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
log_path = './logs/maplesat/'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, 'hebo-simon-12.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

np.random.seed(42)


# Run a single cnf file
def run_single_sample(args: Tuple[str, float, int, float, str]) -> float:
    sample, rinc, phasesaving, rndfreq, sat_samples = args
    sample_path = os.path.join(sat_samples, sample)
    cmd = [
        '/home/ma-user/SatBenchmark/optimize/maplesat/core/maplesat_static',
        sample_path,
        '-no-luby',
        f'-rinc={rinc}',
        f'-phase-saving={phasesaving}',
        f'-rnd-freq={rndfreq}',
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
def run_kissat(rinc: float, phasesaving: int, rndfreq: float, sat_samples: str, n: int) -> float:
    # Select n random samples from the sat_samples directory
    params = (rinc, phasesaving, rndfreq)
    sat_files = os.listdir(sat_samples)
    selected_sat_samples = np.random.choice(sat_files, n).tolist()
    
    # Prepare arguments for parallel execution
    args = [(sample, rinc, phasesaving, rndfreq, sat_samples) 
            for sample in selected_sat_samples]
    
    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n, os.cpu_count())) as executor:
        times = list(executor.map(run_single_sample, args))
    
    return np.mean(times), params


# Run one iteration of the optimization
def obj(params: pd.DataFrame, sat_samples: str, n: int) -> np.ndarray:
    times = []
    for _, row in params.iterrows():
        rinc = float(row['rinc'])
        phasesaving = int(row['phase-saving'])
        rndfreq = float(row['rnd-freq'])
        
        logging.info(f"Testing params: rinc={rinc}, phase-saving={phasesaving}, rnd-freq={rndfreq}")
        
        # 直接调用run_kissat函数，不使用线程池
        avg_time, params = run_kissat(rinc, phasesaving, rndfreq, sat_samples, n)
        
        if avg_time is not None:
            times.append(avg_time)
            logging.info(f"Average time: {avg_time:.2f} seconds for params: rinc={rinc}, phase-saving={phasesaving}, rnd-freq={rndfreq}")
        else:
            logging.info("Error occurred, skipping")
                
    return np.array(times).reshape(-1, 1)



if __name__ == '__main__':

    # Define the design space
    space = DesignSpace().parse([
        {'name': 'rinc', 'type': 'num', 'lb': 1, 'ub': 10000}, # 应该是1到正无穷
        {'name': 'phase-saving',  'type': 'int', 'lb': 0, 'ub': 2},
        {'name': 'rnd-freq',      'type': 'num', 'lb': 0, 'ub': 1}
    ])
    opt = HEBO(space)

    # Start from the Initial Observation
    initial_params = pd.DataFrame({
        'rinc': [1.5],
        'phase-saving': [0],
        'rnd-freq': [0.02]
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