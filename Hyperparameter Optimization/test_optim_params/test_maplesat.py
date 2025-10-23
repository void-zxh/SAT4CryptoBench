import os
import time
import logging
import subprocess
import concurrent.futures


logging.basicConfig(
    filename='/home/ma-user/SatBenchmark/optimize/logs/maplesat/hebo-simon12-ho.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def extract_best_param(log_path):
    '''
    Extract the best parameters and the corresponding time from the log file.
    :param log_path: The path to the log file.
    :return:
        best_params: The best parameters (a dictionary).
        best_time: The corresponding time (a float).
    Example:
        Best params: {'restartint': '5767', 'reduceint': '98602', 'decay': '171'}
        Best time: 0.08
    '''
    best_time = float('inf')
    best_params = None
    with open(log_path, 'r') as f:
        current_params = None
        for line in f:
            line = line.strip()
            if 'Testing params' in line:
                params_str = line.split('Testing params:')[1].strip()
            elif 'Average time:' in line:
                time_str = line.split('Average time:')[1].strip().split('seconds')[0].strip()
                avg_time = float(time_str)
                if avg_time < best_time:
                    best_time = avg_time
                    best_params = params_str
    if best_params:
        param_dict = {}
        for param in best_params.split(','):
            key, value = param.strip().split('=')
            param_dict[key] = value
        return param_dict, best_time
    else:
        return None, None


def test_single(execution_path, cnf_path, params):
    cmd = [
        execution_path,
        cnf_path,
        '-no-luby',
        f'-rinc={params["rinc"]}',
        f'-phase-saving={params["phase-saving"]}',
        f'-rnd-freq={params["rnd-freq"]}',
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5000)
        end_time = time.time()
        process_time = end_time - start_time
        
        if result.returncode != 10:
            logging.info(f"{cnf_path} run error")
        else:
            logging.info(f"{cnf_path} took {process_time} seconds")
    except subprocess.TimeoutExpired:
        end_time = time.time()
        logging.info(f"{cnf_path} took 5000 seconds")


def test_params(execution_path, cnf_path, params):
    cnf_files = [f for f in os.listdir(cnf_path) if f.endswith('.cnf')]
    logging.info('Start Running')
    logging.info(f'Testing params: {params}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for cnf_file in cnf_files:
            tmp = os.path.join(cnf_path, cnf_file)
            futures.append(executor.submit(test_single, execution_path, tmp, params))
        concurrent.futures.wait(futures)



if __name__ == '__main__':
    log_path = '/home/ma-user/SatBenchmark/optimize/logs/maplesat/hebo-simon12-ho.log'
    data_path = '/home/ma-user/workspace_sxh/SatBenchmark/test_data/simon/simon-12-32-64-final/sat'
    execution_path = '/home/ma-user/SatBenchmark/optimize/maplesat/core/maplesat_static'
    # best_params, best_time = extract_best_param(log_path)
    # print(f'Best params: {best_params}')
    # print(f'Best time: {best_time}')
    # best_params = {
    #     'rinc': '2009.1',
    #     'phase-saving': '0',
    #     'rnd-freq': '0.027',
    # }
    # best_params = {
    #     'rinc': '1.5',
    #     'phase-saving': '0',
    #     'rnd-freq': '0.02',
    # }
    best_params = {
        'rinc': '1346',
        'phase-saving': '0',
        'rnd-freq': '0.002',
    }
    test_params(execution_path, data_path, best_params)
