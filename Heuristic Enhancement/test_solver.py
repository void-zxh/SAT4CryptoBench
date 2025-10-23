import os
import subprocess
import time
import logging
import concurrent.futures

EXPERIMENT_NUM = 7
SOLVER_NUM = 0
execution_path = './kissat/build/kissat' ## Change this to your own execution command for your solver

solver_options = ['kissat', 'MaplePainless', 'CDCL-Crypto', 'Maplesat', 'glucose-4.1-bmm', 'maplecomsps_lrb_vsids_18-bmm', 'maplelcmdistchronobt-bmm', 
                'maplesat-bmm', 'glucose-4.1', 'maplecomsps_lrb_vsids_18', 'maplelcmdistchronobt', 'cadical']
data_options = ['cipher-8', 'cipher-9', 'cipher-10', 'cipher-11', 'cipher-12',
                'simon-10-32-64', 'simon-11-32-64', 'simon-12-32-64',
                'md4-20rounds', 'sha1-21rounds', 'sha256-18rounds']

data_path_options = [
    'Cipher/cipher-8', 'Cipher/cipher-9', 'Cipher/cipher-10', 'Cipher/cipher-11', 'Cipher/cipher-12',
    'Simon/simon-10-32-64', 'Simon/simon-11-32-64', 'Simon/simon-12-32-64',
    'md4-20rounds/md4', 'sha1-21rounds/sha1', 'sha256-18rounds/sha256'
]

data_path = f'../{data_path_options[EXPERIMENT_NUM]}' ## Change this to your own data path
cnf_files = [f for f in os.listdir(data_path) if f.endswith('.cnf')]

log_path = f'./logs/{solver_options[SOLVER_NUM]}' ## Change this to your own log path
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, f'{data_options[EXPERIMENT_NUM]}.log'), ## Change this to your own log name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_single(execution_path, cnf_path):
    command = f'{execution_path} {cnf_path}'
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5000)
        end_time = time.time()
        execution_time = end_time - start_time
        log_msg = f'{cnf_path} took {execution_time} seconds'
        logging.info(log_msg)
    except subprocess.TimeoutExpired as e:
        end_time = time.time()
        log_msg = f'{cnf_path} took 5000 seconds'
        logging.info(log_msg)

if __name__ == '__main__':
    logging.info('Start Running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for cnf_file in cnf_files:
            cnf_path = os.path.join(data_path, cnf_file)
            futures.append(executor.submit(run_single, execution_path, cnf_path))
        concurrent.futures.wait(futures)