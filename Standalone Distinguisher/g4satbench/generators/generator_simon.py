import os
import subprocess
import time
from copy import deepcopy
import argparse
import numpy as np
import random
import networkx as nx

from tqdm import tqdm

WDSat_path = "./WDSat-master/wdsat_solver"
kissat_path = "./kissat/build/kissat"
MAX_RETRIES = 20

class Generator:
    def __init__(self, seed, rounds, size, hsize, opts):
        self.seed = seed    
        self.rounds = rounds    
        self.ABC = [
            0x0123456789abcdef, 0x123456789abcdef0,
            0x23456789abcdef01, 0x3456789abcdef012,
            0x456789abcdef0123, 0x56789abcdef01234,
            0x6789abcdef012345, 0x789abcdef0123456,
            0x89abcdef01234567, 0x9abcdef012345678,
            0xabcdef0123456789, 0xbcdef0123456789a,
            0xcdef0123456789ab, 0xdef0123456789abc,
            0xef0123456789abcd, 0xf0123456789abcde,
            0x0123456789abcdef, 0x123456789abcdef0,
            0x23456789abcdef01, 0x3456789abcdef012,
            0x456789abcdef0123, 0x56789abcdef01234,
            0x6789abcdef012345, 0x789abcdef0123456,
            0x89abcdef01234567, 0x9abcdef012345678,
            0xabcdef0123456789, 0xbcdef0123456789a,
            0xcdef0123456789ab, 0xdef0123456789abc,
            0xef0123456789abcd, 0xf0123456789abcde
        ]
        self.HSIZE = hsize 
        self.SIZE = size  
        self.opts = opts

    def run(self, out_dir):
        for split in ['train', 'valid', 'test']:
            if split[-1] == 'n':
                is_train = 1
            else:
                is_train = 0
            # n_instances = getattr(self.opts, f'{split}_instances')
            n_instances = getattr(self.opts, f'{split}_instances')
            # n_st = getattr(self.opts, f'{split}_st')
            if n_instances > 0:
                sat_out_dir_cnf = os.path.join(os.path.abspath(out_dir), f'cnf/{split}/sat')
                unsat_out_dir_cnf = os.path.join(os.path.abspath(out_dir), f'cnf/{split}/unsat')
                sat_out_dir_anf = os.path.join(os.path.abspath(out_dir), f'anf/{split}/sat')
                unsat_out_dir_anf = os.path.join(os.path.abspath(out_dir), f'anf/{split}/unsat')
                plain_cipher_out_dir = os.path.join(os.path.abspath(out_dir), f'plain_cipher/{split}')
                os.makedirs(sat_out_dir_cnf, exist_ok=True)
                os.makedirs(unsat_out_dir_cnf, exist_ok=True)
                os.makedirs(sat_out_dir_anf, exist_ok=True)
                os.makedirs(unsat_out_dir_anf, exist_ok=True)
                os.makedirs(plain_cipher_out_dir, exist_ok=True)
                print(f'Generating simon {split} set...')
                for i in tqdm(range(n_instances)):
                    # retry_count = 0
                    # while not self.generate(i, sat_out_dir_cnf, unsat_out_dir_cnf, sat_out_dir_anf, unsat_out_dir_anf):
                    #     retry_count += 1
                    #     if retry_count > MAX_RETRIES:
                    #         break
                    while not self.generate(i, sat_out_dir_cnf, unsat_out_dir_cnf, sat_out_dir_anf, unsat_out_dir_anf, plain_cipher_out_dir):
                        print(f"generating {i} failure")
                        pass

    def srand32(self):
        for i in range(31):
            self.ABC[i] ^= (self.seed + i) * i
        for i in range(256):
            self.ABC[(31 + i) & 0x1f] = self.ABC[i & 0x1f] + self.ABC[(i + 3) & 0x1f]

    def rand32(self):
        for i in range(32):
            self.ABC[(31 + i) & 0x1f] = self.ABC[i & 0x1f] + self.ABC[(i + 3) & 0x1f]
        return self.ABC[0] >> 32
        # return self.ABC[0] & 0xFFFFFFFF

    def add_xor_sat(self, a, b, c, clauses):
        clauses.append([-a, b, c, 0])
        clauses.append([a, -b, c, 0])
        clauses.append([a, b, -c, 0])
        clauses.append([-a, -b, -c, 0])
        return 4, 16

    def add_and_sat(self, a, b, c, clauses):
        clauses.append([a, -b, -c, 0])
        clauses.append([-a, b, 0])
        clauses.append([-a, c, 0])
        return 3, 10

    def simon_onerun(self, plain, cipher, key):
        for i in range(self.HSIZE, self.SIZE):
            cipher[i] = plain[i - self.HSIZE]
        for i in range(self.HSIZE):
            cipher[i] = (plain[(i + 1) % self.HSIZE] & plain[(i + 8) % self.HSIZE]) ^ plain[(i + 2) % self.HSIZE]
        for i in range(self.HSIZE):
            cipher[i] ^= (plain[i + self.HSIZE] ^ key[i])

    def simon_mulruns(self, plain, cipher, key, rounds):
        for i in range(rounds): #0,2,4
            if i % 2 == 0:
                self.simon_onerun(plain, cipher, key)
            else:   #1,3,5
                self.simon_onerun(cipher, plain, key)
            # print(f'{i}: {cipher} {plain}')
        if rounds % 2 == 0:
            cipher[:] = plain[:]
            
    # def simon_onerun(plain, cipher, key):
    #     for i in range(HSIZE, SIZE):
    #         cipher[i] = plain[i - HSIZE]
    #     for i in range(HSIZE):
    #         cipher[i] = (plain[(i + 1) % HSIZE] & plain[(i + 8) % HSIZE]) ^ plain[(i + 2) % HSIZE]
    #     for i in range(HSIZE):
    #         cipher[i] ^= (plain[i + HSIZE] ^ key[i])

    # def simon_mulruns(plain, cipher, key, RN):
    #     for i in range(RN):
    #         if i % 2 == 0:
    #             simon_onerun(plain, cipher, key)
    #         else:
    #             simon_onerun(cipher, plain, key)
    #     if RN % 2 == 0:
    #         cipher[:] = plain[:]

    def generate_key_sat(self, key_vars, clauses, var_count, clause_count, literal_count):
        for i in range(self.HSIZE):
            var_count += 1
            key_vars[i] = var_count
        return var_count, clause_count, literal_count

    def generate_simon_sat(self, data_out, data_in, key_vars, clauses, var_count, clause_count, literal_count):
        varnum, clanum, litnum = var_count, clause_count, literal_count
        plain = [varnum + i + 1 for i in range(self.SIZE)]
        cipher = [0] * self.SIZE
        
        for i in range(self.SIZE):
            varnum += 1
            clauses.append([(data_in[i] * 2 - 1) * plain[i], 0])
            clanum += 1
            litnum += 2
        
        for round_idx in range(self.rounds):
            for idx in range(self.HSIZE):
                i = (idx + round_idx * self.HSIZE) % self.HSIZE
                varnum += 1
                c, l = self.add_and_sat(varnum, plain[(i + 1) % self.HSIZE], plain[(i + 8) % self.HSIZE], clauses)
                litnum += l
                clanum += c
                cipher[i] = varnum

                varnum += 1
                c, l = self.add_xor_sat(varnum, cipher[i], plain[(i + 2) % self.HSIZE], clauses)
                litnum += l
                clanum += c
                cipher[i] = varnum

                varnum += 1
                c, l = self.add_xor_sat(varnum, cipher[i], plain[i + self.HSIZE], clauses)
                litnum += l
                clanum += c
                cipher[i] = varnum
                
                varnum += 1
                c, l = self.add_xor_sat(varnum, cipher[i], key_vars[i], clauses)
                litnum += l
                clanum += c
                cipher[i] = varnum

            cipher[self.HSIZE:] = plain[:self.HSIZE]
            plain[:] = cipher[:]
        
        for i in range(self.SIZE):
            clauses.append([(data_out[i] * 2 - 1) * cipher[i], 0])     
            # print([(cipher[i] * 2 - 1) * plain[i], 0])
            clanum += 1
            litnum += 2
        
        return varnum, clanum, litnum

    def get_clause_str(self, clause_str, T_clause):     
        if T_clause == True:
            return clause_str + f' T 0\n'
        else:
            return clause_str + f' 0\n'
        
    def check_sat_anf(self, filename):
        command = f'{WDSat_path} -i {filename} -x'
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
            output = result.stdout.decode()
            # print(output)
            if 'UNSAT' in output:
                return False
            else:
                return True
        except:
            print("Error occur")
            
    # def check_sat_cnf(self, filename):
    #     command = f'{kissat_path} {filename}'
    #     try:
    #         result = subprocess.run(command, capture_output=True, text=True, timeout=10000)
    #         if result.returncode == 10:
    #             return True
    #         elif result.returncode == 20:
    #             return False
    #         else:
    #             return None
    #     except subprocess.TimeoutExpired:
    #         print("Kissat timed out")
    #         return None
    # [
    #     './kissat-rel-3.1.0-restart/build/kissat',
    #     f'--restartint={int(restartint)}',
    #     f'--reduceint={int(reduceint)}',
    #     f'--decay=1',       
    #     '--time=10000',
    #     sample_path
    # ]
        
    def check_sat_cnf(self, filename):
        command = [
            kissat_path,
            filename
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10000)
            # print(result)
            if result.returncode == 10:
                return True
            elif result.returncode == 20:
                return False
            else:
                print(f"File {filename} error {result.returncode}")
                return None
        except subprocess.TimeoutExpired:
            print(f"File {filename} on Kissat time out")
            return None
        except Exception as e:
            print(f"Exception error：{e}")
            return None


    def generate_anf(self, x_input, k_input, c_output):
        key_length = self.HSIZE
        rounds = self.rounds
        half_length = key_length
        PT_1 = deepcopy(x_input[:half_length])
        PT_2 = deepcopy(x_input[half_length:])
        CT_1 = [0 for _ in range(half_length)]
        CT_2 = [0 for _ in range(half_length)]
        c_output_1 = deepcopy(c_output[:half_length])
        c_output_2 = deepcopy(c_output[half_length:])
        ppre_idx = 0
        pre_idx = 0
        nxt_idx = 1 + half_length
        clauses_list = []
        # print(half_length)
        if rounds == 3:
            # print('generating 3 rounds')
            for i in range(1, rounds + 1):
                for idx in range(half_length):
                    re_clause = 'x'
                    T_clause = True
                    CT_1[idx] = ((PT_1[(idx + 1) % half_length] & PT_1[(idx + 8) % half_length]) +      #PT_1
                                PT_1[(idx + 2) % half_length] + PT_2[idx] + k_input[idx]) % 2
                    CT_2[idx] = PT_1[idx]   
                    # if i == rounds - 1:
                    #     CT_1[idx] = c_output[idx + half_length] 
                    # if i == rounds:
                    #     CT_1[idx] = c_output[idx]   
                    if i == 1:
                        re_clause += f' {idx + 1} {nxt_idx + idx}'
                        if ((PT_1[(idx + 1) % half_length] & PT_1[(idx + 8) % half_length]) +
                            PT_1[(idx + 2) % half_length] + PT_2[idx]) % 2 == 1:
                            T_clause = False
                        else:
                            T_clause = True
                    elif i == 2:
                        re_clause += f' .2 {pre_idx + ((idx + 1) % half_length)} {pre_idx + ((idx + 8) % half_length)} {pre_idx + ((idx + 2) % half_length)} {idx + 1}'
                        if (PT_2[idx] + c_output_2[idx]) % 2 == 1:
                            T_clause = False
                        else:
                            T_clause = True
                    elif i == 3:
                        re_clause += f" {ppre_idx + idx} {idx + 1}"
                        if ((c_output_2[(idx + 1) % half_length] & c_output_2[(idx + 8) % half_length]) + c_output_2[(idx + 2) % half_length] + c_output_1[idx]) % 2 == 1:
                            T_clause = False
                        else:
                            T_clause = True
                    clauses_list.append((re_clause, T_clause))
                for idx in range(half_length):
                    PT_1[idx] = CT_1[idx]
                    PT_2[idx] = CT_2[idx]
                ppre_idx = deepcopy(pre_idx)
                pre_idx = deepcopy(nxt_idx)
                nxt_idx += half_length
            n_vars = ppre_idx - 1   
            clause_idx = len(clauses_list)
            wrong_index = None     
        elif rounds > 3:
            # print(f'generating {rounds} rounds')
            for i in range(1, rounds + 1):
                # print("index:", i)
                for idx in range(half_length):
                    re_clause = 'x'
                    T_clause = True
                    CT_1[idx] = ((PT_1[(idx + 1) % half_length] & PT_1[(idx + 8) % half_length]) +      #PT_1
                                PT_1[(idx + 2) % half_length] + PT_2[idx] + k_input[idx]) % 2
                    CT_2[idx] = PT_1[idx]   

                    if i == 1:  # first round
                        re_clause += f' {idx + 1} {nxt_idx + idx}'
                        # print(((PT_1[(idx + 1) % half_length] & PT_1[(idx + 8) % half_length]) +
                        #     PT_1[(idx + 2) % half_length] + PT_2[idx]) % 2)
                        if ((PT_1[(idx + 1) % half_length] & PT_1[(idx + 8) % half_length]) +
                            PT_1[(idx + 2) % half_length] + PT_2[idx]) % 2 == 1:
                            T_clause = False
                        else:
                            T_clause = True
                    elif i == 2:
                        re_clause += f' .2 {pre_idx + ((idx + 1) % half_length)} {pre_idx + ((idx + 8) % half_length)} {pre_idx + ((idx + 2) % half_length)} {idx + 1} {nxt_idx + idx}'
                        if PT_2[idx] == 1:
                            T_clause = False
                        else:
                            T_clause = True
                    elif i == (rounds - 1):   
                        # CT_1 = deepcopy(c_output[half_length:])
                        # re_clause += f' .2 {pre_idx + ((idx + 1) % half_length)} {ppre_idx + ((idx + 8) % half_length)} {pre_idx + ((idx + 2) % half_length)} {ppre_idx + idx} {idx + 1}'
                        re_clause += f' .2 {pre_idx + ((idx + 1) % half_length)} {pre_idx + ((idx + 8) % half_length)} {pre_idx + ((idx + 2) % half_length)} {ppre_idx + idx} {idx + 1}'
                        if c_output_2[idx] == 1:
                            T_clause = False
                        else:
                            T_clause = True                    
                    elif i == rounds:   
                        # CT_1 = deepcopy(c_output[:half_length]) 
                        # CT_2 = deepcopy(c_output[half_length:])
                        #re_clause += f' .2 {pre_idx + ((idx + 1) % half_length)} {ppre_idx + ((idx + 8) % half_length)} {pre_idx + ((idx + 2) % half_length)} {ppre_idx + idx} {idx + 1}'
                        re_clause += f' {ppre_idx + idx} {idx + 1}'
                        # print(CT_1, CT_2)
                        # print(((((CT_2[(idx + 1) % half_length] & CT_2[(idx + 8) % half_length]) + CT_2[(idx + 2) % half_length]) % 2) + CT_1[idx]) % 2)
                        if ((c_output_2[(idx + 1) % half_length] & c_output_2[(idx + 8) % half_length]) + c_output_2[(idx + 2) % half_length] + c_output_1[idx]) % 2 == 1:
                            T_clause = False
                        else:
                            T_clause = True
                    else:   #ppre -> pre, pre -> now
                        re_clause += f' .2 {pre_idx + ((idx + 1) % half_length)} {pre_idx + ((idx + 8) % half_length)} {pre_idx + ((idx + 2) % half_length)} {ppre_idx + idx} {idx + 1} {nxt_idx + idx}'
                        T_clause = True
                    clauses_list.append((re_clause, T_clause))
                for idx in range(half_length):
                    PT_1[idx] = CT_1[idx]
                    PT_2[idx] = CT_2[idx]
                ppre_idx = deepcopy(pre_idx)
                pre_idx = deepcopy(nxt_idx)
                nxt_idx += half_length
            n_vars = ppre_idx - 1
            clause_idx = len(clauses_list)
            wrong_index = None     
            # print(CT_1,CT_2)    #[0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0] [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]    加密没问题
        return n_vars, clause_idx, clauses_list                 
    
    def generate(self, index, sat_out_dir_cnf, unsat_out_dir_cnf, sat_out_dir_anf, unsat_out_dir_anf, plain_cipher_out_dir) -> bool:     #generate_cnf,  generate_anf,   加check.
        self.srand32()
        # folder_name = "sat_simon_round" if label == 1 else "unsat_simon_round"
        # output_dir_cnf = os.path.join("cnf", folder_name, f"{self.rounds}")
        # output_dir_anf = os.path.join("anf", folder_name, f"{self.rounds}")
        # os.makedirs(output_dir_cnf, exist_ok=True)
        # os.makedirs(output_dir_anf, exist_ok=True)

        # for t in range(num):
        key_origin = [self.rand32() & 1 for _ in range(self.HSIZE)]    
        plain_origin = [self.rand32() & 1 for _ in range(self.SIZE)]   
        mw = plain_origin[:]    
        
        # print(plain_origin)
        # print(key_origin)
        cipher_sat_origin = [0] * self.SIZE
        self.simon_mulruns(mw, cipher_sat_origin, key_origin, rounds=self.rounds)  
        

        cipher_unsat_origin = [self.rand32() & 1 for _ in range(self.SIZE)]  
        
        x_input = deepcopy(plain_origin)
        k_input = deepcopy(key_origin)
        c_output_sat = deepcopy(cipher_sat_origin)
        c_output_unsat = deepcopy(cipher_unsat_origin)
            
        #generating cnf sat
        # print("P:",plain_origin)     
        # print("K:",key_origin)       
        # print("C:",cipher_sat_origin)    
        '''
        P: [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        K: [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0]
        C: [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
            1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0
        '''
                                    
        #(cipher, plain)    #0，2，4： cipher  1，3，5： plain
        plain_cipher_fname = os.path.join(plain_cipher_out_dir, f"{index}.txt")
        with open(plain_cipher_fname, "w") as fs:
            fs.write(f"plaintext:{plain_origin}\n")
            fs.write(f"ciphertext:{cipher_sat_origin}\n")
            fs.write(f"key:{key_origin}")

        var_count, clause_count, literal_count = 0, 0, 0
        key_vars = [0] * self.HSIZE
        clauses = []
        plain = deepcopy(plain_origin)
        cipher_sat = deepcopy(cipher_sat_origin)
        # cipher_mw = deepcopy(cipher_sat_origin) #副本
        var_count, clause_count, literal_count = self.generate_key_sat(key_vars, clauses, var_count, clause_count, literal_count)
        var_count, clause_count, literal_count = self.generate_simon_sat(cipher_sat, plain, key_vars, clauses, var_count, clause_count, literal_count)

        dimacs_fname = os.path.join(sat_out_dir_cnf, f"{index}.cnf")
        with open(dimacs_fname, "w") as file:
            file.write(f"p cnf {var_count} {clause_count}\n")
            for clause in clauses:
                file.write(" ".join(map(str, clause)) + "\n")
                # print(clause)
        # cnf_sat = self.check_sat_cnf(dimacs_fname)  #True
        
        # #generating cnf unsat
        var_count, clause_count, literal_count = 0, 0, 0
        key_vars = [0] * self.HSIZE
        clauses = []
        plain = deepcopy(plain_origin)
        cipher_unsat = deepcopy(cipher_unsat_origin)
        var_count, clause_count, literal_count = self.generate_key_sat(key_vars, clauses, var_count, clause_count, literal_count)
        var_count, clause_count, literal_count = self.generate_simon_sat(cipher_unsat, plain, key_vars, clauses, var_count, clause_count, literal_count)

        dimacs_fname = os.path.join(unsat_out_dir_cnf, f"{index}.cnf")
        with open(dimacs_fname, "w") as file:
            file.write(f"p cnf {var_count} {clause_count}\n")
            for clause in clauses:
                file.write(" ".join(map(str, clause)) + "\n")
        
        # cnf_unsat = self.check_sat_cnf(dimacs_fname)
        
        # print(cnf_sat, cnf_unsat)
                
        #generating anf sat
        # print(c_output_sat, c_output_unsat)
        n_vars, clauses_count, clauses_list = self.generate_anf(x_input, k_input, c_output_sat)
        
        anf_fname = os.path.join(sat_out_dir_anf, f"{index}.cnf")
        with open(anf_fname, "w") as f:
            f.write(f'p cnf {n_vars} {clauses_count}\n')
            for idx in range(0, clauses_count):
                clause_str, T_clause = clauses_list[idx]
                f.write(self.get_clause_str(clause_str, T_clause))
                    
        # anf_sat = self.check_sat_anf(anf_fname)
        
        #generating anf unsat
        n_vars, clauses_count, clauses_list = self.generate_anf(x_input, k_input, c_output_unsat)
        
        anf_fname = os.path.join(unsat_out_dir_anf, f"{index}.cnf")
        with open(anf_fname, "w") as f:
            f.write(f'p cnf {n_vars} {clauses_count}\n')
            for idx in range(0, clauses_count):
                clause_str, T_clause = clauses_list[idx]
                f.write(self.get_clause_str(clause_str, T_clause))
                    
        # anf_unsat = self.check_sat_anf(anf_fname)
        # print(anf_sat, anf_unsat)
        # if cnf_sat == True and cnf_unsat == False and anf_sat == True and anf_unsat == False:   #Add checkpoint for the generating instances satisfiability.
        #     return True
        # return False
        return True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instances', type=int, default=0, help='The number of training instances')
    parser.add_argument('--valid_instances', type=int, default=0, help='The number of validating instances')
    parser.add_argument('--test_instances', type=int, default=0, help='The number of testing instances')
    opts = parser.parse_args()
    
    generater = Generator(0x0123456789abcdef, 10, 64, 32, opts)
    generater.run("./cnf_anf/simon-10-32-64")