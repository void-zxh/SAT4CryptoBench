import os
import subprocess
import time

TXT_FOLDER = "./simon-12-32-64/plain_cipher/test"
CNF_FOLDER = "./simon-12-32-64/anf/test/sat"
CHECKPOINT_PATH = "./ckpt.pt"

def get_cnf_path(txt_path):
    filename = os.path.basename(txt_path).replace(".txt", ".cnf")
    return os.path.join(CNF_FOLDER, filename)

def parse_prediction_output(output_str):
    for line in output_str.strip().splitlines():
        line = line.strip()
        if len(line) == 32 and all(c in '01' for c in line):
            return line
    return None

def predict_all_initializations(txt_files):
    init_dict = {}
    print("🧠 Generating initialization bitstrings using neural network...\n")

    for f in txt_files:
        txt_path = os.path.join(TXT_FOLDER, f)
        cnf_path = get_cnf_path(txt_path)

        if not os.path.exists(cnf_path):
            print(f"❌ Missing CNF file for {f}")
            continue

        predict_cmd = (
            f"python prediction.py assignment {cnf_path} {CHECKPOINT_PATH} "
            f"--graph anf --seed 123 --model cryptoanfnet --test_splits sat"
        )

        prediction_proc = subprocess.run(predict_cmd, shell=True, capture_output=True, text=True)
        if prediction_proc.returncode != 0:
            print(f"❌ Prediction failed for {f}:\n{prediction_proc.stderr}")
            continue

        bitstring = parse_prediction_output(prediction_proc.stdout)
        if bitstring is None:
            print(f"❌ Could not extract valid bitstring from prediction for {f}")
            continue

        init_dict[f] = bitstring

    return init_dict

def solve_with_initializations(txt_files, init_dict, init_time):
    total_times = []
    print("⚙️  Solving with predicted initializations...\n")

    for f in txt_files:
        if f not in init_dict:
            print(f"⚠️  Skipping {f}: no init bitstring found.")
            continue

        txt_path = os.path.join(TXT_FOLDER, f)
        bitstring = init_dict[f]

        start = time.time()
        try:
            subprocess.run(["./simon_anf", txt_path, bitstring], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end = time.time()
            total_times.append(end - start)
            print(f"⏱️  {f}: {end - start:.4f} seconds")
        except subprocess.CalledProcessError as e:
            print(f"❌ SIMON failed on {f}: {e}")

    if total_times:
        avg_time = (sum(total_times) + init_time) / len(total_times)
        print(f"\n📊 Average solving time (excluding prediction): {avg_time:.4f} seconds over {len(total_times)} files.")
    else:
        print("❌ No successful solves.")

def main():
    if not os.path.isdir(TXT_FOLDER):
        print(f"❌ TXT folder not found: {TXT_FOLDER}")
        return

    txt_files = sorted(f for f in os.listdir(TXT_FOLDER) if f.endswith(".txt"))
    if not txt_files:
        print("⚠️ No .txt files found.")
        return

    pre_start = time.time()
    init_dict = predict_all_initializations(txt_files)
    pre_end = time.time()
    init_time = pre_end - pre_start
    print(f"process_time: {init_time}")

    solve_with_initializations(txt_files, init_dict, init_time)

if __name__ == "__main__":
    main()