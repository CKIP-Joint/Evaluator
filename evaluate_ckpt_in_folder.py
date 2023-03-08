import os
import sys
import subprocess
import shutil
import time
import tqdm
from multiprocessing import Pool

# Note: currently we cannot activate all tasks on all cudas
# we need to use `python3 evaluate_ckpt_in_folder.py CUDA_NUM` multiple times
# This is becasue we cannot use pool in pool. Still finding better way

## TODO
# sort in every exp directory
# retrying
# distribute task to GPU


##### Modify here #####
helm_benchmark_output_path = "/nobackup/gpu_d_98001_t2/creative_plasma/mtk53587/benchmark_output_all_new"
symbols_lists = [
    ["23209"]
]
symbol_must_exist = ['global_step']
symbol_must_not_exist = ['hf', "HF", "MEGATRON"]
CUDA_DEVICES=[0]
task_num_per_CUDA_DEVICES=3

ckpts_root_folder = "/nobackup/gpu_d_98001_t2/creative_plasma/mtk53535"
match_patterns = [
    "\/[0-9]+\/",
    "global_step[0-9]+",
]
sleep_for_N_sec = 60
#######################
cuda_device_counter = {}
for CUDA_DEVICE in CUDA_DEVICES:
    cuda_device_counter[CUDA_DEVICE] = 0

MEGATRON_PATH = os.environ.get("MEGATRON_PATH")
HELM_PATH = os.environ.get("HELM_PATH")
bloom_1b1_folder = "/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloom-1B1/snapshots/1e718be072aa40714c9dc34e35ea6b64979a65ad/"

## check variables
assert len(HELM_PATH) > 0, "HELM_PATH is not specified!"
assert len(MEGATRON_PATH) > 0, "MEGATRON_PATH is not specified!"

def convert_ckpt(bloom_ckpt_path, HF_ckpt_path):
    print("Try to converting ...")
    convert_ckpt_script_path = os.path.join(MEGATRON_PATH, "tools/convert_checkpoint/convert_bloom_original_to_huggingface.py")
        
    arg_ckpt = f"--bloom_checkpoint_path={bloom_ckpt_path}"
    arg_hf_ckpt = f"--pytorch_dump_folder_path={HF_ckpt_path}"
    bloom_config_file = os.path.join(bloom_1b1_folder, "config.json")
    arg_bloom_config = f"--bloom_config_file {bloom_config_file}"
    arg_arch = "--pretraining_tp=1 "
    convert_ckpt_cmd = f'python3  {convert_ckpt_script_path} {arg_ckpt} {arg_hf_ckpt} {arg_bloom_config} {arg_arch}'
    subprocess.run(convert_ckpt_cmd, shell=True)
    subprocess.run(f"chmod -R 777 {HF_ckpt_path}", shell=True)

def copy_tokenizer_file(tgt_path):
    print("Copying tokenizer data ...")
    file_list = [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    for file in file_list:
        src = os.path.join(bloom_1b1_folder, file)
        tgt = os.path.join(tgt_path, file)
        if os.path.exists(tgt):
            os.remove(tgt)
        shutil.copy(src, tgt)

def eval_helm(output_path, model_id, load_path, CUDA_ID=0):
    print("Running HELM ...")
    #global cuda_device_counter
    current_path = os.getcwd()
    evaluation_cmd = f"CUDA_VISIBLE_DEVICES={CUDA_ID} python3 main.py --local --suite v1 -m=10000 -c=run_specs.conf --num-thread=1 --output-path={output_path} --model_id={model_id} --load_path={load_path}"
    logs_folder = os.path.join(output_path, 'logs')
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    log_file = os.path.join(logs_folder, f"{model_id}.log")
    try:
        #cuda_device_counter[CUDA_DEVICE] += 1
        os.chdir(HELM_PATH)
        cmd = f"{evaluation_cmd} > {log_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        #cuda_device_counter[CUDA_DEVICE] = max(cuda_device_counter[CUDA_DEVICE]-1, 0)
    except:
        #cuda_device_counter[CUDA_DEVICE] = max(cuda_device_counter[CUDA_DEVICE]-1, 0)
        print("Fail to run HELM")
    os.chdir(current_path)
    #subprocess.run(f"chmod -R 777 {HF_ckpt_path}", shell=True)
    
    
if __name__== "__main__":
    import re, itertools
    import sys

    all_combinations = list(itertools.product(*symbols_lists))
    subfolders_exists = [x[0] for x in os.walk(ckpts_root_folder, followlinks=True)]
    subfolders_exists.reverse()
    
    #all_subfolders_need_to_be_evaluted = [os.path.join(ckpts_root_folder, pattern.format(*combination)) for combination in all_combinations]
    all_subfolders_need_to_be_evaluted = []
    for subfolder in subfolders_exists:
        
        # Only when at least one symbols_lists (exp num) in subfolder we evaluate it 
        symbols_lists_matched = [False for _ in range(len(symbols_lists))]
        for i, symbols_list in enumerate(symbols_lists):
            for symbol in symbols_lists[i]:
                if f"/{symbol}/" in subfolder:
                    symbols_lists_matched[i] = True
                    break
        if False in symbols_lists_matched:
            remove = True
        else:
            remove = False

        # check matching rules
        for symbol in symbol_must_exist:
            if symbol not in subfolder:
                remove = True
                break
        for symbol in symbol_must_not_exist:
            if symbol in subfolder:
                remove = True
                break
        

        for match_pattern in match_patterns:
            found = re.search(match_pattern, subfolder)
            if not found:
                remove = True
                break

        if not remove:
            all_subfolders_need_to_be_evaluted.append(subfolder)

    #all_subfolders_need_to_be_evaluted = all_subfolders_need_to_be_evaluted[::4]
    #all_subfolders_need_to_be_evaluted = [f for f in all_subfolders_need_to_be_evaluted if f not in all_subfolders_need_to_be_evaluted[::4]]
    with open("to_be_evaluated_subfolder_list.txt", 'w') as f:
        f.writelines(line+'\n' for line in all_subfolders_need_to_be_evaluted)


    if len(sys.argv) > 1:
        task_per_cuda = len(all_subfolders_need_to_be_evaluted)//len(CUDA_DEVICES)
        tasks_split = [ all_subfolders_need_to_be_evaluted[i*task_per_cuda:(i+1)*task_per_cuda] for i in range(len(CUDA_DEVICES))]
        tasks_split[-1] = all_subfolders_need_to_be_evaluted[(len(CUDA_DEVICES)-1)*task_per_cuda:]

        CUDA_DEVICE = CUDA_DEVICES[int(sys.argv[1])]
        task_list = tasks_split[int(sys.argv[1])]

        with open(f"to_be_evaluated_subfolder_sublist_{sys.argv[1]}.txt", 'w') as f:
            f.writelines(line+'\n' for line in task_list)
        
        def task_set(subfolder):
            pattern_elements = []
            for match_pattern in match_patterns:
                found = re.search(match_pattern, subfolder)
                pattern_elements.append(found.group(0))
            model_id = "_".join([k.strip('/') for k in pattern_elements])
            if not os.path.exists(os.path.join(subfolder, "HF", "tokenizer_config.json")):
                convert_ckpt(subfolder, os.path.join(subfolder, "HF"))
                copy_tokenizer_file(os.path.join(subfolder, "HF"))
            eval_helm(helm_benchmark_output_path,  model_id=model_id, load_path=os.path.join(subfolder, "HF"), CUDA_ID=CUDA_DEVICE)

        print(f"Task num: {len(task_list)}")
        pool = Pool(task_num_per_CUDA_DEVICES)
        pool.map(task_set, task_list)
                