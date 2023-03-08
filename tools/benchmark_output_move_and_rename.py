import sys
import os 
import subprocess
import json

lines = []

try:
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()
except:
    with open('tools/output_to_be_moved.txt', "r") as f:
        lines = f.readlines()

try:
    output_dir = sys.argv[2]
except:
    output_dir = "/nobackup/gpu_d_98001_t2/creative_plasma/mtk53587/benchmark_output_all/runs/v1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for line in lines:
    ckpt, raname_name = line.split(',')
    eval_result_path = os.path.join(ckpt, 'runs', 'v1')
    subfolder_names = os.listdir(eval_result_path)
    for subfolder_name in subfolder_names:
        if "=" in subfolder_name:
            new_subfolder_name = '{}={}'.format(subfolder_name.split('=')[0], raname_name.strip())
            target_path = os.path.join(output_dir, new_subfolder_name)
            
            cmd = f"cp -r {os.path.join(eval_result_path, subfolder_name)} {target_path}"
            os.system(cmd)

            # modify "run_specs"
            run_spec = json.load(open(os.path.join(target_path, "run_spec.json"), "r"))
            run_spec['name'] = new_subfolder_name
            json.dump(run_spec, open(os.path.join(target_path, "run_spec.json"), "w"))



### move model in benchmark_output
subfolder_names = os.listdir("benchmark_output/runs/v1")
for subfolder_name in subfolder_names:
    if "=" in subfolder_name:
        original_path = os.path.join("benchmark_output/runs/v1", subfolder_name)
        target_path = os.path.join(output_dir, subfolder_name)
        
        cmd = f"cp -r {original_path} {target_path}"
        os.system(cmd)