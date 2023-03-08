import json
import sys
import pandas as pd
import sys
import os

try:
    runs_json_path = "/proj/gpu_mtk53587/Desktop/gpu_d_98001_t2/creative_plasma/mtk53587/benchmark_output_all_new/runs/v1/runs.json"
    runs_dic = json.load(open(runs_json_path, "r"))
except:
    runs_json_path = "benchmark_output/runs/v1/runs.json"
    runs_dic = json.load(open(runs_json_path, "r"))

scenerios = ['drcd', 'fgc', 'tcwsc', 'lambada', 'tcic', 'sltp']
wanted_metrics = ["f1_score", "exact_match", "prefix_exact_match", "quasi_exact_match"]

for scenerio in scenerios:

    results = {
        "model": [],
        "metrics":{
            
        }
    }

    for run in runs_dic:
        if scenerio in run['run_path']:
            model_name = run['run_spec']['name'].split('=')[-1]
            results["model"].append(model_name)
            # check metric exists in run
            for metric in wanted_metrics:
                metirc_specs = [key for spec in run['run_spec']['metric_specs'] for key in spec['args']['names'] ]
                if metric in metirc_specs and metric not in results["metrics"].keys() :
                    results["metrics"][metric] = []
        
            for metric in results["metrics"].keys():
                for stat in run['stats']:
                    if stat['name']['name'] == metric:
                        results["metrics"][metric].append(stat['sum'])
                        break

    for metric in results["metrics"].keys():
        results[metric] = results["metrics"][metric]
    results.pop('metrics', None)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(os.path.dirname(runs_json_path), f"{scenerio}.csv"))
        


