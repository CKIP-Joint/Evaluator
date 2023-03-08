import sys, os, json, re, tqdm
import matplotlib.pyplot as plt
import numpy as np
### TODO:
# when the result num is large and there is new scenario_json_subfolder generated, 
# You should not always go through the previous folder again
###

if __name__ == "__main__":
    benchmark_output_path = "/nobackup/gpu_d_98001_t2/creative_plasma/mtk53587/benchmark_output_all_new/" #sys.argv[1]

    root_of_scenario_json_subfolders = os.path.join(benchmark_output_path, 'runs', 'v1')
    figure_save_path = os.path.join(benchmark_output_path, 'figures')
    if not os.path.exists(figure_save_path): os.makedirs(figure_save_path)

    scenarios_to_show = ['tcwsc', 'fgc', 'sltp']#, 'tcic', 'lambada', 'drcd']
    metrics = ["f1_score", "exact_match", "prefix_exact_match", "quasi_exact_match"]
    scenario_json_subfolders = os.listdir(root_of_scenario_json_subfolders)
    scenario_json_subfolders = [f for f in scenario_json_subfolders if "=" in f]

    all_results = {k:{} for k in scenarios_to_show}

    for scenario in scenarios_to_show:
        print(f"Processing {scenario}...")
        for subfolder in tqdm.tqdm(scenario_json_subfolders):
            if scenario in subfolder:
                re_pattern = f"{scenario}:model=[0-9]+_global_step[0-9]+"
                matched = re.search(re_pattern, subfolder)
                if matched: # sometimes there are other exps with different naming pattern so `matched` maybe none
                    exp_id: str = matched.group(0).split('=')[1].split('_')[0]
                    step: int = int(matched.group(0).split('global_step')[1])
                    stat_json_path = os.path.join(root_of_scenario_json_subfolders, subfolder, 'stats.json')
                    
                    # check stat_json is generated 
                    if not os.path.exists(stat_json_path):
                        continue
                    stat_dict = json.load(open(stat_json_path, "r"))
                    if not stat_dict:
                        continue

                    # starting to record value
                    for metric in metrics:
                        # check whehter subfolder is checked. This is useful for future application. (But redundant for current stage 2023/02/16)
                        if  metric in all_results[scenario].keys() \
                            and exp_id in all_results[scenario][metric].keys() \
                            and step in all_results[scenario][metric][exp_id].keys():
                            continue
                        
                        # Get value
                        value = None
                        for i, single_stat in enumerate(stat_dict):
                            if single_stat['name']['name']==metric:
                                value = single_stat['sum']
                                break

                        # record value
                        if value:
                            if metric not in all_results[scenario].keys():
                                all_results[scenario][metric] = {}
                            if exp_id not in all_results[scenario][metric].keys():
                                all_results[scenario][metric][exp_id] = {}
                            all_results[scenario][metric][exp_id][step] = value

    print("Ploting")
    for scenario in all_results.keys():
        for metric in all_results[scenario].keys():
            plt.figure()
            for exp_id in all_results[scenario][metric].keys():
                steps = np.array([k for k in all_results[scenario][metric][exp_id].keys()])
                v = np.array([v for k, v in all_results[scenario][metric][exp_id].items()])
                steps = steps - np.min(steps)
                arg = np.argsort(steps)
                plt.plot(steps[arg], v[arg])

            plt.title(f"{scenario}")
            plt.xlabel("Steps")
            plt.ylabel(f"{metric}")
            plt.legend(all_results[scenario][metric].keys())
            plt.savefig(os.path.join(figure_save_path, f"{scenario}_{metric}.png"))


    