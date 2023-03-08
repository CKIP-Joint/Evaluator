import pandas as pd
import os, sys
csv_folder = "/proj/gpu_mtk53587/Desktop/gpu_d_98001_t2/creative_plasma/mtk53587/benchmark_output_all_new/runs/v1"
csv_files = [
    #"drcd.csv",
    "tcic.csv",
    "fgc.csv",
    #"lambada.csv",
    "sltp.csv",
    "tcwsc.csv",
]
sort_targets = ["prefix_exact_match", "exact_match", "f1_score"]

dfs = []
for csv_file in csv_files:
    dfs.append(pd.read_csv(os.path.join(csv_folder, csv_file)))


model_scores = {}
all_results = {}
for i, df in enumerate(dfs):
    sort_targets_ = [sort_target for sort_target in sort_targets if sort_target in df.columns]
    df_sorted = df.sort_values(by=sort_targets_, ascending=False)
    df_sorted = df_sorted[['model',*sort_targets_]]
    print(df_sorted.head())
    
    all_results[csv_files[i].split('.')[0]] = df_sorted.model.values
    for j, model in enumerate(df_sorted.model):
        if model not in model_scores.keys():
            model_scores[model] = [j]
        else:
            model_scores[model].append(j)


max_len = max([len(all_results[key]) for key in all_results.keys()])
for key in all_results.keys():
    if len(all_results[key])!=max_len:
        all_results[key] = all_results[key].tolist() + ['nan' for _ in range(max_len-len(all_results[key]))]
all_results = pd.DataFrame(all_results)
all_results.to_csv(os.path.join(csv_folder, 'ranking.csv'))

model_score_mean = {}
for model in model_scores.keys():
    if len(model_scores[model])>= 4:
        model_score_mean[model] = sum(model_scores[model])/len(model_scores[model])

model_score_mean = {k:v for k, v in sorted(model_score_mean.items(), key=lambda item: item[1])}
print(model_score_mean)
