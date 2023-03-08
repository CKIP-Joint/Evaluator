import pandas as pd
import os, sys
import numpy as np


csv_folder = "/proj/gpu_mtk53587/Desktop/gpu_d_98001_t2/creative_plasma/mtk53587/benchmark_output_all_new/runs/v1"
csv_files = [
    #"drcd.csv",
    #"tcic.csv",
    "fgc.csv",
    #"lambada.csv",
    "sltp.csv",
    "tcwsc.csv",
]
evaluate_targets = ["prefix_exact_match"]
min_threshold_model = "huggingface_bloomz_1b1"

dfs = []
for csv_file in csv_files:
    dfs.append(pd.read_csv(os.path.join(csv_folder, csv_file)))

model_scores = {}
for i, df in enumerate(dfs):

    #threshold_score=df[df['model'].str.match(min_threshold_model)][evaluate_targets[0]].iloc[0]

    df['rank'] = df[evaluate_targets[0]].rank(method='dense')
    df['rank'] -= 1

    df = df.sort_values(by='rank', ascending=False)
    df.reset_index()

    max_rank = np.max(df['rank'])
    for j, model in enumerate(df.model):
        if model not in model_scores.keys():
            model_scores[model] = [df['rank'].iloc[j]/max_rank]
        else:
            model_scores[model].append(df['rank'].iloc[j]/max_rank)

model_score_mean = {}
for model in model_scores.keys():
    if len(model_scores[model])==len(csv_files):
        model_score_mean[model] = sum(model_scores[model])/len(model_scores[model])           

sorted_dict = sorted(model_score_mean.items(), key=lambda item: item[1])
sorted_dict.reverse()

def get_exp_performance_ranking(top_k, sd):
    exps_out = []
    exps_ranked = []
    for k,v in sd:
        exp_id = k.split('_')[0]
        exps_ranked.append(exp_id)
        if exps_ranked.count(exp_id)==top_k:
            exps_out.append(exp_id)
    return exps_out

experiment_rankings ={
    f'top-1': get_exp_performance_ranking(1,sorted_dict),
    f'top-3': get_exp_performance_ranking(3,sorted_dict),
    f'top-5': get_exp_performance_ranking(5,sorted_dict),
}

model_score_mean_df = {
    'model':[k for k, v in sorted_dict],
    'average rank score':[v for k, v in sorted_dict],
}

model_score_mean_df = pd.DataFrame(model_score_mean_df)
model_score_mean_df.to_csv(os.path.join(csv_folder, 'checkpoint_ranking.csv'))

experiment_ranking_df = pd.DataFrame.from_dict(experiment_rankings, orient='index').transpose()
experiment_ranking_df.to_csv(os.path.join(csv_folder, 'experiment_ranking.csv'))

