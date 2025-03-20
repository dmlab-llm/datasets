import pandas as pd
import csv

data_path = f"train.csv"
uncertainty_paths = [f"keyword/llama3_sent_temp_1_arxiv_semantic_uncertainty_cluster_jaccard.csv", f"keyword/llama3_sent_temp_1_arxiv_semantic_uncertainty_cluster_llama.csv"]

indices = ["j_uncertainty", "l_uncertainty"]

for uncertainty_path, index in zip(uncertainty_paths, indices):
    dataset = pd.read_csv(data_path)
    uncertainty = pd.read_csv(uncertainty_path)

    dataset_context = dataset["context"]
    dataset_idx = dataset["idx"]
    uncertainty_context = uncertainty["context"]
    uncertainty_list = uncertainty["uncertainty"]
    uncertainty_idx = uncertainty["idx"]
    uncertainties = []
    cnt = 0
    print("start comparing")
    for context in dataset_idx:
        for unc_cont, unc in zip(uncertainty_idx, uncertainty_list):
            #print(unc_cont)
            if context == unc_cont:
                #print(cnt)
                cnt+=1
                uncertainties.append(unc)
                break
    dataset[index] = uncertainties

    dataset.sort_values(by=[index], ascending=True, inplace = True) #ascending=False: from big to small
    dataset.to_csv("train_r_"+index[0]+".csv")

    dataset.sort_values(by=[index], ascending=False, inplace = True) #ascending=False: from big to small
    dataset.to_csv("train_"+index[0]+".csv")
