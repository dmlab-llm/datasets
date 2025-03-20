import pandas as pd
import csv
import random

data_path = "../train_max_en_up.csv"
seed = 42
data_format = "squad_coreset_10_"

fraction = 0.1
rank = 8
batch_size = 2

def unbalanced_partition(df, rank, batch_size):
    high = rank // 2
    
    tot_len = len(df)
    half_len = tot_len // 2

    high_df = df[:half_len]
    low_df = df[half_len:]
    low_df = low_df.iloc[::-1]
    
    high_bs = high * batch_size
    
    result = pd.DataFrame()
    for i in range(half_len // high_bs):
        result = pd.concat([result, high_df[i*high_bs: (i+1)*high_bs], low_df[i*high_bs: (i+1)*high_bs]])

    tot = high_bs * (half_len // high_bs)
    if half_len > tot:
        result = pd.concat([result, high_df[tot:], low_df[tot:]])
    return result

def main():
    data = pd.read_csv(data_path)

    # random 10% coreset
    random_data = data.sample(frac=fraction, random_state=seed)
    random_data.to_csv(data_format + "rand.csv")

    # random 10% coreset sorted by semantic_uncertainty
    random_data_up = random_data.sort_values(by=["j_uncertainty"], ascending=True)
    random_data_up.to_csv(data_format + "rand_su_b_asc.csv")

    random_data_down = random_data.sort_values(by=["j_uncertainty"], ascending=False)
    random_data_down.to_csv(data_format + "rand_su_b_des.csv")

    # random 10% coreset sorted by semantic_uncertainty, unbalanced partition
    rand_data_un = unbalanced_partition(random_data_up, rank, batch_size)
    rand_data_un.to_csv(data_format + "rand_su_un.csv")

    # select by uncertainty for random partitioning and balanced partitioning
    data_len = int(len(data) * fraction)
    data_up = data.sort_values(by=["j_uncertainty"], ascending = True)
    data_up = data_up[:data_len]
    data_up.to_csv(data_format + "su_bot.csv")

    data_down = data.sort_values(by=["j_uncertainty"], ascending=False)
    data_down=data_down[:data_len]
    data_down.to_csv(data_format +"su_top.csv")


    data_len_half = int(len(data) * fraction / 2)
    data_topbot = pd.concat([data_up[:data_len_half], data_down[:data_len_half]])
    data_topbot.to_csv(data_format+"su_topbot.csv")


    # select by uncertainty for random partitioning and unbalanced partitioning
    data_up_half = data_up[:data_len_half]
    data_down_half = data_down[:data_len_half]
    data_topbot_half = pd.concat([data_up_half, data_down_half])
    data_topbot_un = unbalanced_partition(data_topbot_half, rank, batch_size)
    data_topbot_un.to_csv(data_format + "su_topbot_su_un.csv")


if __name__ == "__main__":
    main()
