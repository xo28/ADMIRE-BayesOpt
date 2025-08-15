import torch
import pandas as pd

def prepare_thepile_data(cls_idx, tkwargs):
    # 1M model train set
    df_config = pd.read_csv("./regmix-data/train_mixture_1m.csv")
    df_wandb = pd.read_csv("./regmix-data/train_pile_loss_1m.csv")
    X_train_1M = torch.tensor(df_config[df_config.columns[1:]].values, **tkwargs)
    y_train_1M = df_wandb[df_wandb.columns[1:]].values
    domain_name = df_wandb.columns[1:][cls_idx].split("/")[1]
    y_train_1M = -torch.tensor(y_train_1M[:,cls_idx], **tkwargs)
    # 1M model test set
    df_config = pd.read_csv("./regmix-data/test_mixture_1m.csv")
    df_wandb = pd.read_csv("./regmix-data/test_pile_loss_1m.csv")
    X_test_1M = torch.tensor(df_config[df_config.columns[1:]].values, **tkwargs)
    y_test_1M = df_wandb[df_wandb.columns[1:]].values
    y_test_1M = -torch.tensor(y_test_1M[:,cls_idx], **tkwargs)
    # 60M model test set
    df_config = pd.read_csv("./regmix-data/test_mixture_60m.csv")
    df_wandb = pd.read_csv("./regmix-data/test_pile_loss_60m.csv")
    X_60M = torch.tensor(df_config[df_config.columns[1:]].values, **tkwargs)
    y_60M = df_wandb[df_wandb.columns[1:]].values
    y_60M = -torch.tensor(y_60M[:,cls_idx], **tkwargs)
    # 1B model test set
    df_config = pd.read_csv("./regmix-data/test_mixture_1B.csv")
    df_wandb = pd.read_csv("./regmix-data/test_pile_loss_1B.csv")
    X_1B = torch.tensor(df_config[df_config.columns[1:]].values, **tkwargs)
    y_1B = df_wandb[df_wandb.columns[1:]].values
    y_1B = -torch.tensor(y_1B[:,cls_idx], **tkwargs)
    return X_train_1M, y_train_1M, X_60M, y_60M, X_1B, y_1B, domain_name

def prepare_admire_ift_runs(cls_idx, tkwargs):
    threshold = 60
    df = pd.read_csv("./admire_ift_runs/admire_ift_runs.csv")
    domain_name = df.columns[df.columns.str.startswith('metric')][cls_idx][7:]
    print(domain_name)
    model_groups = dict(tuple(df.groupby('model')))
    # Access individual DataFrames
    df_05B = model_groups.get('Qwen2.5-0.5B')
    df_3B = model_groups.get('Qwen2.5-3B')
    df_7B = model_groups.get('Qwen2.5-7B')

    # Display the first few rows of the DataFrame
    ratio_05B = torch.tensor(df_05B.loc[:, df_05B.columns.str.startswith('ratio')].values, **tkwargs)
    metric_05B = torch.tensor(df_05B.loc[:, df_05B.columns.str.startswith('metric')].values, **tkwargs)[:, cls_idx]
    ratio_3B = torch.tensor(df_3B.loc[:, df_3B.columns.str.startswith('ratio')].values, **tkwargs)
    metric_3B = torch.tensor(df_3B.loc[:, df_3B.columns.str.startswith('metric')].values, **tkwargs)[:, cls_idx]
    ratio_7B = torch.tensor(df_7B.loc[:, df_7B.columns.str.startswith('ratio')].values, **tkwargs)
    metric_7B = torch.tensor(df_7B.loc[:, df_7B.columns.str.startswith('metric')].values, **tkwargs)[:, cls_idx]
    return ratio_05B, metric_05B, ratio_3B, metric_3B, ratio_7B, metric_7B, domain_name, threshold


def add_fidelity(fidelity_weights, x_low, x_medium, x_high, y_low, y_medium, y_high, tkwargs):
    # Fidelity levels
    fidelity_low = torch.ones((x_low.shape[0], 1), **tkwargs) * fidelity_weights['low']
    fidelity_medium = torch.ones((x_medium.shape[0], 1), **tkwargs) * fidelity_weights['medium']
    fidelity_high = torch.ones((x_high.shape[0], 1), **tkwargs) * fidelity_weights['high']
    # Combine features with fidelity levels
    x_low = torch.cat([x_low, fidelity_low], dim=1)
    x_medium = torch.cat([x_medium, fidelity_medium], dim=1)
    x_high = torch.cat([x_high, fidelity_high], dim=1)
    # Concatenate all training data and all objective values
    x_with_fidelity = torch.cat([x_low, x_medium, x_high], dim=0)
    y = torch.cat([y_low, y_medium, y_high], dim=0).unsqueeze(-1)  # add output dimension
    return x_with_fidelity, x_low, x_medium, x_high, y
