'''
This implementaion is based on an official BoTorch tutorial: "Multi-fidelity Bayesian optimization with discrete fidelities using KG"
https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/. We followed its comparasions between BayesOpt and MFBayesOpt.
'''
import os
import csv
import math
import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import argparse
import datetime
from gpytorch.metrics import mean_squared_error
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.exceptions.errors import ModelFittingError
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.utils import project_to_target_fidelity
from load_data import add_fidelity
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Description of your script")
parser.add_argument("--idx", type=int, help="Test domain index")
parser.add_argument("--dataset", choices=['admire_ift_runs', 'pile'], type=str, help="which dataset? admire_ift_runs/pile")
args = parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def evaluate(gp, valX, valY):
    gp = gp.eval()
    valY_likehd = gp(valX)
    mse = mean_squared_error(valY_likehd, valY, squared=True).item()
    return mse

def initialize_model(train_x, train_obj, X_test_high, y_test_high, idx):
    model = SingleTaskMultiFidelityGP(train_x, train_obj, data_fidelity=idx)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    mse = evaluate(model, X_test_high, y_test_high)
    return mse, model

#====================== MFBayesOpt data preparation ========================
def prepare_admire_ift_runs():
    def prepare_admire_ift_runs_data(cls_idx, tkwargs):
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
        ratio_low = torch.tensor(df_05B.loc[:, df_05B.columns.str.startswith('ratio')].values, **tkwargs)
        metric_low = torch.tensor(df_05B.loc[:, df_05B.columns.str.startswith('metric')].values, **tkwargs)[:, cls_idx]
        ratio_med = torch.tensor(df_3B.loc[:, df_3B.columns.str.startswith('ratio')].values, **tkwargs)
        metric_med = torch.tensor(df_3B.loc[:, df_3B.columns.str.startswith('metric')].values, **tkwargs)[:, cls_idx]
        ratio_high = torch.tensor(df_7B.loc[:, df_7B.columns.str.startswith('ratio')].values, **tkwargs)
        metric_high = torch.tensor(df_7B.loc[:, df_7B.columns.str.startswith('metric')].values, **tkwargs)[:, cls_idx]
        fd7B = torch.tensor(df_7B.loc[:, df_7B.columns.str.startswith('train_time')].values, **tkwargs).mean().item()
        fd3B = torch.tensor(df_3B.loc[:, df_3B.columns.str.startswith('train_time')].values, **tkwargs).mean().item()
        fd05B = torch.tensor(df_05B.loc[:, df_05B.columns.str.startswith('train_time')].values, **tkwargs).mean().item()
        return ratio_low, metric_low, ratio_med, metric_med, ratio_high, metric_high, domain_name, threshold, {'low':fd05B/fd7B, 'medium':fd3B/fd7B, 'high':1}

    ratio_low, metric_low, ratio_med, metric_med, ratio_high, metric_high, domain_name, threshold, fidelity_weights = prepare_admire_ift_runs_data(args.idx, tkwargs)
    X_train_with_fidelity, X_train_low, X_train_medium, X_train_high, Y_train = add_fidelity(fidelity_weights, ratio_low, ratio_med, ratio_high, metric_low, metric_med, metric_high, tkwargs)
    X_test_high = X_train_high[threshold:]
    y_test_high = metric_high[threshold:]
    X_train_1B = X_train_with_fidelity[:-(X_train_high.shape[0]-threshold)]
    return X_train_with_fidelity, X_train_low, X_train_medium, X_train_high, X_test_high, y_test_high, Y_train, \
            ratio_low, metric_low, ratio_med, metric_med, ratio_high, metric_high, domain_name, threshold, fidelity_weights
            
def prepare_thepile():
    def prepare_data(cls_idx=8):
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

    fidelity_weights = {'low': 0.001, 'medium': 0.06, 'high': 1.0} # 1M/1B and 60M/1B
    ratio_1M, metric_1M, ratio_60M, metric_60M, ratio_1B, metric_1B, domain_name = prepare_data(args.idx)
    print(domain_name)
    fidelity_low = torch.ones((ratio_1M.shape[0], 1), **tkwargs) * fidelity_weights['low']
    fidelity_medium = torch.ones((ratio_60M.shape[0], 1), **tkwargs) * fidelity_weights['medium']
    fidelity_high = torch.ones((ratio_1B.shape[0], 1), **tkwargs) * fidelity_weights['high']
    X_train_low = torch.cat([ratio_1M, fidelity_low], dim=1)
    X_train_medium = torch.cat([ratio_60M, fidelity_medium], dim=1)
    X_train_high = torch.cat([ratio_1B, fidelity_high], dim=1)
    threshold = 48
    X_test_high = X_train_high[threshold:]
    y_test_high = metric_1B[threshold:]
    Y_train = torch.cat((metric_1M, metric_60M, metric_1B), dim=0) 
    X_train_with_fidelity = torch.cat([X_train_low, X_train_medium, X_train_high], dim=0)
    return X_train_with_fidelity, X_train_low, X_train_medium, X_train_high, X_test_high, y_test_high, Y_train, \
            ratio_1M, metric_1M, ratio_60M, metric_60M, ratio_1B, metric_1B, domain_name, threshold, fidelity_weights

#====================== MFBayesOpt conditions =======================
if args.dataset == 'admire_ift_runs':
    X_train_with_fidelity, X_train_low, X_train_medium, X_train_high, X_test_high, y_test_high, Y_train, \
            ratio_low, metric_low, ratio_med, metric_med, ratio_high, metric_high, \
                domain_name, threshold, fidelity_weights = prepare_admire_ift_runs()
    fd_idx = 19
elif args.dataset == 'pile':
    X_train_with_fidelity, X_train_low, X_train_medium, X_train_high, X_test_high, y_test_high, Y_train,\
            ratio_low, metric_low, ratio_med, metric_med, ratio_high, metric_high, \
                domain_name, threshold, fidelity_weights = prepare_thepile()
    fd_idx = 17
else:
    print('Unknown dataset...')
    exit()
bounds = torch.tensor([[0.0] * X_train_with_fidelity.shape[1], [1.0] * X_train_with_fidelity.shape[1]], **tkwargs)
target_fidelities = {fd_idx: 1.0}
cost_model = AffineFidelityCostModel(fidelity_weights={fd_idx: 1.0}, fixed_cost=0.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

#====================== Fitting hyperparameters =======================
torch.set_printoptions(precision=4, sci_mode=False)
NUM_INIT = 1 # number of initial data points

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_exact_candidate(idx, train_X=None, train_Y=None):
    candidate = train_X[idx]
    fidelity = None
    if math.isclose(candidate[-1], fidelity_weights['low'], rel_tol=1e-6, abs_tol=0.0):
        fidelity = fidelity_weights['low']
    elif math.isclose(candidate[-1], fidelity_weights['medium'], rel_tol=1e-6, abs_tol=0.0):
        fidelity = fidelity_weights['medium']
    elif math.isclose(candidate[-1], fidelity_weights['high'], rel_tol=1e-6, abs_tol=0.0):
        fidelity = fidelity_weights['high']
    else:
        raise ValueError("Invalid fidelity level")
    new_x = train_X[idx].unsqueeze(0)
    new_obj = train_Y[idx].reshape(1, 1)

    mask = torch.ones(train_X.shape[0], dtype=torch.bool).to(tkwargs['device'])
    mask[idx] = False
    train_X = train_X[mask]
    train_Y = train_Y[mask]
    return new_x, new_obj, train_X, train_Y, fidelity

def optimize_mf_and_get_observation(mf_acqf, X_train, y_train):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""
    acq_eval_value = mf_acqf(X_train.unsqueeze(1))
    max_idx = torch.argmax(acq_eval_value).item()
    # print(f"max id: {max_idx} | max val: {acq_eval_value[max_idx].item():.4f}")

    cost = cost_model(X_train[max_idx, None, :]).sum()
    new_x, new_obj, X_train, y_train, fd = get_exact_candidate(max_idx, X_train, y_train)
    return new_x, new_obj, cost, X_train, y_train, fd

def run_mfbayesopt(X_train, y_train, num_iter):
    acq_stats = {fidelity_weights['low']:0, fidelity_weights['medium']:0, fidelity_weights['high']:0}
    cumulative_cost = [NUM_INIT*fidelity_weights['low']]
    mses = []
    rec_ys = []
    rec_xs = []
    train_x_batch = []
    train_obj_batch = []
    # initialize with NUM_INIT data points randomly
    for _ in range(NUM_INIT):
        idx = random.randint(0, X_train_low.shape[0]-1) # initialize with the lowest fidelity
        train_x, train_obj, X_train, y_train, fd = get_exact_candidate(idx, X_train, y_train)
        acq_stats[fd] += 1
        acq_orders = [fd]
        best_ys = [train_obj.max().item()]
        train_x_batch.append(train_x)
        train_obj_batch.append(train_obj)
    train_x = torch.stack(train_x_batch, dim=0).squeeze(1)
    train_obj = torch.stack(train_obj_batch, dim=0).squeeze(1)

    # Fit and sample
    for i in tqdm(range(num_iter)):
        mse, model = initialize_model(train_x, train_obj, X_test_high, y_test_high, fd_idx)
        mses.append(mse)
        mf_acqf = qMultiFidelityMaxValueEntropy(model=model, num_mv_samples=1, cost_aware_utility=cost_aware_utility, project=project, candidate_set=train_x)
        new_x, new_obj, cost, X_train, y_train, fd = optimize_mf_and_get_observation(mf_acqf, X_train, y_train)
        acq_stats[fd] += 1
        acq_orders.append(fd)
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        best_ys.append(train_obj.max().item())
        cumulative_cost.append(cost.item() + cumulative_cost[-1])

        # Make a final recommendation
        rec_y, rec_x = get_recommendation(mf_acqf)
        rec_ys.append(rec_y)
        rec_xs.append(rec_x.tolist())
        print(f"mse: {mse:.4f} | recmd: {np.array(rec_ys).max():.4f} | {acq_stats} ")
        print(f"cumulative cost: {cumulative_cost[-1]:.4f}")

        # save intermediate results
        filename = f'saved_logs/MFBayesOpt/maxvalue-{timestamp}-{args.dataset}_{num_runs}_{domain_name}_{NUM_INIT}.csv'
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i+1, rec_y, cost.item(), mse, rec_x.tolist()])

        if i == num_iter - 1:
            mse, model = initialize_model(train_x, train_obj, X_test_high, y_test_high, fd_idx)
            mses.append(mse)
            # Make a final recommendation
            rec_y, rec_x = get_recommendation(mf_acqf)
            rec_ys.append(rec_y)
            rec_xs.append(rec_x.tolist())
            print(f"mse: {mse:.4f} | recmd: {np.array(rec_ys).max():.4f} | {acq_stats} ")
            print(f"cumulative cost: {cumulative_cost[-1]:.4f}")

            # ===save intermediate results
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i+2, rec_y, cost.item(), mse, rec_x.tolist()])

    return mses, rec_ys, cumulative_cost, acq_orders, acq_stats, [train_x.tolist(), train_obj.tolist()], rec_xs

def get_recommendation(rec_acqf):
    values = rec_acqf(X_train_high[:threshold].unsqueeze(1))
    rec_id = torch.argmax(values).item()
    # print(f"{rec_id}/{torch.argmax((metric_high[:threshold])).item()} | {metric_high[rec_id].item():.4f} | {values}")
    return metric_high[rec_id].item(), ratio_high[rec_id]

##################################### average runs ###################################################
mfbayesopt_MSEs = []
mfbayesopt_COST = []
mfbayesopt_Ys = []
mfbayesopt_ACQ = []
mfbayesopt_Xs = []

num_runs = 5
for i in tqdm(range(num_runs)):
    seed = torch.randint(0, 100, (1,)).item() 
    seed_everything(seed)
    # remove test dataset by slicing w/ [:-(X_train_high.shape[0]-threshold)]
    mfbayesopt_mses, mfbayesopt_ys, mfbayesopt_cost, sample_orders, sample_stats, mfbayesopt_acq, mfbayesopt_xs = \
        run_mfbayesopt(X_train_with_fidelity[:-(X_train_high.shape[0]-threshold)], Y_train[:-(X_train_high.shape[0]-threshold)], Y_train[:-(X_train_high.shape[0]-threshold)].shape[0]-NUM_INIT)
    mfbayesopt_MSEs.append(np.array(mfbayesopt_mses))
    mfbayesopt_COST.append(np.array(mfbayesopt_cost))
    mfbayesopt_Ys.append(np.array(mfbayesopt_ys))
    mfbayesopt_ACQ.append(mfbayesopt_acq)
    mfbayesopt_Xs.append(mfbayesopt_xs)
mfbayesopt_mses = np.mean(mfbayesopt_MSEs, axis=0)

#=========== Save file
arrays = {'MFBayesOpt': [np.array(mfbayesopt_COST), np.array(mfbayesopt_Ys)]}
arrays['MFBayesOpt MSE'] = [list(range(len(mfbayesopt_mses))), np.array(mfbayesopt_MSEs)]
arrays['MFBayesOpt acq'] = mfbayesopt_ACQ
arrays['MFBayesOpt recX'] = mfbayesopt_Xs
filename = f'saved_logs/MFBayesOpt/MAXVALUE-{timestamp}-{args.dataset}_{num_runs}_{domain_name}_{NUM_INIT}.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    for k, vs in arrays.items():
        write_list = [k]
        for v in vs:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            write_list.append(v)
        writer.writerow(write_list)






