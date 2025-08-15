'''
This implementaion is based on an official BoTorch tutorial: "Multi-fidelity Bayesian optimization with discrete fidelities using KG"
https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/. We followed its comparasions between BayesOpt and MFBayesOpt.
'''
import os
import csv
import torch
import random
from tqdm import tqdm
import numpy as np
import warnings
import argparse
import datetime
from gpytorch.metrics import mean_squared_error
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.acquisition import qLogExpectedImprovement
from load_data import prepare_thepile_data
warnings.filterwarnings("ignore")

torch.set_printoptions(precision=3, sci_mode=False)
parser = argparse.ArgumentParser(description="Description of your script")
parser.add_argument("--idx", type=int, help="Test domain index")
args = parser.parse_args()
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate(gp, valX, valY):
    gp = gp.eval()
    valY_likehd = gp(valX)
    mse = mean_squared_error(valY_likehd, valY, squared=True).item()
    return mse

def initialize_model(train_x, train_obj, X_test_high, y_test_high, idx=17):
    model = SingleTaskMultiFidelityGP(train_x, train_obj, data_fidelity=idx)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    mse = evaluate(model, X_test_high, y_test_high)
    return mse, model

# Used for the GP kernel only; not used in acquisition or recommendation. Always the same value for data with the same fidelity.
fidelity_weights = {'low': 0.001, 'medium': 0.06, 'high': 1.0}
ratio_1M, metric_1M, ratio_60M, metric_60M, ratio_1B, metric_1B, domain_name = prepare_thepile_data(args.idx, tkwargs)
# Fidelity levels
fidelity_low = torch.ones((ratio_1M.shape[0], 1), **tkwargs) * fidelity_weights['low']
fidelity_medium = torch.ones((ratio_60M.shape[0], 1), **tkwargs) * fidelity_weights['medium']
fidelity_high = torch.ones((ratio_1B.shape[0], 1), **tkwargs) * fidelity_weights['high']
# Combine features with fidelity levels
X_train_low = torch.cat([ratio_1M, fidelity_low], dim=1)
X_train_medium = torch.cat([ratio_60M, fidelity_medium], dim=1)
X_train_high = torch.cat([ratio_1B, fidelity_high], dim=1)
# divide the largest 1B model dataset into train and test
threshold = 48
X_test_high = X_train_high[threshold:]
y_test_high = metric_1B[threshold:]
# Concatenate all training data and all objective values
X_train_with_fidelity = torch.cat([X_train_low, X_train_medium, X_train_high[:threshold]], dim=0)

bounds = torch.tensor([[0.0] * X_train_with_fidelity.shape[1], [1.0] * X_train_with_fidelity.shape[1]], **tkwargs)
cost_model = AffineFidelityCostModel(fidelity_weights={17: 1.0}, fixed_cost=0.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
target_fidelities = {17: 1.0}
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
NUM_INIT = 1

def get_candidate(train_X, train_Y, candidate=None, closest_idx=None):
    """Get the closest candidate to the training data."""
    if  closest_idx is None:
        try:
            closest_idx = torch.nonzero(torch.all(candidate == train_X, dim=1)).item()
        except:
            closest_idx = torch.norm(candidate - train_X, dim=1).argmin()
    new_x = train_X[closest_idx].unsqueeze(0)
    new_obj = train_Y[closest_idx].reshape(1,1)

    mask = torch.ones(train_X.shape[0], dtype=torch.bool).to(tkwargs['device'])
    mask[closest_idx] = False
    train_X = train_X[mask]
    train_Y = train_Y[mask]
    return new_x, new_obj, train_X, train_Y

def get_ei(model, best_f, fidelity=1):
    return FixedFeatureAcquisitionFunction(
        acq_function=qLogExpectedImprovement(model=model, best_f=best_f), 
        d=X_train_with_fidelity.shape[1], 
        columns=[X_train_with_fidelity.shape[1]-1], 
        values=[fidelity])

def optimize_ei_and_get_observation(ei_acqf, X_train, y_train):
    """Optimizes EI and returns a new candidate, observation, and cost."""
    candidate, _ = optimize_acqf_discrete(
        acq_function=ei_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
        choices=X_train[:, :-1] # Exclude the last fidelity column
    )

    # add the fidelity parameter
    candidate = ei_acqf._construct_X_full(candidate)
    # observe new values
    cost = cost_model(candidate).sum()
    new_x, new_obj, X_train, y_train = get_candidate(X_train, y_train, candidate=candidate)
    return new_x, new_obj, cost, X_train, y_train

def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=X_train_with_fidelity.shape[1],
        columns=[X_train_with_fidelity.shape[1]-1],
        values=[1],
    )

    values = rec_acqf(ratio_1B[:threshold].unsqueeze(1)) # recommend without fidelity
    rec_id = torch.argmax(values).item()
    return metric_1B[rec_id].item(), ratio_1B[rec_id]

def run_standardbayesopt(X_train, y_train, fidelity, num_iter):
    cumulative_cost = [NUM_INIT*fidelity]
    mses = []
    rec_ys = []
    rec_xs = []
    train_x_batch = []
    train_obj_batch = []
    for _ in range(NUM_INIT):
        idx = random.randint(0, X_train.shape[0]-1)
        train_x, train_obj, X_train, y_train = get_candidate(X_train, y_train, closest_idx=idx)
        best_ys = [train_obj.max().item()]
        train_x_batch.append(train_x)
        train_obj_batch.append(train_obj)
    train_x = torch.stack(train_x_batch, dim=0).squeeze(1)
    train_obj = torch.stack(train_obj_batch, dim=0).squeeze(1)

    for i in tqdm(range(num_iter)):
        mse, model = initialize_model(train_x, train_obj, X_test_high, y_test_high)
        mses.append(mse)
        ei_acqf = get_ei(model, best_f=train_obj.max(), fidelity=fidelity)
        new_x, new_obj, cost, X_train, y_train = optimize_ei_and_get_observation(ei_acqf, X_train, y_train)
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        best_ys.append(train_obj.max().item())
        cumulative_cost.append(cost.item() + cumulative_cost[-1])
        # Make a final recommendation
        rec_y, rec_x = get_recommendation(model)
        rec_ys.append(rec_y)
        rec_xs.append(rec_x.tolist())
        print(f"mse: {mse:.4f} | recmd: {rec_y:.4f}")
        print(f"cumulative cost: {cumulative_cost[-1]:.4f}")

        if i == num_iter - 1:
            mse, model = initialize_model(train_x, train_obj, X_test_high, y_test_high)
            mses.append(mse)
            # Make a final recommendation
            rec_y, rec_x = get_recommendation(model)
            rec_ys.append(rec_y)
            rec_xs.append(rec_x.tolist())
            print(f"mse: {mse:.4f} | recmd: {rec_y:.4f}")
            print(f"cumulative cost: {cumulative_cost[-1]:.4f}")

    return mses, rec_ys, cumulative_cost, [train_x.tolist(), train_obj.tolist()], rec_xs

##################################### average runs ###################################################
bayesopt_MSEs_1B = []
bayesopt_MSEs_60M = []
bayesopt_MSEs_1M = []
bayesopt_Ys_1B = []
bayesopt_Ys_60M = []
bayesopt_Ys_1M = []
bayesopt_ACQ_1B = []
bayesopt_ACQ_60M = []
bayesopt_ACQ_1M = []
bayesopt_Xs_1B = []
bayesopt_Xs_60M = []
bayesopt_Xs_1M = []

num_runs = 5
for i in range(num_runs):
    seed = torch.randint(0, 100, (1,)).item() 
    seed_everything(seed)
    bayesopt_mses_1B, bayesopt_ys_1B, bayesopt_cost_1B, bayesopt_acq_1B, bayesopt_xs_1B = run_standardbayesopt(X_train_high[:threshold], metric_1B[:threshold], fidelity_weights['high'], threshold-NUM_INIT)
    bayesopt_mses_60M, bayesopt_ys_60M, bayesopt_cost_60M, bayesopt_acq_60M, bayesopt_xs_60M = run_standardbayesopt(X_train_medium, metric_60M, fidelity_weights['medium'], 100-NUM_INIT)
    bayesopt_mses_1M, bayesopt_ys_1M, bayesopt_cost_1M, bayesopt_acq_1M, bayesopt_xs_1M = run_standardbayesopt(X_train_low, metric_1M, fidelity_weights['low'], 100-NUM_INIT)
    bayesopt_MSEs_1B.append(np.array(bayesopt_mses_1B))
    bayesopt_MSEs_60M.append(np.array(bayesopt_mses_60M))
    bayesopt_MSEs_1M.append(np.array(bayesopt_mses_1M))
    bayesopt_Ys_1B.append(np.array(bayesopt_ys_1B)) 
    bayesopt_Ys_60M.append(np.array(bayesopt_ys_60M))
    bayesopt_Ys_1M.append(np.array(bayesopt_ys_1M))
    bayesopt_ACQ_1B.append(bayesopt_acq_1B)
    bayesopt_ACQ_60M.append(bayesopt_acq_60M)
    bayesopt_ACQ_1M.append(bayesopt_acq_1M)
    bayesopt_Xs_1B.append(bayesopt_xs_1B)
    bayesopt_Xs_60M.append(bayesopt_xs_60M)
    bayesopt_Xs_1M.append(bayesopt_xs_1M)

bayesopt_mses_1B = np.mean(bayesopt_MSEs_1B, axis=0)
bayesopt_mses_60M = np.mean(bayesopt_MSEs_60M, axis=0)
bayesopt_mses_1M = np.mean(bayesopt_MSEs_1M, axis=0)

##################################### save ###################################################
arrays = {
    'BayesOpt 1B': [bayesopt_cost_1B, np.array(bayesopt_Ys_1B)],
    'BayesOpt 60M': [bayesopt_cost_60M, np.array(bayesopt_Ys_60M)],
    'BayesOpt 1M': [bayesopt_cost_1M, np.array(bayesopt_Ys_1M)],
}

arrays['BayesOpt 1B MSE'] = [list(range(len(bayesopt_mses_1B))), np.array(bayesopt_MSEs_1B)]
arrays['BayesOpt 60M MSE'] = [list(range(len(bayesopt_mses_60M))), np.array(bayesopt_MSEs_60M)]
arrays['BayesOpt 1M MSE'] = [list(range(len(bayesopt_mses_1M))), np.array(bayesopt_MSEs_1M)]
arrays['BayesOpt 1B acq'] = bayesopt_ACQ_1B
arrays['BayesOpt 60M acq'] = bayesopt_ACQ_60M
arrays['BayesOpt 1M acq'] = bayesopt_ACQ_1M
arrays['BayesOpt 1B recX'] = bayesopt_Xs_1B
arrays['BayesOpt 60M recX'] = bayesopt_Xs_60M
arrays['BayesOpt 1M recX'] = bayesopt_Xs_1M

filename = f'saved_logs/thepile_{num_runs}_{domain_name}_{NUM_INIT}.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    for k, vs in arrays.items():
        write_list = [k]
        for v in vs:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            write_list.append(v)
        writer.writerow(write_list)









