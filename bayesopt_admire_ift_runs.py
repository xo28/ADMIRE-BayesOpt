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
from load_data import prepare_admire_ift_runs, add_fidelity
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

def initialize_model(train_x, train_obj, X_test_high, y_test_high, idx=19):
    model = SingleTaskMultiFidelityGP(train_x, train_obj, data_fidelity=idx)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    mse = evaluate(model, X_test_high, y_test_high)
    return mse, model

# Used for the GP kernel only; not used in acquisition or recommendation. Always the same value for data with the same fidelity.
fidelity_weights = {'low':95.3487726190051/537.467857142887, 'medium':258.8735687499986/537.467857142887, 'high':1.0}
ratio_05B, metric_05B, ratio_3B, metric_3B, ratio_7B, metric_7B, domain_name, threshold = prepare_admire_ift_runs(args.idx, tkwargs)
X_train_with_fidelity, X_train_low, X_train_medium, X_train_high, Y_train = add_fidelity(fidelity_weights, ratio_05B, ratio_3B, ratio_7B, metric_05B, metric_3B, metric_7B, tkwargs)
X_test_high = X_train_high[threshold:]
y_test_high = metric_7B[threshold:]
X_train_high = X_train_with_fidelity[:-(X_train_high.shape[0]-threshold)]
bounds = torch.tensor([[0.0] * X_train_with_fidelity.shape[1], [1.0] * X_train_with_fidelity.shape[1]], **tkwargs)
target_fidelities = {19: 1.0}
cost_model = AffineFidelityCostModel(fidelity_weights={19: 1.0}, fixed_cost=0.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
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

    values = rec_acqf(ratio_7B[:threshold].unsqueeze(1)) # recommend without fidelity
    rec_id = torch.argmax(values).item()
    return metric_7B[rec_id].item(), ratio_7B[rec_id]

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
        print(f"mse: {mse:.4f} | recmd: {np.array(rec_ys).max():.4f}")
        print(f"cumulative cost: {cumulative_cost[-1]:.4f}")

        if i == num_iter - 1:
            mse, model = initialize_model(train_x, train_obj, X_test_high, y_test_high)
            mses.append(mse)
            # Make a final recommendation
            rec_y, rec_x = get_recommendation(model)
            rec_ys.append(rec_y)
            rec_xs.append(rec_x.tolist())
            print(f"mse: {mse:.4f} | recmd: {np.array(rec_ys).max():.4f}")
            print(f"cumulative cost: {cumulative_cost[-1]:.4f}")

    return mses, rec_ys, cumulative_cost, [train_x.tolist(), train_obj.tolist()], rec_xs


##################################### average runs ###################################################
bayesopt_MSEs_7B = []
bayesopt_MSEs_3B = []
bayesopt_MSEs_05B = []
bayesopt_Ys_7B = []
bayesopt_Ys_3B = []
bayesopt_Ys_05B = []
bayesopt_ACQ_7B = []
bayesopt_ACQ_3B = []
bayesopt_ACQ_05B = []
bayesopt_Xs_7B = []
bayesopt_Xs_3B = []
bayesopt_Xs_05B = []

num_runs = 5
for i in tqdm(range(num_runs)):
    seed = torch.randint(0, 100, (1,)).item()
    seed_everything(seed)
    bayesopt_mses_7B, bayesopt_ys_7B, bayesopt_cost_7B, bayesopt_acq_7B, bayesopt_xs_7B = run_standardbayesopt(X_train_high[:threshold], metric_7B[:threshold], fidelity_weights['high'], threshold-NUM_INIT) 
    bayesopt_mses_3B, bayesopt_ys_3B, bayesopt_cost_3B, bayesopt_acq_3B, bayesopt_xs_3B = run_standardbayesopt(X_train_medium, metric_3B, fidelity_weights['medium'], 100-NUM_INIT)
    bayesopt_mses_05B, bayesopt_ys_05B, bayesopt_cost_05B, bayesopt_acq_05B, bayesopt_xs_05B = run_standardbayesopt(X_train_low, metric_05B, fidelity_weights['low'], 100-NUM_INIT)
    bayesopt_MSEs_7B.append(np.array(bayesopt_mses_7B))
    bayesopt_MSEs_3B.append(np.array(bayesopt_mses_3B))
    bayesopt_MSEs_05B.append(np.array(bayesopt_mses_05B))
    bayesopt_Ys_7B.append(np.array(bayesopt_ys_7B)) 
    bayesopt_Ys_3B.append(np.array(bayesopt_ys_3B))
    bayesopt_Ys_05B.append(np.array(bayesopt_ys_05B))
    bayesopt_ACQ_7B.append(bayesopt_acq_7B)
    bayesopt_ACQ_3B.append(bayesopt_acq_3B)
    bayesopt_ACQ_05B.append(bayesopt_acq_05B)
    bayesopt_Xs_7B.append(bayesopt_xs_7B)
    bayesopt_Xs_3B.append(bayesopt_xs_3B)
    bayesopt_Xs_05B.append(bayesopt_xs_05B)


bayesopt_mses_7B = np.mean(bayesopt_MSEs_7B, axis=0)
bayesopt_mses_3B = np.mean(bayesopt_MSEs_3B, axis=0)
bayesopt_mses_05B = np.mean(bayesopt_MSEs_05B, axis=0)

##################################### save ###################################################
arrays = {
    'BayesOpt 7B': [bayesopt_cost_7B, np.array(bayesopt_Ys_7B)],
    'BayesOpt 3B': [bayesopt_cost_3B, np.array(bayesopt_Ys_3B)],
    'BayesOpt 0.5B': [bayesopt_cost_05B, np.array(bayesopt_Ys_05B)],
}
arrays['BayesOpt 7B MSE'] = [list(range(len(bayesopt_mses_7B))), np.array(bayesopt_MSEs_7B)]
arrays['BayesOpt 3B MSE'] = [list(range(len(bayesopt_mses_3B))), np.array(bayesopt_MSEs_3B)]
arrays['BayesOpt 0.5B MSE'] = [list(range(len(bayesopt_mses_05B))), np.array(bayesopt_MSEs_05B)]
arrays['BayesOpt 7B acq'] = bayesopt_ACQ_7B
arrays['BayesOpt 3B acq'] = bayesopt_ACQ_3B
arrays['BayesOpt 0.5B acq'] = bayesopt_ACQ_05B
arrays['BayesOpt 7B recX'] = bayesopt_Xs_7B
arrays['BayesOpt 3B recX'] = bayesopt_Xs_3B
arrays['BayesOpt 0.5B recX'] = bayesopt_Xs_05B

filename = f'saved_logs/admire_ift_runs-{timestamp}_{num_runs}_{domain_name}_{NUM_INIT}.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    for k, vs in arrays.items():
        write_list = [k]
        for v in vs:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            write_list.append(v)
        writer.writerow(write_list)







