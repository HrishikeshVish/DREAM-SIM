import os
import warnings
import os.path as osp
import torch
from planner.data import AV2DataModule
from planner.data.dataclass import Scenario, ObjectProperty, Trajectory, MapPoint
from planner.model import SeNeVAMLightningModule
from planner.data.viz import plot_scenario
import matplotlib.pyplot as plt
import math
from typing import Any, Literal, Optional
import numpy as np
import numpy.typing as npt

# import math

from dataclasses import dataclass, field
import numpy as np



# Set the environment variable to suppress the tensorflow warnings
def load_way_points(ax,data,ds):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Suppress the warnings
    warnings.filterwarnings("ignore")
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
        # torch.device("cpu")
    )
    print(device)
    ckpt_dir = os.path.abspath(
        "/home/prashanth/planner/src/planner/checkpoints"
    )

    model = SeNeVAMLightningModule.load_from_checkpoint(
        os.path.join(ckpt_dir, "epoch_032_b2ed65b.ckpt"), map_location=device, strict=True
    )


    assert isinstance(data, Scenario)
    data = data.to(device=device)

    with torch.no_grad():
        current, curr_valid = model.get_current(scenario=data, include_sdc=False)
        current_no_sdc, curr_valid_no_sdc = model.get_current(scenario=data, include_sdc=True)
        target_no_sdc, tar_val_no_sdc = model.get_target(scenario=data, include_sdc=True)
        target, tar_valid = model.get_target(scenario=data, include_sdc=False)
        output = model.forward(scenario=data, horizon=60, include_sdc=False)
    print(current.shape, curr_valid.shape)
    print(target.shape, tar_valid.shape)
    print(output.y_means.shape, output.y_covars.shape)

    ego_idx = None
    for i, traj in enumerate(target_no_sdc[0]):
        is_ego = True
        for j, traj_other in enumerate(target[0]):
            if torch.allclose(traj, traj_other, equal_nan=True):
                is_ego = False
                break
        if is_ego:
            ego_idx = i
            break

    BATCH_INDEX = 0
    scenario_id = data[BATCH_INDEX].scenario_id.decode("utf-8")
    map_api = ds.get_map_api(scenario_id=scenario_id)
    # plot the ground-truth observations
    cum_target = target_no_sdc.cumsum(dim=-2) + current_no_sdc
    end_x = cum_target[0][ego_idx][-1][0].cpu().numpy()
    end_y = cum_target[0][ego_idx][-1][1].cpu().numpy()
    ego_traj = cum_target[0][ego_idx]  # shape: [time, 5]
    ego_points = ego_traj[:, :2].cpu().numpy().tolist()  # Extract x and y

    # Optional: convert to list of (x, y) tuples
    ego_points = [tuple(pt) for pt in ego_points]
    print(ego_points)

    # plot the predictions
    cum_y_means = output.y_means.cumsum(dim=-2) + current.unsqueeze(-3)
    cum_y_covars = output.y_covars.cumsum(dim=-3)
    paths = []
    for y_means, val in zip(cum_y_means[BATCH_INDEX], tar_valid[BATCH_INDEX]):
        for y_mean in y_means:
            xy = y_mean[val][..., 0:2].cpu().numpy()
            l = list(zip(xy[..., 0], xy[..., 1]))
            paths.extend(l)
    
    uncertainty_data_all_agents = []
    for i in range(0,output.y_means.shape[1]):
        probs, points, grid_shape = plot_full_uncertainty(
            means=cum_y_means[BATCH_INDEX, i, ..., 0:2].cpu().numpy(),
            covars=cum_y_covars[BATCH_INDEX, i, ..., 0:2, 0:2].cpu().numpy(),
            mixtures=np.ones(output.y_means.shape[1]) / output.y_means.shape[1],
            ax=ax,
            n_std=1,
            heatmap_type="log_prob",
            alpha=0.75,
            zorder=15,
        )
        uncertainty_data_all_agents.append((probs, points, grid_shape))

    return uncertainty_data_all_agents, paths, (end_x, end_y), ego_points

def plot_full_uncertainty(
    means: npt.NDArray,
    covars: npt.NDArray,
    mixtures: npt.NDArray,
    ax: Optional[plt.Axes] = None,
    n_std: int = 1,
    heatmap_type: Literal["prob", "log_prob", "logit"] = "log_prob",
    colorbar: bool = True,
    *args: Any,
    **kwargs: Any,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # step 1: obtain the rectangle region covering at least 68%
    # of the probability density function
    minx = np.min(means[:, :, 0] - n_std * np.sqrt(covars[..., 0, 0]))
    maxx = np.max(means[:, :, 0] + n_std * np.sqrt(covars[..., 0, 0]))
    miny = np.min(means[:, :, 1] - n_std * np.sqrt(covars[..., 1, 1]))
    maxy = np.max(means[:, :, 1] + n_std * np.sqrt(covars[..., 1, 1]))

    probs, mask = [], []

    # NOTE: efficient way to evaluate prob
    x, y = np.linspace(minx, maxx, 100), np.linspace(miny, maxy, 100)
    grid_shape = (len(y), len(x))
    x, y = np.meshgrid(x, y)
    points = np.vstack((x.ravel(), y.ravel())).T  # shape (10000, 2)

    for mean, covar, pi in zip(means, covars, mixtures):
        # calculate the Gaussian probability density function
        inv_covar = np.linalg.inv(covar)  # shape (T, 2, 2)
        det_covar = np.linalg.det(covar)  # shape (T,)
        diff = points[:, None, ...] - mean[None, ...]  # shape (10000, T, 2)
        mahalanobis = np.einsum("btj, tij, bti -> bt", diff, inv_covar, diff)
        log_prob = -0.5 * (
            mahalanobis + np.log(det_covar + 1e-10) + 2 * np.log(2 * np.pi)
        )
        log_prob += np.log(pi + 1e-10)
        probs.append(np.exp(log_prob))
        mask.append(
            np.reshape(
                np.all(mahalanobis > n_std * math.sqrt(2), -1),
                (100, 100),
            ).astype(bool)
        )
    mask = np.all(mask, axis=0)

    probs = np.stack(probs, axis=0)
    probs = np.mean(np.sum(probs, axis=0), axis=-1)
    probs = np.reshape(probs, (100, 100))
    probs = np.ma.masked_where(mask, probs)
    if heatmap_type == "log_prob":
        probs = np.log(probs + 1e-10)
        heatmap_name = "Log-Probability"
    elif heatmap_type == "logit":
        probs = np.log((probs + 1e-10) / (1 - probs - 1e-10))
        heatmap_name = "Logit"
    else:
        heatmap_name = "Probability"
    return probs, points, grid_shape