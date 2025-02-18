from typing import Deque

import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)


def global_trajectory_to_states(
    global_trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = True,
):
    """
    Convert global trajectory to states
    :param global_trajectory: global trajectory
    :param ego_history: history of ego vehicle
    :param future_horizon: future horizon
    :param step_interval: step interval
    :param include_ego_state: whether to include ego state in the output
    :return: list of states
    """
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    global_states = [StateSE2.deserialize(pose) for pose in global_trajectory]

    velocities, accelerations = _get_velocity_and_acceleration(
        global_states, ego_history, timesteps
    )
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            global_states, velocities, accelerations, timesteps
        )
    ]

    if include_ego_state:
        agent_states.insert(0, ego_state)

    return agent_states


def load_checkpoint(checkpoint: str):
    """
    Load checkpoint
    :param checkpoint: checkpoint path
    :return: state dictionary
    """
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    return state_dict
