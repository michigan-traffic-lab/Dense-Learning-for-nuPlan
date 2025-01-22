import gc
import logging
import warnings
from typing import List, Optional, Type, cast
import time
import os

import torch
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.serialization.scene import Trajectory

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_closed_planner import (
    AbstractPDMClosedPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_feature_utils import (
    create_pdm_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from safedriver_nuplan.planner.utils.pdm_report import (
    PDMPlannerReport,
)
from safedriver_nuplan.planner.planTF_planner_utils import load_checkpoint
from safedriver_nuplan.planner.models.planTF.feature_builders.common.utils import rotate_round_z_axis
import safedriver_nuplan.planner.models.safedriver as safe_driver
from safedriver_nuplan.planner.utils.pdm_emergency_brake_safedriver import PDMEmergencyBrakeSafeDriver

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class ObservationSpace:
    def __init__(self):
        self.iteration = None
        self.emergency_flag = None
        self.pdm_feature = None
        self.pdm_encoded_features = None
        self.safedriver_obs = None
        self.uncorrected_states = None
        self.planner_input = None

    def add_planner_input(self, planner_input):
        self.planner_input = planner_input
    
    def update(self, planner):
        self.iteration = planner._iteration
        # print(planner._iteration, planner.simulations_runner.simulation.setup.time_controller.current_iteration_index)

        planner_input = self.planner_input

        self.emergency_flag, self.pdm_feature, self.pdm_encoded_features, self.safedriver_obs, self.uncorrected_states = planner.get_observation(planner_input)


class PDMHybridWithSafeDriverPlannerInference(AbstractPDMClosedPlanner):
    """PDM-Hybrid with SafeDriver planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        model_planner: TorchModuleWrapper,
        model_safedriver_encoder: TorchModuleWrapper,
        correction_horizon: float,
        checkpoint_path_pdm_offset: str,
        checkpoint_path_safedriver_encoder: str,
        offset_flag: bool,
    ):
        """
        Constructor for PDM-Hybrid with SafeDriver.
        :param trajectory_sampling: sampling parameters for final trajectory
        :param proposal_sampling: sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        :param model: torch model
        :param correction_horizon: time to apply open-loop correction [s]
        :param checkpoint_path_pdm_offset: path to checkpoint for PDM-Offset model as string
        :param checkpoint_path_safedriver_encoder: path to checkpoint for SafeDriver encoder as string
        :param offset_flag: flag to apply offset correction
        """

        super(PDMHybridWithSafeDriverPlannerInference, self).__init__(
            trajectory_sampling,
            proposal_sampling,
            idm_policies,
            lateral_offsets,
            map_radius,
        )
        # overwrite the default emergency brake
        self._emergency_brake = PDMEmergencyBrakeSafeDriver(trajectory_sampling)

        self._device = "cpu"

        self._model_planner = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path_pdm_offset,
            model=model_planner,
            map_location=self._device,
        ).model
        self._model_planner.eval()

        torch.set_grad_enabled(False)

        self._correction_horizon: float = correction_horizon  # [s]

        self.simulations_runner = None

        self._safedriver_encoder = model_safedriver_encoder
        self._safedriver_encoder_feature_builder = model_safedriver_encoder.get_list_of_required_feature()[0]
        self._safedriver_encoder_ckpt = checkpoint_path_safedriver_encoder
        self._safedriver_encoder_initialization: Optional[PlannerInitialization] = None
        self._safedriver_encoder.load_state_dict(load_checkpoint(self._safedriver_encoder_ckpt))
        self._safedriver_encoder.eval()
        self._safedriver_encoder = self._safedriver_encoder.to("cpu")

        self._future_horizon = 8.0
        self._step_interval = 0.1
        replan_interval = 1
        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self.offset_flag = offset_flag
        self.emergency_stats = []

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        self._observation_space = ObservationSpace()
        gc.collect()

        self._safedriver_encoder_initialization = initialization

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

        self.emergency_stats = []

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def get_observation(self, current_input: PlannerInput):
        """
        Get observation for PDM-Hybrid with SafeDriver.
        :param current_input: planner input
        :return: observation
        """
        emergency_flag, pdm_feature, pdm_encoded_features, _, uncorrected_states = self.get_observation_PDMHybrid(current_input)
        safedriver_obs = self.get_observation_safedriver_encoder(current_input)
        return (
            emergency_flag,
            pdm_feature,
            pdm_encoded_features,
            safedriver_obs,
            uncorrected_states,
        )

    def get_observation_PDMHybrid(self, current_input: PlannerInput):
        """
        Get observation for PDM-Hybrid.
        :param current_input: planner input
        :return: observation
        """
        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        # trajectory of PDM-Closed
        closed_loop_trajectory, emergency_flag = self._get_closed_loop_trajectory(
            current_input
        )
        uncorrected_states = closed_loop_trajectory.get_sampled_trajectory()

        # trajectory of PDM-Offset
        pdm_feature = create_pdm_feature(
            self._model_planner,
            current_input,
            self._centerline,
            closed_loop_trajectory,
            self._device,
        )
        pdm_encoded_features = self._model_planner.encoding({"pdm_features": pdm_feature})
        safedriver_obs = pdm_encoded_features.squeeze().detach().numpy()
        return (
            emergency_flag,
            pdm_feature,
            pdm_encoded_features,
            safedriver_obs,
            uncorrected_states,
        )

    def get_observation_safedriver_encoder(self, current_input: PlannerInput):
        """
        Get observation for SafeDriver encoder.
        :param current_input: planner input
        :return: observation
        """
        obs = np.zeros(128*4)
        planner_feature = self._safedriver_encoder_feature_builder.get_features_from_simulation(
            current_input, self._safedriver_encoder_initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor().to_device("cpu")]
        )
        obs_safedriver_encoder = self._safedriver_encoder.get_encoded_features(planner_feature_torch.data)
        obs[:obs_safedriver_encoder.shape[1]] = obs_safedriver_encoder.detach().cpu().numpy().reshape(-1)
        return obs

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        """
        Get global trajectory for SafeDriver encoder.
        :param local_trajectory: local trajectory
        :param ego_state: ego state
        :return: global trajectory
        """
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory

    def compute_planner_trajectory(
        self,
        current_input: PlannerInput,
        safedriver_action: Optional[torch.Tensor] = None,
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""
        self._observation_space.add_planner_input(current_input)
        pdm_feature = self.observation_space.pdm_feature
        emergency_flag = self.observation_space.emergency_flag
        uncorrected_states = self.observation_space.uncorrected_states
        pdm_encoded_features = self.observation_space.pdm_encoded_features
        safedriver_obs = self.observation_space.safedriver_obs

        self.emergency_stats.append(emergency_flag)

        if emergency_flag:
            o = torch.reshape(torch.tensor(safedriver_obs), (1,len(safedriver_obs)))
            out = safe_driver.safedriver_module({"obs":o},[torch.tensor([0.0])],torch.tensor([1]))
            safedriver_action = out[0][0,:48].reshape(1,-1).detach().cpu().numpy()
            # safedriver_action = safe_driver.safedriver_module.compute_single_action(safedriver_obs).reshape(1,-1)

        predictions = self._model_planner.forward(
            {
                "pdm_features": pdm_feature,
                "pdm_encoded_features": pdm_encoded_features,
                "safedriver_action": safedriver_action,
                "emergency_flag": emergency_flag,
            },
            offset_flag = self.offset_flag,
        )
        # assert
        trajectory_data = (
            cast(Trajectory, predictions["trajectory"]).data.cpu().detach().numpy()[0]
        )
        corrected_states = transform_predictions_to_states(
            trajectory_data,
            current_input.history.ego_states,
            self._model_planner.trajectory_sampling.time_horizon,
            self._model_planner.trajectory_sampling.step_time,
        )

        # apply correction by fusing
        trajectory = self._apply_trajectory_correction(
            uncorrected_states, corrected_states, emergency_flag
        )
        self._iteration += 1
        return trajectory

    def compute_trajectory(
        self, current_input: PlannerInput, safedriver_action=None
    ) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory, where we check that if planner can not consume batched inputs,
            we require that the input list has exactly one element
        :param current_input: List of planner inputs for where for each of them trajectory should be computed
            In this case the list represents batched simulations. In case consume_batched_inputs is False
            the list has only single element
        :param safedriver_action: SafeDriver action to be applied
        :return: Trajectories representing the predicted ego's position in future for every input iteration
            In case consume_batched_inputs is False, return only a single trajectory in a list.
        """
        start_time = time.perf_counter()
        # If it raises an exception, catch to record the time then re-raise it.
        try:
            trajectory = self.compute_planner_trajectory(
                current_input, safedriver_action
            )
        except Exception as e:
            self._compute_trajectory_runtimes.append(time.perf_counter() - start_time)
            raise e

        self._compute_trajectory_runtimes.append(time.perf_counter() - start_time)
        return trajectory

    def _apply_trajectory_correction(
        self,
        uncorrected_states: List[EgoState],
        corrected_states: List[EgoState],
        emergency_flag: bool,
    ) -> InterpolatedTrajectory:
        """
        Applies open-loop correction and fuses to a single trajectory.
        :param uncorrected_states: ego vehicles states of PDM-Closed trajectory
        :param corrected_states: ego-vehicles states of PDM-Offset with SafeDriver trajectory
        :param emergency_flag: flag to apply SafeDriver action
        :return: trajectory after applying correction.
        """

        # split trajectory
        if emergency_flag:
            uncorrected_duration: TimeDuration = TimeDuration.from_s(0.0)
            # uncorrected_duration: TimeDuration = TimeDuration.from_s(
            #     self._correction_horizon
            # )
        else:
            uncorrected_duration: TimeDuration = TimeDuration.from_s(
                self._correction_horizon
            )
        cutting_time_point: TimePoint = (
            uncorrected_states[0].time_point + uncorrected_duration
        )

        uncorrected_split = [
            ego_state
            for ego_state in uncorrected_states
            if ego_state.time_point <= cutting_time_point
        ]

        corrected_split = [
            ego_state
            for ego_state in corrected_states
            if ego_state.time_point > cutting_time_point
        ]

        return InterpolatedTrajectory(uncorrected_split + corrected_split)

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """

        ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )
        emergency_flag = (
            self._emergency_brake._time_to_infraction_current
            <= self._emergency_brake._time_to_infraction_threshold
        )

        # 6.b Otherwise, extend and output best proposal
        if trajectory is None:
            trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))

        return trajectory, emergency_flag

    @property
    def observation_space(self):
        if self._observation_space.iteration != self._iteration:
            self._observation_space.update(self)
        return self._observation_space
    
    def generate_planner_report(self, clear_stats: bool = True):
        """
        Generate planner report.
        :param clear_stats: flag to clear statistics
        :return: planner report
        """
        num_collisions = 0
        report = PDMPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            time_to_infraction_list=self._emergency_brake._time_to_infraction_list,
            num_collisions=num_collisions,
        )

        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._emergency_brake._time_to_infraction_list = []
            if hasattr(self, "emergency_stats"):
                self.emergency_stats = []
 
        return report