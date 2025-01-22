from typing import Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import PDMEmergencyBrake


class PDMEmergencyBrakeSafeDriver(PDMEmergencyBrake):
    """Class for emergency brake maneuver of PDM-Closed."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        time_to_infraction_threshold: float = 2.0,
        max_ego_speed: float = 5.0,
        max_long_accel: float = 2.40,
        min_long_accel: float = -4.05,
        infraction: str = "collision",
    ):
        """
        Constructor for PDMEmergencyBrakeSafeDriver
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param time_to_infraction_threshold: threshold for applying brake, defaults to 2.0
        :param max_ego_speed: maximum speed to apply brake, defaults to 5.0
        :param max_long_accel: maximum longitudinal acceleration for braking, defaults to 2.40
        :param min_long_accel: min longitudinal acceleration for braking, defaults to -4.05
        :param infraction: infraction to determine braking (collision or ttc), defaults to "collision"
        """
        super().__init__(
            trajectory_sampling,
            time_to_infraction_threshold,
            max_ego_speed,
            max_long_accel,
            min_long_accel,
            infraction,
        )

        self._time_to_infraction_current = np.inf
        self._time_to_infraction_list = []

    def brake_if_emergency(
        self, ego_state: EgoState, scores: npt.NDArray[np.float64], scorer: PDMScorer
    ) -> Optional[InterpolatedTrajectory]:
        """
        Applies emergency brake only if an infraction is expected within horizon.
        :param ego_state: state object of ego
        :param scores: array of proposal scores
        :param metric: scorer class of PDM
        :return: brake trajectory or None
        """

        trajectory = None
        ego_speed: float = ego_state.dynamic_car_state.speed

        proposal_idx = np.argmax(scores)

        # retrieve time to infraction depending on brake detection mode
        if self._infraction == "ttc":
            time_to_infraction = scorer.time_to_ttc_infraction(proposal_idx)

        elif self._infraction == "collision":
            time_to_infraction = scorer.time_to_at_fault_collision(proposal_idx)

        # check time to infraction below threshold
        if (
            time_to_infraction <= self._time_to_infraction_threshold
            and ego_speed <= self._max_ego_speed
        ):
            trajectory = self._generate_trajectory(ego_state)

        # update time to infraction
        self._time_to_infraction_current = time_to_infraction
        self._time_to_infraction_list.append(time_to_infraction)

        return trajectory