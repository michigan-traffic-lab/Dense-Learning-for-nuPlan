import torch
import torch.nn as nn
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    SE2Index,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pdm_feature_builder import (
    PDMFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)
from tuplan_garage.planning.training.modeling.models.pdm_offset_model import (
    PDMOffsetModel,
)


class PDMOffsetModelforSafeDriver(PDMOffsetModel):
    def encoding(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "pdm_features": PDFeature,
                        }
        :return: planner_features: encoded features for planner
        """

        input: PDMFeature = features["pdm_features"]

        batch_size = input.ego_position.shape[0]

        ego_position = input.ego_position.reshape(batch_size, -1).float()
        ego_velocity = input.ego_velocity.reshape(batch_size, -1).float()
        ego_acceleration = input.ego_acceleration.reshape(batch_size, -1).float()

        # encode ego history states
        state_features = torch.cat(
            [ego_position, ego_velocity, ego_acceleration], dim=-1
        )
        state_encodings = self.state_encoding(state_features)

        state_features = torch.cat(
            [ego_position, ego_velocity, ego_acceleration], dim=-1
        )
        state_encodings = self.state_encoding(state_features)

        # encode PDM-Closed trajectory
        planner_trajectory = input.planner_trajectory.reshape(batch_size, -1).float()
        trajectory_encodings = self.trajectory_encoding(planner_trajectory)

        # encode planner centerline
        planner_centerline = input.planner_centerline.reshape(batch_size, -1).float()
        centerline_encodings = self.centerline_encoding(planner_centerline)

        # decode future trajectory
        planner_features = torch.cat(
            [state_encodings, centerline_encodings, trajectory_encodings], dim=-1
        )
        return planner_features

    def forward(self, features: FeaturesType, offset_flag=True) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "pdm_features": PDFeature,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """

        input: PDMFeature = features["pdm_features"]

        batch_size = input.ego_position.shape[0]
        planner_trajectory = input.planner_trajectory.reshape(batch_size, -1).float()

        planner_features = features["pdm_encoded_features"]
        planner_output = self.planner_head(planner_features)
        output_trajectory = planner_trajectory + planner_output
        output_trajectory = output_trajectory.detach().cpu()
        if features["emergency_flag"] and features["safedriver_action"] is not None:
            if offset_flag:
                output_trajectory += features["safedriver_action"]
            else:
                output_trajectory = torch.from_numpy(features["safedriver_action"])
        output_trajectory = output_trajectory.reshape(batch_size, -1, len(SE2Index))

        return {"trajectory": Trajectory(data=output_trajectory)}