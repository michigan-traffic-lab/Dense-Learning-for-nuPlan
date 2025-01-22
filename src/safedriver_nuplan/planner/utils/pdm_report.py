from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from dataclasses import fields

from nuplan.planning.simulation.planner.planner_report import PlannerReport


@dataclass(frozen=True)
class PDMPlannerReport(PlannerReport):
    """
    Information about planner runtimes, etc. to store to disk.
    """

    time_to_infraction_list: List[float]  # time series of time to infraction [s]
    num_collisions: int  # number of collisions

    def compute_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics over report fields.
        :return: dictionary containing summary statistics of each field.
        """
        summary = {}
        for field in fields(self):
            attr_value = getattr(self, field.name)
            # Compute summary stats for each field. They are all lists of floats, defined in PlannerReport.
            if field.name == "time_to_infraction_list":
                summary[f"{field.name}_leq_0_0"] = np.sum(np.array(attr_value) <= 0.0)
                summary[f"{field.name}_leq_0_5"] = np.sum(np.array(attr_value) <= 0.5)
                summary[f"{field.name}_leq_1_0"] = np.sum(np.array(attr_value) <= 1.0)
                summary[f"{field.name}_leq_1_5"] = np.sum(np.array(attr_value) <= 1.5)
                summary[f"{field.name}_leq_2_0"] = np.sum(np.array(attr_value) <= 2.0)
                summary[f"{field.name}_leq_2_5"] = np.sum(np.array(attr_value) <= 2.5)
                summary[f"{field.name}_leq_3_0"] = np.sum(np.array(attr_value) <= 3.0)
                summary[f"{field.name}_leq_3_5"] = np.sum(np.array(attr_value) <= 3.5)
                summary[f"{field.name}_leq_4_0"] = np.sum(np.array(attr_value) <= 4.0)
                summary[f"{field.name}_leq_4_5"] = np.sum(np.array(attr_value) <= 4.5)
                summary[f"{field.name}_leq_5_0"] = np.sum(np.array(attr_value) <= 5.0)
            elif field.name in ["num_collisions", "num_ego_at_fault_collisions"]:
                summary[f"{field.name}"] = attr_value
            else:
                summary[f"{field.name}_mean"] = np.mean(attr_value)
                summary[f"{field.name}_median"] = np.median(attr_value)
                summary[f"{field.name}_std"] = np.std(attr_value)

        return summary
