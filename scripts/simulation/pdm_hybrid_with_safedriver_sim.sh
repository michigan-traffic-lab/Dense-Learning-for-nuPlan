export NUPLAN_DATA_ROOT=/path/to/nuPlan/dataset # define the path to the nuplan dataset
export NUPLAN_EXP_ROOT=/path/to/experiment # define the path to the nuplan experiments
export NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit # define the path to the nuplan devkit
export DENSE_LEARNING_ROOT=/path/to/Dense-Learning-for-nuPlan # define the path to the Dense-Learning-for-nuPlan
export SAFEDRIVER_ACTIONHEAD_PATH=/path/to/Dense-Learning-for-nuPlan/scripts/simulation/agents/safedriver_actionhead.pt # define the path to the safedriver actionhead
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=closed_loop_reactive_agents \
planner=pdm_hybrid_with_safedriver_eval \
planner.pdm_hybrid_with_safedriver_eval.checkpoint_path_pdm_offset=$DENSE_LEARNING_ROOT/scripts/simulation/agents/pdm_offset_checkpoint.ckpt \
planner.pdm_hybrid_with_safedriver_eval.checkpoint_path_safedriver_encoder=$DENSE_LEARNING_ROOT/scripts/simulation/agents/safedriver_encoder.ckpt \
scenario_filter=val14_split \
scenario_builder=nuplan \
hydra.searchpath="[pkg://safedriver_nuplan.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"