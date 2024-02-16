# AAMAS Submission - 995: Mixed-Initiative Human-Robot Teaming under Suboptimality with Online Bayesian Adaptation




---

### Dependencies:
- numpy (version 1.21.2)
- matplotlib (version 3.6.2)
- gym (version 0.26.2)
- pygame (version 2.5.0)
- scipy (version 1.9.0)
- R packages required for statistical analysis (can be installed inside a conda env.)
- Tested on Python version (3.9.16)

---
### File Structure:
1. frozen_lake:
   1. data_analysis: Contains R code (used in analyzing data from human-subjects experiments)

      (Note that the csv files of the user data are not included).
   2. img: Images used in rendering the Frozen Lake Domain.
   3. plots: Files used to generate plots for the paper (csv files not included).
   4. ``frozen_lake_env.py``: Implementation of the Frozen Lake domain (Adapted from [Open AI Gym](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)).
   5. ``frozen_lake_interface.py``: Pygame interface for Frozen Lake.
   6. ``frozen_lake_map.py``: Maps used in the user study and for simulation experiments.
   7. ``run_user_study.py``: Main script used for our human-subject experiments.
   

2. pomcp_solvers: Class definitions for running the POMCP solver variants
   1. ``dynamic_users.py``: Simulated human model where users latent states change over time.
   2. ``human_action_node.py``: Defines the human action node used in the POMCP search.
   3. ``robot_action_node.py``: Defines the robot action node used in the POMCP search.
   4. ``root_node.py``: Defines the root node used in the POMCP search.
   5. ``simulated_human.py``: Defines the static human model.
   6. ``solver.py``: Defines the POMCP search routine.


3. simulated_human_experiments:
   1. ``ba_pomcp.py``: Main script to run bayes-pomcp. If the parameter ``update_belief=False`` in the main function, then it runs the regular POMCP.
   2. ``evaluate_heuristics.py``: Runs the heuristic policies with different simulated human models (``HeuristicAgent=1`` indicates the interrupt agent, and ``HeuristicAgent=2`` indicates the take-control agent).
   3. ``utils.py``: Miscellaneous functions.


4. ``visualize_rollouts.py``: Utility file to visualize rollouts from human interaction data (from the user study), or from simulated human models.
---
### Main Scripts:
1. Running the interface: To try out our frozen lake interface, please run the following from the ``frozen_lake_baselines`` folder:

   
   ``python frozen_lake/run_user_study.py``
2. To run the simulation experiments, you need to set appropriate latent parameters for the human models in the ``simulated_human_experiments/ba_pomcp.py``:
   1. For static users with varying expertise (in the paper), we set the ``epsilon_vals = [0.2, 0.4, 0.6, 0.8]`` and ``true_trust=[(75,25)]`` (the beta parameter).
   2. For static users with varying compliance (in the paper), we set the ``true_trust=[(20,80), (40,60), (60,40), (80,20)]``, and ``epsilon_vals=0.5``.
   3. We average our results across three seeds (0,2,4) on five maps (4,5,7,10,12).
   4. To run the same for POMCP, set ``update_belief=False`` in the main function in ``simulated_human_experiments/ba_pomcp.py``
   5. To run the same with heuristics, use ``simulated_human_experiments/evaluate_heuristics.py`` with the same parameters.

---
### Codebase adapted from:


Lee, Joshua, et al. "Getting to know one another: Calibrating intent, capabilities and trust for human-robot collaboration." 
2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.
[[TICC-MCP Github Repo](https://github.com/clear-nus/TICC-MCP)]

---

