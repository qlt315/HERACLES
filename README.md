# HierFusion

This work is extended from our previous ICDCS'25 paper"HERACLES:Hierarchical Semantic Communications for Distributed Dynamic Sensor Fusion"

## Abstract

The ability to collaboratively interpret multi-sensor data at the network edge improves the range of operating circumstances and overall task performance in a variety of applications, including autonomous systems, and is considered one of the crucial components of distributed sensor fusion.  However, due to sensor data redundancy, inconsistent sensor data transmission, and rigid sensor fusion techniques, current distributed sensor fusion frameworks struggle to provide effective transmission and computation. In this work, we present \texttt{HierFusion}, an innovative framework that integrates dynamic neural networks with multiple branches of varying computational complexity. The proposed framework adopts a decentralized execution strategy that distributes portions of the neural architecture across mobile devices and edge servers, enabling adaptive semantic feature extraction and fusion. \texttt{HierFusion} supports 5th Generation New Radio (5GNR) using MIMO+OFDM transmission protocols, where  semantic features are dynamically assigned different  modulation and coding schemes (MCS) depending on their relevance to ensure appropriate levels of error protection. A system-level controller orchestrates the mapping between semantic features and transmission modes, and dynamically adjusts model complexity to meet accuracy requirements under latency and energy constraints. Through this design, \texttt{HierFusion} establishes a tightly integrated pipeline that harmonizes computation, communication, and resource coordination in a context-aware and efficient manner. We evaluate \texttt{HierFusion} using real-world autonomous driving datasets, and show a total delay and energy consumption decreased by $34.5\%$--$86.5\%$ and $24.4\%$--$75.9\%$ (resp.) while maintaining near-optimal inference accuracy. 



---

## Directory Structure
- **`baselines/`**: Implements baseline algorithms, such as AMAC and DQN, along with related data files.
- **`envs/`**: Defines custom environments for reinforcement learning simulations, including some utility functions for building corresponding environment.
- **`experiments/`**: Contains scripts and data for running and analyzing various experiments in the evaluation part, such as evaluating the impact of different contexts / reward weights / SNR values, etc.
- **`rainbow_dqn/`**: Houses the implementation of the Rainbow DQN algorithm, including training and evaluation scripts.
- **`system_data/`**: Provides system-related datasets, such as 5G datasets and customized hierarchical / typical modulation results.
- **`tools/`**: Includes auxiliary scripts and utilities for data processing / fitting, model saving / loading / visualization, and analysis.


## Key Files and Scripts

### `baselines/`
- **`dqn/dqn_train.py`**: Script for training the Rainbow DQN agent as a baseline.
- **`dqn/dqn_eval.py`**: Script for evaluating the trained Rainbow DQN agent.
- **`amac.py`**: Implements the AMAC scheme for baseline evaluation.
- **`observations/one_or_two_sensors.py`**: Obtains the observation results for one or two sensor combinations in Sec. I. 
- **`observations/all_sensors_no_stem.py`**: Obtains the observation results for "all sensors" in Sec. I .This script needs to run after the **`observations/one_or_two_sensors.py`**.

### `envs/`
- **`env_proposed_erf.py`**: Defines the HERACLES environment with erf-based reward function.
- **`env_proposed_origin.py`**: Defines the HERACLES environment with linear reward function.
- **`env_sse.py`**: Defines the SSE environment with erf-based reward function.
- **`env_tem.py`**: Defines the TEM environment with erf-based reward function.

### `experiments/`
- **`eval_snr.py`**: Script to evaluate the performance across varying SNR values.
- **`eval_bad_actions.py`**: Script to count the number of bad actions (actions that violate the accuracy).
- **`eval_channel_est_err.py`**: Contains tools for statistical analysis of experiment results.
- **`eval_reward_weights.py`**: Script to evaluate the performance across varying delay / energy / accuracy reward weights (To run this script, you should first run **`train_reward_weight.py`** to obtain the Rainbow agents trained under each specific reward weight setting). 
- **`eval_context.py`**: Script to evaluate the performance across varying task contexts (To run this script, you should first run **`train_context.py`** to obtain the Rainbow agents trained under each specific context). 

### `rainbow_dqn/`
- **`rainbow_train.py`**: Script for training the Rainbow DQN agent. The agent will be trained under a "mix" environment combined all the task contexts. 
- **`rainbow_eval.py`**: Script for evaluating the performance of a trained Rainbow DQN agent. The agent will be evaluated under a "mix" environment combined all the task contexts. 
 

### `system_data/`
- **`hierarchical modulation/hm_main.m`**: Script for obtaining the 2-layer hierarchical modulation BER values under various SNR values.
- **`hierarchical modulation/hm_main_v2.m`**: Script for obtaining up to 4-layer hierarchical modulation BER values under various SNR values.


### `tools/`
- **`data_process.m`**: Script for smoothing DRL reward curves.
- **`modulation fitting.py`**: Script for checking the BER-SNR fitting results.
- **`playgrund.py`**: You can test anything you are not sure here :).
- **`saving_loading.py`**: Script for saving and loading different DRL neural network models.
- **`figure_gen.m`**: Script for generating all the figures in the paper.
---

### Important Notes  

1. This repository does not include the training and evaluation code of the multi-branch dynamic split neural network. It only provides an interface between the neural network evaluation data and the optimization framework. If you have any questions regarding neural network details, please contact Yashuo Wu (yashuow@uci.edu).  

2. When running the Python scripts, please set the working directory to the root of the HERACLES repository, e.g., `D:/your_path_to_this_repo/HERACLES`.


## Contributing
We welcome contributions to improve HERACLES. To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork.
4. Open a pull request detailing the changes.

---


---

## Acknowledgments


The Rainbow DQN scheme is developed based on https://github.com/Lizhi-sjtu/Rainbow-DQN-pytorch

The DQN scheme is developed based on https://github.com/Curt-Park/rainbow-is-all-you-need

The hierarchical / typical modulation scripts with LDPC coding are developed based on https://github.com/ashbhagat/wes_c9_hqam_capstone
and https://github.com/tavildar/LDPC

The 5G wireless dataset is obtained from https://github.com/uccmisl/5Gdataset

Special thanks to these developers for their well-structured open source repositories.


