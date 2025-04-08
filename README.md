# HERACLES



## Abstract

Distributed sensor fusion is a key component of a broad spectrum of applications, such as autonomous systems, where the ability of jointly process multi-sensor data at the edge boosts the range of operating conditions and overall task performance. However, existing distributed sensor fusion approaches encounter limitations in achieving efficient transmission and computation, primarily due to sensor data redundancy, unreliable sensor data transmission, and inflexible sensor fusion methods. In this paper, we propose HERACLES, a distributed sensor fusion framework that connects multi-branched dynamic neural network architectures, which we extend to include branches of different complexity, to (i) a computing methodology that distributes portions of the multi-branched neural network across mobile devices and edge servers, enabling flexible semantic feature extraction and sensor fusion; (ii) a hierarchical modulation-based transmission strategy, where multi-modal semantic features are allocated to different modulation layers to provide varying levels of error protection, and (iii) an infrastructure-level logic that controls the matching between semantic features and modulation layers, and the complexity of the neural model itself to meet an accuracy target while minimizing latency and energy consumption. As a result, HERACLES deeply connects computing, communications and resource allocations in a semantic and context-aware fashion. We evaluate HERACLES using real-world datasets and demonstrate that it can reduce the total delay and energy consumption by 20.39%--$89.41% and 4.86%--88.17% (resp.), while maintaining near-optimal inference accuracy.



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


