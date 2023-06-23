# rl-chess-nova-ims

Welcome! This is a repository for Reinforcement Learning project ! 

| Student Number | Student Names |
|---|---|
| 20220595 | Joice Cardoso | 
| 20220593 | Jaime Kuei | 
| 20220594 | Maxwell Marcos | 
| 20210581 | Miguel Ramos |

## Contents
This repository is organanized as follows:
```
learn.py - this file is responsible for run the learning process
agents.py - this file is responsible for creating the agent classes  
utils.py - this file is responsible for creating helper functions for the project
```
## How to configure the local environment for the project?

 ```
python3 -m venv .reinforcement-learning-env
source .reinforcement-learning-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
 ```

## How to configure the configuration parameters for the differents learnings?

It's possible to configure the configuration parameters in the file called `rl_agent_config.yaml` that contains the configuration information for the different learning algorithms that will be used to train an agent to play chess.

See the following example of the configuration parameters for an agent that plays agains stockfish and uses the white pieces:

```
q_learning_white_stockfish:
  ENV_NAME: "ChessAlphaZero-v0"
  NUM_EPISODES: 100000
  DISCOUNT_FACTOR: 1
  ALPHA: 0.6
  EPSILON: 0.1
  STRATEGY: 'q_learning_white_stockfish'
  CHECKPOINT_METRICS: 10
  CHECKPOINT_ARTEFACTS: 50
  COLOR_PLAYER: 'white'
  TYPE: 'stockfish'
```
## How to set with stockfish?
For `macos` only need to pass `macos` when run the learning process
For `ubuntu` and `windows` users you need to past the stockfish path in the root folder of the 
project and pass the `--stock-fish-path` or `-sfpath` when run. 
Ex.: `--stock-fish-path 'stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2'`

## What are the possible arguments?

```
`--file or -f`: configuration file name.
`--config-strategy or -c`: used to choose the configured strategy within the configuration file.
`--version or -v`: used to select the learn version.
`--so-type or -so`: used to choose the user's operating system between 'ubuntu', 'macos' and 'windows'.
`--checkpoint or -ch`: when set to true, retrieves the last checkpoint made for a specific configuration.
`--stock-fish-path or -sfp`: used when the OS is Windows or Ubuntu, specifies the path to the Stockfish folder.
```

## How to run locally the learning process for the agents?

To inicialize the learning process using DQN for an agent that plays white pieces:
```
python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy dqn_white_stockfish_v2 --version v1
```

To inicialize the learning process using Q-learning for an agent that plays white pieces:
```
python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy q_learning_white_stockfish --version v1
```