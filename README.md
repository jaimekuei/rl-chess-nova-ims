# rl-chess-nova-ims

Welcome! This is a repository for Reinforcement Learning project ! 

| Student Number | Student Names |
|---|---|
| 20220593 | Joice Preuss | 
| 20220594 | Jaime Kuei | 
| 20220595 | Maxwell Marcos | 
| 20210581 | Miguel Ramos |

## Contents
This repository is organanized as follows:

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

## How to run locally the learning process for the agents?

To inicialize the learning process using Q-Learning for an agent that plays white pieces:
```
python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy q_learning_white_stockfish --version v3
```

To inicialize the learning process using DQN for an agent that plays white pieces:
```
python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy dqn_white_stockfish --version v3
```

To inicialize the learning process using Q-Learning for an agent that plays black pieces:
```
python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy q_learning_black_stockfish --version v3
```

python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy dqn_white_stockfish_v3 --version v1


python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy dqn_white_stockfish --version v2

python learn.py -f rl_agent_config.yaml --so-type macos --config-strategy dqn_white_stockfish_v2 --version v1
