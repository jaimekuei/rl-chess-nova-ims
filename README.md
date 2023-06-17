# rl-chess-nova-ims

Welcome! This is a repository for Reinforcement Learning project ! 

| Student Number | Student Names |
|---|---|
| 20220593 | Joice Preuss | 
| 20220594 | Jaime Kuei | 
| 20220595 | Maxwell Marcos | 
|  | Miguel Ramos |

## Contents
This repository is organanized as follows:

## How to configure the local environment for the project

 ```
python3 -m venv .reinforcement-learning-env
source .reinforcement-learning-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
 ```



python learn.py -f rl_agent_config.yaml --config-strategy q_learning_white_stockfish --version v1
python learn.py -f rl_agent_config.yaml --config-strategy q_learning_black_stockfish --version v1
python learn.py -f rl_agent_config.yaml --config-strategy q_learning --version v1