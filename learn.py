import time
import chess
import gym
import gym_chess
from tqdm import tqdm

from utils import save_checkpoint, update_game_metrics, get_custom_reward
from agents import QLearningAgent, DQNAgent, MCTSAgent

from stockfish import Stockfish
import argparse
import yaml
import json

# setting arguments for the script
parser = argparse.ArgumentParser(prog='rl_learn', description='Train an agent to play chess.')
parser.add_argument('-f', '--file', type=str, required=True)
parser.add_argument('-c', '--config-strategy', type=str, required=True)
parser.add_argument('-v', '--version', type=str, required=True)
parser.add_argument('-so', '--so-type', type=str, required=True, choices=['ubuntu', 'macos'])
args = parser.parse_args()

# loading the configuration file
with open(args.file) as file:
    rl_config = yaml.load(file, Loader=yaml.FullLoader)

# setting the configuration
CONFIGURATION = rl_config[args.config_strategy]
CONFIGURATION['VERSION'] = args.version

print('--------------- CONFIGURATION ---------------')
print(json.dumps(CONFIGURATION, indent=4))
print('---------------------------------------------')

def self_learn(CONFIGURATION, agent_class, checkpoint=None):
    # Set all the variables
    env = gym.make(CONFIGURATION['ENV_NAME'])
    num_episodes = CONFIGURATION['NUM_EPISODES']
    discount_factor = CONFIGURATION['DISCOUNT_FACTOR']
    alpha = CONFIGURATION['ALPHA']
    epsilon = CONFIGURATION['EPSILON']
    strategy = CONFIGURATION['STRATEGY']
    version = CONFIGURATION['VERSION']
    checkpoint_metrics = CONFIGURATION['CHECKPOINT_METRICS']
    checkpoint_artefacts = CONFIGURATION['CHECKPOINT_ARTEFACTS']

    done = False

    agent_white = agent_class(env)
    agent_black = agent_class(env)

    # verification if it's to start in a q_table already trained
    if not checkpoint:
        game_metrics = {
            'game': [],
            'avg_plays': [],
            'last_reward_white': [],
            'last_reward_black': [],
            'cummulated_reward_white': [],
            'cummulated_reward_black': [],
            'process_time': []
        }

        cummulated_reward_white = 0
        cummulated_reward_black = 0
        game_index = 0
    else:
        agent_white = checkpoint['Q_dict_white']
        agent_black = checkpoint['Q_dict_black']
        game_metrics = checkpoint['game_metrics']
        cummulated_reward_white = game_metrics['cummulated_reward_white'][-1]
        cummulated_reward_black = game_metrics['cummulated_reward_black'][-1]
        game_index = game_metrics['game'][-1]

    old_play_white = {}
    old_play_black = {}
    reward_competior = 0

    print('Start learning process from game: ', game_index+1, ' to game: ', game_index+num_episodes)
    for game in range(game_index+1, game_index+num_episodes+1):
        print('--- Simulation Game: ', game)
        start_time = time.time()
        state = env.reset()
        count = 0

        # initialize a game
        while not done:
            # Selecting legal actions from environment
            legal_actions = env.legal_actions

            if count % 2 == 0: # White scenario
                # evaluate if the movement of the competior was good or not
                reward_competior = get_custom_reward(state, 0, type='old_state')
                # verify if we had a penalization in the last movement from our opponent
                # or if it's the first movement
                if ((reward_competior < 0) and count > 0):
                    # update the q_table with the penalization
                    agent_white.update(
                        old_play_white['legal_actions'], reward_competior, 
                        old_play_white['action'], old_play_white['state'], 
                        old_play_white['next_state'], discount_factor, alpha,
                        done
                        )
                
                # select an action based on the epsilon greedy policy
                action = agent_white.get_epsilon_greedy_action(legal_actions, state, epsilon)
                # extract the next state, reward and if the game is done
                next_state, reward, done, _ = env.step(action)
                # evaluate the reward based on the next state (custom reward)
                custom_reward_white = get_custom_reward(next_state, reward, type='new_state')
                # update the q_table with the custom reward
                agent_white.update(
                    legal_actions, custom_reward_white, action, 
                    state, next_state, discount_factor, alpha,
                    done
                    )

                # save the last play
                old_play_white = {
                    'legal_actions': legal_actions,
                    'state': state,
                    'action': action,
                    'next_state': next_state
                }
                # calculate the cummulated reward
                cummulated_reward_white += custom_reward_white + reward_competior

            else: # Black scenario
                # evaluate if the movement of the competior was good or not
                reward_competior = get_custom_reward(state, 0, type='old_state')
                # verify if we had a penalization in the last movement from our opponent
                # or if it's the first movement
                if ((reward_competior < 0) and count > 0):
                    # update the q_table with the penalization
                     agent_black.update(
                        old_play_black['legal_actions'], reward_competior, 
                        old_play_black['action'], old_play_black['state'], 
                        old_play_black['next_state'], discount_factor, alpha,
                        done
                        )
                     
                # select an action based on the epsilon greedy policy
                action = agent_black.get_epsilon_greedy_action(legal_actions, state, epsilon)
                # extract the next state, reward and if the game is done
                next_state, reward, done, _ = env.step(action)
                # evaluate the reward based on the next state (custom reward)
                custom_reward_black = get_custom_reward(next_state, reward, type='new_state')
                # update the q_table with the custom reward
                agent_black.update(
                    legal_actions, custom_reward_black, action, 
                    state, next_state, discount_factor, alpha,
                    done
                    )

                # save the last play
                old_play_black = {
                    'legal_actions': legal_actions,
                    'state': state,
                    'action': action,
                    'next_state': next_state
                }
                # calculate the cummulated reward
                cummulated_reward_black += custom_reward_black + reward_competior

            # update the state
            state = next_state
            # update the count
            count += 1

        # if the white player won, penalize the black player
        if reward == 1:
            reward_black = -1
            agent_black.update(
                        old_play_black['legal_actions'], reward_black, 
                        old_play_black['action'], old_play_black['state'], 
                        old_play_black['next_state'], discount_factor, alpha,
                        done
                        )
            cummulated_reward_black += reward_black
        # if the black player won, penalize the white player
        elif reward == -1:
            reward_white = -1
            agent_white.update(
                        old_play_white['legal_actions'], reward_white, 
                        old_play_white['action'], old_play_white['state'], 
                        old_play_white['next_state'], discount_factor, alpha,
                        done
                        )
            cummulated_reward_white += reward_white
        else:
            reward_white = 0
            reward_black = 0
        
        # print("--- game: ", game, "count: ", count, "reward white: ", reward_white, 
        #       "reward black: ", reward_black, "cummulated_reward_white: ", round(cummulated_reward_white,2),
        #       "cummulated_reward_black: ", round(cummulated_reward_black,2),
        #       'sum q table', round(sum(agent_white.q_table.values()), 2),
        #       'sum q table', round(sum(agent_black.q_table.values()),2))
        
        finish_time = time.time()
        
        game_metrics_values = {
            'game': game,
            'avg_plays': count,
            'last_reward_white': reward_white,
            'last_reward_black': reward_black,
            'cummulated_reward_white': cummulated_reward_white,
            'cummulated_reward_black': cummulated_reward_black,
            'process_time': round((finish_time-start_time), 2)
        }

        # update the game metrics
        update_game_metrics(game_metrics, game_metrics_values)
        
        # cummulated reward for each player
        cummulated_reward_black = 0
        cummulated_reward_white = 0
        # reset the game
        done = False
        
        # save the metrics for each 50 games played
        if game % checkpoint_metrics == 0:
            print("Simulation Games: ", game, 'Saving intermediate metrics...')
            agents = {
                'agent_white': agent_white,
                'agent_black': agent_black
            }
            save_checkpoint(strategy, version, game, None, game_metrics, save_type='metrics')
        # save the full artefacts for each 250 games played
        if game % checkpoint_artefacts == 0:
            print("---> Simulation Games: ", game, 'Saving full artefacts and metrics...')
            agents = {
                'agent_white': agent_white,
                'agent_black': agent_black
            }
            save_checkpoint(strategy, version, game, agents, game_metrics, save_type='full')

    return agent_white, agent_black, game_metrics

def learn(CONFIGURATION, agent_class, checkpoint=None):
    # Set all the variables
    env = gym.make(CONFIGURATION['ENV_NAME'])
    num_episodes = CONFIGURATION['NUM_EPISODES']
    discount_factor = CONFIGURATION['DISCOUNT_FACTOR']
    alpha = CONFIGURATION['ALPHA']
    epsilon = CONFIGURATION['EPSILON']
    strategy = CONFIGURATION['STRATEGY']
    version = CONFIGURATION['VERSION']
    color_player = CONFIGURATION['COLOR_PLAYER']
    checkpoint_metrics = CONFIGURATION['CHECKPOINT_METRICS']
    checkpoint_artefacts = CONFIGURATION['CHECKPOINT_ARTEFACTS']

    if color_player == 'white':
        player = 0
    elif color_player == 'black':
        player = 1
    else:
        raise ValueError('Color player must be white or black')

    done = False

    agent = agent_class(env)
    if args.so_type == 'ubuntu':
        stockfish = Stockfish("./stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2")
    else:
        stockfish = Stockfish()
    stockfish.set_elo_rating(1)

    # verification if it's to start in a q_table already trained
    if not checkpoint:
        game_metrics = {
            'game': [],
            'avg_plays': [],
            'last_reward': [],
            'cummulated_reward': [],
            'process_time': []
        }

        cummulated_reward = 0
        game_index = 0
    else:
        agent = checkpoint['q_table']
        game_metrics = checkpoint['game_metrics']
        cummulated_reward = game_metrics['cummulated_reward'][-1]
        game_index = game_metrics['game'][-1]

    old_play = {}
    reward_competior = 0

    print('Start learning process from game: ', game_index+1, ' to game: ', game_index+num_episodes)
    for game in range(game_index+1, game_index+num_episodes+1):
        print('--- Simulation Game: ', game)
        start_time = time.time()
        state = env.reset()
        #reseting stockfish
        if args.so_type == 'ubuntu':
            stockfish = Stockfish("./stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2")
        else:
            stockfish = Stockfish()
        stockfish.set_elo_rating(1)
        count = 0

        # initialize a game
        while not done:
            # Selecting legal actions from environment
            legal_actions = env.legal_actions
            if count % 2 == player: # White scenario
                # evaluate if the movement of the competior was good or not
                reward_competior = get_custom_reward(state, 0, type='old_state')
                # verify if we had a penalization in the last movement from our opponent
                # or if it's the first movement
                if ((reward_competior < 0) and count > 0):
                    # update the q_table with the penalization
                    agent.update(
                        old_play['legal_actions'], reward_competior, 
                        old_play['action'], old_play['state'], 
                        old_play['next_state'], discount_factor, alpha,
                        done
                        )
                
                # select an action based on the epsilon greedy policy
                action = agent.get_epsilon_greedy_action(legal_actions, state, epsilon)
                # decode the action to update the stockfish
                decoded_action = str(env.decode(action))
                stockfish.make_moves_from_current_position([decoded_action])
                # extract the next state, reward and if the game is done
                next_state, reward, done, _ = env.step(action)
                # evaluate the reward based on the next state (custom reward)
                custom_reward = get_custom_reward(next_state, reward, type='new_state')
                # update the q_table with the custom reward
                agent.update(
                    legal_actions, custom_reward, action, 
                    state, next_state, discount_factor, alpha,
                    done
                    )

                # save the last play
                old_play = {
                    'legal_actions': legal_actions,
                    'state': state,
                    'action': action,
                    'next_state': next_state
                }
                # calculate the cummulated reward
                cummulated_reward += custom_reward + reward_competior

            else: # Stockfish scenario
                decoded_action = stockfish.get_best_move()
                action = env.encode(chess.Move.from_uci(decoded_action))
                stockfish.make_moves_from_current_position([decoded_action])
                next_state, reward, done, info = env.step(action)
                # print(sum([next_state[:,:,i]*((i%6)+1)  for i in range(12)]))
                
            # update the state
            state = next_state
            # update the count
            count += 1

        if color_player == 'white':
            reward = reward
        else:
            reward = -reward

        agent.update(
            old_play['legal_actions'], reward, 
            old_play['action'], old_play['state'], 
            old_play['next_state'], discount_factor, alpha,
            done
        )
        cummulated_reward += reward
        
        # print("--- game: ", game, "count: ", count, "reward: ", reward, 
        #       "cummulated_reward: ", round(cummulated_reward,2),
        #       'sum q table', sum(agent.q_table.values()))
        
        finish_time = time.time()
        
        game_metrics_values = {
            'game': game,
            'avg_plays': count,
            'last_reward': reward,
            'cummulated_reward': cummulated_reward,
            'process_time': round((finish_time-start_time), 2)
        }

        # update the game metrics
        update_game_metrics(game_metrics, game_metrics_values)
        
        # cummulated reward for each player
        cummulated_reward = 0
        # reset the game
        done = False
        
        # save the metrics for each 50 games played
        if game % checkpoint_metrics == 0:
            print("Simulation Games: ", game, 'Saving intermediate metrics...')
            save_checkpoint(strategy, version, game, None, game_metrics, save_type='metrics')
        # save the full artefacts for each 250 games played
        if game % checkpoint_artefacts == 0:
            print("---> Simulation Games: ", game, 'Saving full artefacts and metrics...')
            save_checkpoint(strategy, version, game, agent, game_metrics, save_type='full')

    return agent, game_metrics

def learn_v2(CONFIGURATION, agent_class, checkpoint=None):
    # Set all the variables
    env = gym.make(CONFIGURATION['ENV_NAME'])
    num_episodes = CONFIGURATION['NUM_EPISODES']
    discount_factor = CONFIGURATION['DISCOUNT_FACTOR']
    alpha = CONFIGURATION['ALPHA']
    epsilon = CONFIGURATION['EPSILON']
    strategy = CONFIGURATION['STRATEGY']
    version = CONFIGURATION['VERSION']
    color_player = CONFIGURATION['COLOR_PLAYER']
    checkpoint_metrics = CONFIGURATION['CHECKPOINT_METRICS']
    checkpoint_artefacts = CONFIGURATION['CHECKPOINT_ARTEFACTS']

    if color_player == 'white':
        player = 0
    elif color_player == 'black':
        player = 1
    else:
        raise ValueError('Color player must be white or black')

    done = False

    agent = agent_class(env)
    if args.so_type == 'ubuntu':
        stockfish = Stockfish("./stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2")
    else:
        stockfish = Stockfish()
    stockfish.set_elo_rating(1)

    # verification if it's to start in a q_table already trained
    if not checkpoint:
        game_metrics = {
            'game': [],
            'avg_plays': [],
            'last_reward': [],
            'cummulated_reward': [],
            'process_time': []
        }

        cummulated_reward = 0
        game_index = 0
    else:
        agent = checkpoint['q_table']
        game_metrics = checkpoint['game_metrics']
        cummulated_reward = game_metrics['cummulated_reward'][-1]
        game_index = game_metrics['game'][-1]

    print('Start learning process from game: ', game_index+1, ' to game: ', game_index+num_episodes)
    for game in range(game_index+1, game_index+num_episodes+1):
        print('--- Simulation Game: ', game)
        start_time = time.time()
        state = env.reset()
        #reseting stockfish
        if args.so_type == 'ubuntu':
            stockfish = Stockfish("./stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2")
        else:
            stockfish = Stockfish()
        stockfish.set_elo_rating(1)
        count = 0

        # initialize a game
        while not done:
            # Selecting legal actions from environment
            legal_actions = env.legal_actions
            if count % 2 == player: # White scenario
                # select an action based on the epsilon greedy policy
                action = agent.get_epsilon_greedy_action(legal_actions, state, epsilon)
                # decode the action to update the stockfish
                decoded_action = str(env.decode(action))
                stockfish.make_moves_from_current_position([decoded_action])
                # extract the next state, reward and if the game is done
                next_state, reward, done, _ = env.step(action)
                # evaluate the reward based on the next state (custom reward)
                custom_reward = get_custom_reward(next_state, reward, type='new_state')
                # update the q_table with the custom reward
                agent.update(
                    legal_actions, custom_reward, action, 
                    state, next_state, discount_factor, alpha,
                    done
                    )
                # calculate the cummulated reward
                cummulated_reward += custom_reward

            else: # Stockfish scenario
                decoded_action = stockfish.get_best_move()
                action = env.encode(chess.Move.from_uci(decoded_action))
                stockfish.make_moves_from_current_position([decoded_action])
                next_state, reward, done, info = env.step(action)
                # evaluate the reward based on the next state from the competitor (custom reward)
                custom_reward = get_custom_reward(next_state, reward, type='new_state')
                # update the q_table with the custom reward
                agent.update(
                    legal_actions, custom_reward, action, 
                    state, next_state, discount_factor, alpha,
                    done
                    )
                
            # update the state
            state = next_state
            # update the count
            count += 1

        if color_player == 'white':
            reward = reward
        else:
            reward = -reward

        cummulated_reward += reward
        
        finish_time = time.time()
        
        game_metrics_values = {
            'game': game,
            'avg_plays': count,
            'last_reward': reward,
            'cummulated_reward': cummulated_reward,
            'process_time': round((finish_time-start_time), 2)
        }

        # update the game metrics
        update_game_metrics(game_metrics, game_metrics_values)
        
        # cummulated reward for each player
        cummulated_reward = 0
        # reset the game
        done = False
        
        # save the metrics for each 50 games played
        if game % checkpoint_metrics == 0:
            print("Simulation Games: ", game, 'Saving intermediate metrics...')
            save_checkpoint(strategy, version, game, None, game_metrics, save_type='metrics')
        # save the full artefacts for each 250 games played
        if game % checkpoint_artefacts == 0:
            print("---> Simulation Games: ", game, 'Saving full artefacts and metrics...')
            save_checkpoint(strategy, version, game, agent, game_metrics, save_type='full')

    return agent, game_metrics

if __name__ == "__main__":
    print('start self learning')
    if CONFIGURATION['TYPE'] =='stockfish':
        print('STARTING TRAINING WITH STOCKFISH')
        if 'q_learning' in CONFIGURATION['STRATEGY']:
            learn_v2(CONFIGURATION, QLearningAgent, checkpoint=None)
        elif 'dqn' in CONFIGURATION['STRATEGY']:
            learn_v2(CONFIGURATION, DQNAgent, checkpoint=None)
        elif 'mcts' in CONFIGURATION['STRATEGY']:
            learn_v2(CONFIGURATION, MCTSAgent, checkpoint=None)
    elif CONFIGURATION['TYPE'] =='self_learning':
        if 'q_learning' in CONFIGURATION['STRATEGY']:
            self_learn(CONFIGURATION, QLearningAgent, checkpoint=None)
        elif 'dqn' in CONFIGURATION['STRATEGY']:
            self_learn(CONFIGURATION, DQNAgent, checkpoint=None)
        elif 'mcts' in CONFIGURATION['STRATEGY']:
            self_learn(CONFIGURATION, MCTSAgent, checkpoint=None)