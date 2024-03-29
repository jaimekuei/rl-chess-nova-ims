import os
import numpy as np
import re
import pickle

def get_last_checkpoint_path(strategy, version):
    """
    Get the path of the last checkpoint

    Parameters
    ----------
    strategy : str
        The strategy used to train the agent
    version : str
        The version of the strategy

    Returns
    -------
    path : str
        The path of the last checkpoint
    """
    path = os.path.join(os.getcwd(), 'checkpoint', strategy, version)
    # select only folders with iteration
    folders = [folder for folder in os.listdir(path) if 'iteration' in folder]
    iterations_index = np.argmax([int(folder.split('_')[1]) for folder in folders])
    return os.path.join(path, folders[iterations_index]), folders[iterations_index]

def load_checkpoint(strategy, version, iteration='latest'):
    """
    Load a checkpoint

    Parameters
    ----------
    strategy : str
        The strategy used to train the agent
    version : str
        The version of the strategy

    Returns
    -------
    checkpoint : dict
        A dictionary with the agent, the Q_dict and the game_metrics
    """
    if iteration == 'latest':
        checkpoint_info = get_last_checkpoint_path(strategy, version)
        path = checkpoint_info[0]
        iteration_name = checkpoint_info[1]

        path_agent = os.path.join(path, 'checkpoint_agent.pkl')
        # path_black = os.path.join(path, 'checkpoint_q_dict_black.pkl')
        path_metrics = os.path.join(path, 'checkpoint_game_metrics.pkl')
    else:
        print('AQUI')
        if re.match(r'iteration_\d+', iteration) is not None:
            path_agent = os.path.join(os.getcwd(), 'checkpoint', strategy, version, iteration, 'checkpoint_agent.pkl')
            # path_black = os.path.join(os.getcwd(), 'checkpoint', strategy, version, iteration, 'checkpoint_q_dict_black.pkl')
            path_metrics =os.path.join(os.getcwd(), 'checkpoint', strategy, version, iteration, 'checkpoint_game_metrics.pkl')
        else:
            raise ValueError('Iteration not found')
    
    print(path_agent)

    print('Loading checkpoint from: ', iteration_name)
    with open(path_agent, 'rb') as f:
        agent = pickle.load(f)
    print('----> Loaded agent')
    # with open(path_black, 'rb') as f:
    #     Q_dict_black = pickle.load(f)
    # print('----> Loaded Q_dict_black')
    with open(path_metrics, 'rb') as f:
        results = pickle.load(f)
    print('----> Loaded game_metrics')
    
    checkpoint = {
        'agent': agent,
        'game_metrics': results
    }

    return checkpoint

def save_checkpoint(strategy, version, iteration, agent, game_metrics, save_type='full'):
    """
    Save a checkpoint

    Parameters
    ----------
    strategy : str
        The strategy used to train the agent
    version : str
        The version of the strategy
    iteration : int
        The iteration number
    agent : Agent
        The agent to save
    game_metrics : dict
        The game metrics to save
    save_type : str
        The type of checkpoint to save
    """
    checkpoint_folder = os.path.join(os.getcwd(), 'checkpoint')

    if save_type == 'full':
        artifacts_folder = os.path.join(checkpoint_folder, strategy, version, f'iteration_{iteration}')
        if not os.path.exists(artifacts_folder):
            os.makedirs(artifacts_folder)
        
        file_name = 'checkpoint_agent.pkl'
        with open(os.path.join(artifacts_folder, file_name), 'wb') as f:
            pickle.dump(agent, f)
        
        file_name = 'checkpoint_game_metrics.pkl'
        with open(os.path.join(artifacts_folder, file_name), 'wb') as f:
            pickle.dump(game_metrics, f)
    elif save_type == 'metrics':
        metrics_folder = os.path.join(checkpoint_folder, strategy, version, 'history_metrics')
        if not os.path.exists(metrics_folder):
            os.makedirs(metrics_folder)

        file_name = f'checkpoint_game_metrics_iteration_{iteration}.pkl'
        with open(os.path.join(metrics_folder, file_name), 'wb') as f:
            pickle.dump(game_metrics, f)

    print(f'Iteration {iteration} - Checkpoint saved !')

def update_game_metrics(game_metrics, game_metrics_values):
    """
    Update the game metrics

    Parameters
    ----------
    game_metrics : dict
        The game metrics
    game_metrics_values : dict
        The new game metrics values
    """
    for key, value in game_metrics_values.items():
        game_metrics[key].append(value)

def get_custom_reward(state, reward, type='new_state'):
    """
    Get the custom reward

    Parameters
    ----------
    state : np.array
        The state of the game
    reward : float
        The reward
    type : str
        The type of reward to get

    Returns
    -------
    reward : float
        The custom reward
    """
    index = 0
        
    current_pawns = sum(state[:,:,index].flatten())
    last_pawns = sum(state[:,:,index+14].flatten())
    current_knights = sum(state[:,:,index+1].flatten())
    last_knights = sum(state[:,:,index+1+14].flatten())
    current_bishops = sum(state[:,:,index+2].flatten())
    last_bishops = sum(state[:,:,index+2+14].flatten())
    current_rooks = sum(state[:,:,index+3].flatten())
    last_rooks = sum(state[:,:,index+3+14].flatten())
    current_queens = sum(state[:,:,index+4].flatten())
    last_queens = sum(state[:,:,index+4+14].flatten())
    
    if type == 'new_state':
        # penalization for repetition boards
        if state[:,:,12].mean() == 1:
            reward += -0.001
        elif state[:,:,13].mean() == 1:
            reward += -0.005
    
        # reward for taking pieces
        reward += 0.01 * (last_pawns - current_pawns)
        reward += 0.03 * (last_knights - current_knights)
        reward += 0.03 * (last_bishops - current_bishops)
        reward += 0.05 * (last_rooks - current_rooks)
        reward += 0.1 * (last_queens - current_queens)
    
    elif type == 'old_state':
        # reward for taking pieces
        reward -= 0.01 * (last_pawns - current_pawns)
        reward -= 0.03 * (last_knights - current_knights)
        reward -= 0.03 * (last_bishops - current_bishops)
        reward -= 0.05 * (last_rooks - current_rooks)
        reward -= 0.1 * (last_queens - current_queens)

    return reward
