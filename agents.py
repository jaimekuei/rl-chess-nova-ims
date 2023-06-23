import numpy as np
import math
import collections

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class QLearningAgent:
    def __init__(self, env):
        self.q_table = collections.defaultdict(float,{((),0) : 0})
        self.epsilon_greedy_action = None

    def _get_max_expected_reward(self, next_state_flatten, legal_actions):
        filtered_dict = {
            (state, action): reward for (state, action), reward in self.q_table.items() 
            if state == next_state_flatten and action in legal_actions
            }
        return max(filtered_dict.values()) if filtered_dict else 0

    def get_epsilon_greedy_action(self, legal_actions, state, epsilon=0.1, method="zeros"):
        state_flatten = self._flatten_state(state)
        if state_flatten not in list(self.q_table.keys())[0]:
            self._initialize_q_value(legal_actions, state_flatten, method)
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(legal_actions)
        else:
            Q_values = np.array([self.q_table[(state_flatten,action)] for action in legal_actions])
            return legal_actions[np.argmax(Q_values)]

    def _initialize_q_value(self, legal_actions, state, method):
        if method == "zeros":
            for action in legal_actions:
                self.q_table[((state),action)] = 0
        elif method == "normal":
            for action in legal_actions:
                self.q_table[((state),action)] = np.random.lognormal(0,1)

    def update(self, legal_actions, reward, action, state, 
            next_state, discount_factor, alpha, done):
        
        state_flatten = self._flatten_state(state)
        next_state_flatten = self._flatten_state(next_state)
        # Take the maximum return for the next state
        max_q_next_state = self._get_max_expected_reward(next_state_flatten, legal_actions)
        # TD Target
        td_target = reward + discount_factor * max_q_next_state
        # TD Error
        td_delta = td_target - self.q_table[(state_flatten, action)]
        # Update the Q value
        self.q_table[(state_flatten, action)] += alpha * td_delta
    
    def _flatten_state(self, state):
        return tuple(state[:,:,:12].flatten())

class DQNAgent:
    def __init__(self, env):
        #define the state size
        self.state_size = (8, 8, 12)
        #define the action size
        self.action_size = env.action_space.n
        #define the replay buffer
        self.replay_buffer = deque(maxlen=5000)
        #define the total plays
        self.total_plays = 0
        #define counter for update the target model
        self.counter_target = 1
        #define counter for update the main model
        self.counter_main_model = 0
        #define the discount factor
        self.gamma = 0.9
        #define the initial epsilon value
        self.initial_epsilon = 1
        #define the epsilon exponential decay
        self.epsilon_decay = 0.01 # bigger : decays faster
        #define the epsilon
        self.epsilon = 1
        #define the update rate at which we want to update the target network
        self.update_rate = 5
        #learning rate
        self.learning_rate = 0.01
        #batch size
        self.batch_size = 128
        #define the main network
        self.main_network = self.build_network()
        #define the target network
        self.target_network = self.build_network()
        #copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        model = Sequential()
        # Block 1
        model.add(Conv2D(filters=128, kernel_size=(2,2), strides=1,padding='same', activation='relu', input_shape=self.state_size))
        model.add(Conv2D(filters=128, kernel_size=(2,2), strides=1,padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Block 2
        model.add(Conv2D(filters=256, kernel_size=(2,2), strides=1,padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(2,2), strides=1,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # FC layers
        model.add(Flatten())
        model.add(Dropout(0.5))
        # model.add(Dense(216, activation='relu'),kernel_initializer=initializers.GlorotNormal(seed=42))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer=initializers.GlorotNormal(seed=42)))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def get_state_dimensions(self, state):
        # return state[:,:,:12]
        return np.expand_dims(state[:,:,:12],axis=0)

    def get_epsilon_greedy_action(self, legal_actions, state, min_epsilon=0.1, method="zeros"):
        self.epsilon = self._exponential_decay(self.total_plays, min_epsilon)
        state = self.get_state_dimensions(state)
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(legal_actions)
        else:
            legal_actions_dict = {action: self.main_network.predict(state, verbose=0).flatten()[action] for action in legal_actions}
            return max(legal_actions_dict, key=legal_actions_dict.get)

    def _exponential_decay(self, step, min_epsilon):
        new_epsilon = self.initial_epsilon * math.exp(-self.epsilon_decay * step)
        
        if new_epsilon < min_epsilon:
            return min_epsilon
        elif new_epsilon > self.initial_epsilon:
            return self.initial_epsilon
        else:
            return new_epsilon
        
    def update(self, legal_actions, reward, action, state, 
            next_state, discount_factor, alpha, done):
        
        state = self.get_state_dimensions(state)
        next_state = self.get_state_dimensions(next_state)

        if done:
            # reset the counter to update the main model for each episode
            # self.counter_main_model = 0
            # count the episode
            self.counter_target += 1
        # count the iteration inside an episode
        self.counter_main_model += 1
        self.total_plays += 1
        if ((self.counter_target % self.update_rate) == 0) and done:
            print('=== Update the target network ===')
            self.update_target_network()
        
        self.store_transistion(state, action, reward, next_state, done)

        if (len(self.replay_buffer) > self.batch_size) & (self.counter_main_model % 64 == 0):
            print('--- Training the main network ---')
            print('-'*15, f'step n:{self.counter_main_model}', '-'*15)
            self.counter_main_model = 0
            self.train(self.batch_size)
    
    def _process_batch(self, batch):
        reshaped_array = np.expand_dims(batch, axis=(1, 2, 3))
        # Repeat the array along the new dimensions to get shape (5, 8, 8, 12)
        final_array = np.repeat(reshaped_array, repeats=8, axis=1)
        final_array = np.repeat(final_array, repeats=8, axis=2)
        final_array = np.repeat(final_array, repeats=12, axis=3)
        return final_array

    def train(self, batch_size):
        minibatch = np.array(random.sample(self.replay_buffer, batch_size), dtype=object)
        state_list = np.array(minibatch[:,0], dtype=object)
        state_list = np.hstack(state_list).reshape(batch_size, 8, 8, 12)
        # state_list = self._process_batch(state_list)

        # Check the shape and data type of the new array
        print("Batch shape:", state_list.shape)
        print("Batch type:", state_list.dtype)    
        
        next_state_list = np.array(minibatch[:,3],dtype=object)
        next_state_list = np.hstack(next_state_list).reshape(batch_size, 8, 8, 12)
        # next_state_list_processed = self._process_batch(next_state_list)

        current_Q_values_list = self.main_network.predict(tf.convert_to_tensor(state_list, dtype=tf.float32), verbose=0)
        max_q = np.amax(self.target_network.predict(tf.convert_to_tensor(next_state_list), verbose=0), axis=1)

        for i, zip_ in enumerate(minibatch):
            state, action, reward, next_state, done = zip_
            if not done:
                target = reward + self.gamma * max_q[i]
            else:
                target = reward

            updated_Q_value = target # (1 - self.learning_rate)*current_Q_values_list[i][action] + self.learning_rate*(target) # - current_Q_values_list[i][action]) # This is a different form of Q-learning (Min Q-Learning)
            current_Q_values_list[i][action] = updated_Q_value
        
        #train the main network
        self.main_network.fit(state_list, current_Q_values_list, epochs=5, verbose=2)
            
    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

class MCTSAgent:
    def __init__(self, move=None, parent=None, state=None):
        self.move = move # The move that leads to this node
        self.parent = parent # Parent node
        self.children = [] # Child nodes
        self.wins = 0 # Number of wins after simulations
        self.visits = 0 # Number of times the node has been visited
        self.untried_moves = list(state.legal_moves) if state else []  # Untried moves from this state

    def select_child(self):
        # Use UCB1 formula () to select the child with the highest UCB value (https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjr29GthK__AhXqR6QEHbWKAwgQFnoECBoQAQ&url=https%3A%2F%2Fieor8100.github.io%2Fmab%2FLecture%25203.pdf&usg=AOvVaw0RZpgEMI5JpoHfKc43vycx)
        return max(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))

    def expand(self, state):
        # Take an untried move, create a new state, and add a child node
        move = self.untried_moves.pop()
        new_state = state.copy()
        new_state.push(move)
        child_node = MCTSAgent(move=move, parent=self, state=new_state)
        self.children.append(child_node)
        return child_node
    
    def get_epsilon_greedy_action(self, legal_actions, state, min_epsilon=0.1, method="zeros"):
        # implement the rational of MCTS
        pass
        # self.epsilon = self._exponential_decay(self.total_plays, min_epsilon)
        # state = self.get_state_dimensions(state)
        # if np.random.uniform(0,1) < self.epsilon:
        #     return np.random.choice(legal_actions)
        # else:
        #     legal_actions_dict = {action: self.main_network.predict(state, verbose=0).flatten()[action] for action in legal_actions}
        #     return max(legal_actions_dict, key=legal_actions_dict.get)

    def update(self, legal_actions, reward, action, state, 
            next_state, discount_factor, alpha, done):
        # Update the wins and visits count of the node
        self.visits += 1
        self.wins += result

        if done:
            # propagate the reward back to the root node
            pass
        else:
            # take action and update the state



