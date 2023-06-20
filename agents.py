import numpy as np
import collections

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class QLearningAgent:
    def __init__(self, env):
        self.q_table = collections.defaultdict(float,{((),0) : 0})
        self.epsilon_greedy_action = None

    def _get_max_expected_reward(self, next_state, legal_actions):
        next_state_flatten = self._flatten_state(next_state)
        filtered_dict = {
            (state, action): reward for (state, action), reward in self.q_table.items() 
            if state == next_state_flatten and action in legal_actions
            }
        return max(filtered_dict.values()) if filtered_dict else 0

    def get_epsilon_greedy_action(self, legal_actions, state, epsilon=0.1, method="zeros"):
        if state not in list(self.q_table.keys())[0]:
            self._initialize_q_value(legal_actions, state, method)
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(legal_actions)
        else:
            Q_values = np.array([self.q_table[(state,action)] for action in legal_actions])
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
        self.replay_buffer = deque(maxlen=1000)
        #define counter for update the target model
        self.counter_target = 0
        #define counter for update the main model
        self.counter_main_model = 0
        #define the discount factor
        self.gamma = 0.9
        #define the epsilon value
        self.epsilon = 0.99
        #define the update rate at which we want to update the target network
        self.update_rate = 5
        #define the main network
        self.main_network = self.build_network()
        #define the target network
        self.target_network = self.build_network()
        #copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())
        #learning rate
        self.learning_rate = .0001
        
    #Let's define a function called build_network which is essentially our DQN. 

    #NAO TA FUNCIONANDO AINDA
    def build_network(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=.0001, epsilon=1e-7))
        return model

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def get_state_dimensions(self, state):
        return state[:,:,:12]

    def get_epsilon_greedy_action(self, legal_actions, state, epsilon=0.1, method="zeros"):
        state = self.get_state_dimensions(state)
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(legal_actions)
        else:
            legal_actions_dict = {action: self.main_network.predict(state, verbose=0).flatten()[action] for action in legal_actions}
            return max(legal_actions_dict, key=legal_actions_dict.get)

    def update(self, legal_actions, reward, action, state, 
            next_state, discount_factor, alpha, done):
        
        state = self.get_state_dimensions(state)
        if done:
            # reset the counter to update the main model for each episode
            self.counter_main_model = 0
            # count the episode
            self.counter_target += 1
        # count the iteration inside an episode
        self.counter_main_model += 1

        if self.counter_target % self.update_rate == 0:
            print('=== Update the target network ===')
            self.update_target_network()
        
        self.store_transistion(state, action, reward, next_state, done)

        if (len(self.replay_buffer) > self.batch_size) & (self.counter_main_model % 10 == 0):
            print('--- Training the main network ---')
            self.train(self.batch_size)

    def train(self, batch_size):
        minibatch = np.array(random.sample(self.replay_buffer, batch_size), dtype=object)

        state_list = np.array(minibatch[:,0], dtype=object)
        state_list = np.hstack(state_list).reshape(batch_size, 8, 8, 12)

        next_state_list = np.array(minibatch[:,3])
        next_state_list = np.hstack(next_state_list).reshape(batch_size, 8, 8, 12)

        current_Q_values_list = self.main_network.predict(state_list, verbose=0)
        max_q = np.amax(self.target_network.predict(next_state_list, verbose=0), axis=1)

        for i, zip_ in enumerate(minibatch):
            state, action, reward, next_state, done = zip_
            if not done:
                target = reward + self.gamma * max_q[i]
            else:
                target = reward

            updated_Q_value = target # (1 - self.learning_rate)*current_Q_values_list[i][action] + self.learning_rate*(target) # - current_Q_values_list[i][action]) # This is a different form of Q-learning (Min Q-Learning)
            current_Q_values_list[i][action] = updated_Q_value
        
        #train the main network
        self.main_network.fit(state_list, current_Q_values_list, epochs=1, verbose=0)
            
    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())