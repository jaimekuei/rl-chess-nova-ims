import numpy as np
import collections

class QLearningAgent:
    def __init__(self):
        self.q_table = collections.defaultdict(float,{((),0) : 0})
        self.epsilon_greedy_action = None

    def get_max_expected_reward(self, next_state_flatten, legal_actions):
        filtered_dict = {
            (state, action): reward for (state, action), reward in self.q_table.items() 
            if state == next_state_flatten and action in legal_actions
            }
        return max(filtered_dict.values()) if filtered_dict else 0

    def get_action(self, legal_actions, state, epsilon=0.1, method="zeros"):
        if state not in list(self.q_table.keys())[0]:
            self.initialize_q_value(legal_actions, state, method)
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(legal_actions)
        else:
            Q_values = np.array([self.q_table[(state,action)] for action in legal_actions])
            return legal_actions[np.argmax(Q_values)]

    def initialize_q_value(self, legal_actions, state, method):
        if method == "zeros":
            for action in legal_actions:
                self.q_table[((state),action)] = 0
        elif method == "normal":
            for action in legal_actions:
                self.q_table[((state),action)] = np.random.lognormal(0,1)

    def update(self, legal_actions, reward, action, state_flatten, 
            next_state_flatten, discount_factor, alpha):
        # Take the maximum return for the next state
        max_q_next_state = self.get_max_expected_reward(next_state_flatten, legal_actions)
        # TD Target
        td_target = reward + discount_factor * max_q_next_state
        # TD Error
        td_delta = td_target - self.q_table[(state_flatten, action)]
        # Update the Q value
        self.q_table[(state_flatten, action)] += alpha * td_delta