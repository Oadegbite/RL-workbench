import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
class brain:

    def __init__(self, agent_name, action_space,observation_space, learning_rate=0.1,eval_epsilon=0,train_epsilon = 0.01,
    discount_factor_gamma = 0.99, debug=False):
        self.name = f"{agent_name}.brain"
        n_actions = action_space.n
        self.action_space = action_space
        self.observation_space = observation_space
        self.eval_epsilon = eval_epsilon
        self.train_epsilon = train_epsilon
        self.debug = debug
        self.discount_factor_gamma = discount_factor_gamma
        obs_shape = observation_space.shape
        print(obs_shape)
        observations_input = keras.layers.Input(obs_shape, name='observations_input')
        action_mask = keras.layers.Input((n_actions,), name='action_mask')
        hidden = keras.layers.Dense(32, activation='relu')(observations_input)
        hidden_2 = keras.layers.Dense(32, activation='relu')(hidden)
        output = keras.layers.Dense(n_actions)(hidden_2)
        filtered_output = keras.layers.multiply([output, action_mask])
        model = keras.models.Model([observations_input, action_mask], filtered_output)
        optimizer = keras.optimizers.Adam(lr=learning_rate, clipnorm=1.0)
        model.compile(optimizer, loss='mean_squared_error')
        self.model = model
        print(f"{agent_name}.brain.__init__() complete")

        self.model.summary() if self.debug else None

    def predict(self, observations):
        action_mask = np.ones((len(observations), self.action_space.n))
        print(f"{self.name}.predict()\nobs: {observations.shape} \naction_mask: {action_mask}") if self.debug else None

        prediction = self.model.predict(x=[observations, action_mask])

        print(f"prediction: {prediction}") if self.debug else None
        
        return prediction

    def fit_batch(self,target_model, batch):
        observations, actions, rewards, next_observations, dones = batch
        # Predict the Q values of the next states. Passing ones as the action mask.
        next_q_values = self.predict(target_model, next_observations)
        # The Q values of terminal states is 0 by definition.
        next_q_values[dones] = 0.0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        q_values = rewards + self.discount_factor_gamma * np.max(next_q_values, axis=1)
        one_hot_actions = np.array([one_hot_encode(self.action_space.n, action) for action in actions])
        history = model.fit(
            x=[observations, one_hot_actions],
            y=one_hot_actions * q_values[:, None],
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        return history.history['loss'][0]

    def greedy_action(self, observation):
        next_q_values = self.predict(observations=[observation])
        return np.argmax(next_q_values)


    def epsilon_greedy_action(self, observation, epsilon):
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            action = self.greedy_action(observation)
        return action


    def save_model(model, step, logdir, name):
        filename = '{}/{}-{}.h5'.format(logdir, name, step)
        model.save(filename)
        print('Saved {}'.format(filename))
        return filename


    
    
    def one_hot_encode(n, action):
        one_hot = np.zeros(n)
        one_hot[int(action)] = 1
        return one_hot


    def load_or_create_model(env, model_filename):
        if model_filename:
            model = keras.models.load_model(model_filename)
            print('Loaded {}'.format(model_filename))
        else:
            model = create_model(env)
        model.summary()
        return model
    

