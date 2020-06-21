from random_search import random_search_policy
from brain import brain
class intelligent_learning_agent:

    def __init__(self, name,env, debug=False):
        self.debug = debug
        self.name = name

        #rewards
        self.lifetime_total_reward = 0
        self.total_reward = 0
        self.best_reward = 0
        
        self.env = env

        self.best_policy = random_search_policy()
        self.policy = random_search_policy()

        self.neural_net = brain(name,env.action_space, env.observation_space,debug=True)
        self.eval_neural_net = None
        print(f"intelligent_learning_agent.{self.name}.__init__() complete")

    def action(self, observation):
        act = self.policy.action(observation)

    def neural_net_action(self, observation):
        print(f"obs: {observation}") if self.debug else None
        action = self.neural_net.predict(observation)

        print(f"action: {action})") if self.debug else None

        return action 

    def neural_net_train(self,batch):
        pass

    def action_best(self, observation):
        act = self.best_policy.action(observation)
        return act

    def reset(self):
        if self.total_reward > self.best_reward:
            self.best_reward = self.total_reward
            self.best_policy = self.policy
        self.policy = random_search_policy()
        self.total_reward = 0 
        pass