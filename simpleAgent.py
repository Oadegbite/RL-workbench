from random_search import random_search_policy

class agent:

    def __init__(self, name,env, debug=False):
        self.debug = debug
        self.name = name
        self.totalReward = 0
        self.env = env
        self.bestReward = 0
        self.bestPolicy = random_search_policy()
        self.policy = random_search_policy()
        print("agent init {}".format(self.name)) if (self.debug) else None

    def action(self, step, observation):
        act = self.policy.action(observation)
        print("Action: {} @ t = {} method for {}, observation: {}".format(act,step,self.name, observation)) if (self.debug) else None
        return act

    def actionBest(self, step, observation):
        act = self.bestPolicy.action(observation)
        print("Action: {} @ t = {} method for {}, observation: {}".format(act,step,self.name, observation)) if (self.debug) else None
        return act

    def reset(self):
        if self.totalReward > self.bestReward:
            self.bestReward = self.totalReward
            self.bestPolicy = self.policy
        self.policy = random_search_policy()
        self.totalReward = 0 
        pass