import simpleAgent
import plotReward

class testBed:

    def __init__(self, env):
        self.agentArray = list()
        self.env = env
        self.currentPlot = None
        self.observation = None
        self.done = False
        self.info = None

    def createAgent(self, name, debug=False):
        newagent = simpleAgent.agent(name, self.env, debug=debug)
        self.agentArray.append(newagent)
        

    def logIteration(self, agent, iter, timeStep,fullLog=False):
        print("Iteration: {}  TimeStep: {}  Agent: {}  CurrentReward: {}".format(iter+1, timeStep, agent.name, agent.totalReward))
        if fullLog:
            print("observation: {}\nDone: {} \nInfo: {}".format(self.observation, self.done, self.info))

    def createPlot(self, agent, iter):
        self.currentPlot = plotReward.plotReward("{} Total Reward Over time. Iteraion {}".format(agent.name, iter+1))

    def logTotalReward(self, agent):
        self.currentPlot.LogResults(agent.totalReward)

    def logAverageReward(self, agent, time):
        self.currentPlot.LogResults(agent.totalReward/time)

    def plotTotalReward(self):
        self.currentPlot.plot()
        pass

    def plotAverageReward(self):
        pass


    def runTraining(self, episode, timesteps, render=False, log=True, createGraph=True):
        for agent in self.agentArray:
            for iter in range(episode):
                self.createPlot(agent, iter) if createGraph else None
                agent.reset()
                self.observation = self.env.reset()
                for time in range(timesteps):
                    if not self.done:
                        self.env.render() if render else None
                        self.observation, reward, self.done, self.info = self.env.step(agent.action(time, self.observation))  # take a random action
                        agent.totalReward += reward
                        self.logIteration(agent, iter, time, fullLog=True) if log else None
                        self.logTotalReward(agent) if createGraph else None
                        #self.logAverageReward(agent, time) if createGraph else None
                    else:
                        self.env.reset()
                self.plotTotalReward() if createGraph else None
                self.done = False

    def runIteration(self, episode, timesteps, render=True, log=True, createGraph=False):
        for agent in self.agentArray:
            for iter in range(episode):
                self.createPlot(agent, iter) if createGraph else None
                self.observation = self.env.reset()
                for time in range(timesteps):
                    if not self.done:
                        self.env.render() if render else None
                        self.observation, reward, self.done, self.info = self.env.step(agent.actionBest(time, self.observation))  # take a random action
                        agent.totalReward += reward
                        self.logIteration(agent, iter, time, fullLog=True) if log else None
                        self.logTotalReward(agent) if createGraph else None
                        #self.logAverageReward(agent, time) if createGraph else None
                    else:
                        self.env.reset()
                self.plotTotalReward() if createGraph else None
                self.done = False
