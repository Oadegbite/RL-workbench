import gym
import simpleAgent
import testBed

env = gym.make('CartPole-v0')
env.reset()

'''
ILA = simpleAgent.agent("ILA",env)
Ultron = simpleAgent.agent("Ultron",env)

for x in range(1000):
    env.render()
    observation, reward, done, info = env.step(ILA.action(x)) # take a random action
    ILA.totalReward += reward
    print("Run")

for x in range(1000):
    env.render()
    env.step(Ultron.action(x)) # take a random action

'''

cartPoleBed = testBed.testBed(env)
cartPoleBed.createAgent("ILA",debug=True)
#cartPoleBed.createAgent("Ultron")
cartPoleBed.runTraining(100, 1000)
cartPoleBed.runIteration(10, 1000)
