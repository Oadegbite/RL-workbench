import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer

class action_log:
	def __init__(self, replay_buffer_size,name,debug=False):
		self.name = name
		self.log_list = list()
		self.replay = ReplayBuffer(replay_buffer_size)
		self.debug = debug
		print("action_log.__init__() complete")

	def log_step(self, agent, obs, action, reward, next_obs, done):
		replay.add(obs, action, reward, next_obs, done)
		self.log_list.append(agent.total_reward)

	def plot_reward(self):
		plt.plot(self.log_list,label=self.name)
		plt.xlabel('Steps')
		plt.ylabel('Total Rewards')
		plt.title(self.name)
		plt.grid(True)
		plt.savefig('GraphOf{}.png'.format(self.name))
		plt.clf() #clear the current graph, can be removed to see all values on one chart
