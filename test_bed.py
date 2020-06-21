import ILA
from action_log import action_log
import cv2

class test_bed:

    def __init__(self, env, name,replay_buffer_size=50000,log_every=2000,debug=False):
        self.agent_array = list()
        self.env = env
        self.current_plot = None
        self.observation = None
        self.done = False
        self.info = None
        self.debug = debug
        self.log_every = log_every
        self.action_log = action_log(replay_buffer_size,name)

    def create_agent(self, name, debug=False):
        newagent = ILA.intelligent_learning_agent(name, self.env, debug=debug)
        self.agent_array.append(newagent)
        
    def create_plot(self, agent, episodes):
        self.current_plot = action_log.plot_reward("{} Total Reward Over {} episodes".format(agent.name, episodes))

    def run_training(self, episode, timesteps, render=False, create_graph=True):
        for agent in self.agent_array:
            for iter in range(episode):
                agent.reset()
                self.observation = self.env.reset()
                for time in range(timesteps):
                    if not self.done:
                        self.env.render() if render else None
                        action = agent.action(self.observation)
                        pre_obs = self.observation
                        self.observation, reward, self.done, self.info = self.env.step(action)  
                        agent.total_reward += reward
                        self.action_log.log_step(agent, pre_obs, action, reward,self.observation, self.done)
                    else:
                        self.env.reset()
                        break
                self.done = False
        self.create_plot if create_graph else None
    
    def run_neural_network_training(self, episode, timesteps, render=False, create_graph=True):
        for agent in self.agent_array:
            for iter in range(episode):
                agent.reset()
                self.observation = self.env.reset()
                for time in range(timesteps):
                    if not self.done:
                        self.env.render() if render else None
                        action = agent.neural_net_action(self.observation)
                        pre_obs = self.observation
                        self.observation, reward, self.done, self.info = self.env.step(action)  
                        agent.total_reward += reward
                        self.action_log.log_step(agent, pre_obs, action, reward,self.observation, self.done)
                    else:
                        self.env.reset()
                        break
                self.done = False
        self.create_plot if create_graph else None

    def run_neural_network_iteration(self, episode, timesteps, render=True, create_graph=False):
        for agent in self.agent_array:
            for iter in range(episode):
                self.create_plot(agent, iter) if create_graph else None
                self.observation = self.env.reset()
                for time in range(timesteps):
                    if not self.done:
                        self.env.render() if render else None
                        action = agent.neural_net_action(self.observation)
                        self.observation, reward, self.done, self.info = self.env.step(action)
                        agent.total_reward += reward
                    else:
                        self.env.reset()
                self.plot_total_reward() if create_graph else None
                self.done = False
    
    def run_iteration(self, episode, timesteps, render=True, create_graph=False):
        for agent in self.agent_array:
            for iter in range(episode):
                self.create_plot(agent, iter) if create_graph else None
                self.observation = self.env.reset()
                for time in range(timesteps):
                    if not self.done:
                        self.env.render() if render else None
                        self.observation, reward, self.done, self.info = self.env.step(agent.action_best(time, self.observation))  
                        agent.total_reward += reward
                    else:
                        self.env.reset()
                self.plot_total_reward() if create_graph else None
                self.done = False


    def save_image(self, episode, step):
        frame = self.env.render(mode='rgb_array')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # following cv2.imwrite assumes BGR
        filename = "{}_{:06d}.png".format(episode, step)
        cv2.imwrite(filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])