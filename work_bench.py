import gym
import test_bed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import strftime, time

env = gym.make('CartPole-v0')
env.reset()

def main(args):
    cartPoleBed = test_bed.test_bed(env,args.name)
    cartPoleBed.create_agent("ILA",debug=True)
    #cartPoleBed.createAgent("Ultron")
    cartPoleBed.run_neural_network_training(100, 1000)
    cartPoleBed.run_neural_network_iteration(10, 1000)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval', action='store_true', default=False, help='run evaluation with log only')
    parser.add_argument('--images', action='store_true', default=False, help='save images during evaluation')
    parser.add_argument('--model', action='store', default=None, help='model filename to load')
    parser.add_argument('--name', action='store', default=strftime("%m-%d-%H-%M"), help='name for saved files')
    parser.add_argument('--seed', action='store', type=int, help='pseudo random number generator seed')
    parser.add_argument('--test', action='store_true', default=False, help='run tests')
    parser.add_argument('--view', action='store_true', default=False, help='view the model playing the game')
    main(parser.parse_args())