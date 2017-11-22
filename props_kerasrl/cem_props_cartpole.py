import numpy as np
import gym
import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from props import PROPSAgent
from rl.memory import EpisodeParameterMemory

import matplotlib.pyplot as plt
import pandas as pd

ENV_NAME = 'CartPole-v0'

parser = argparse.ArgumentParser(description="CEM vs. PROPS")
parser.add_argument("--model_type", default="simple")
parser.add_argument("--train_interval_cem", default=500, type=int)
parser.add_argument("--batch_size_cem", default=500, type=int)
parser.add_argument("--steps_cem", default=100000, type=int)
parser.add_argument("--batch_size_props", default=500, type=int)
parser.add_argument("--steps_props", default=100000, type=int)
parser.add_argument("--trunc_thres", default=1, type=float)
parser.add_argument("--Lmax", default=10, type=int)
parser.add_argument("--delta", default=0.05, type=float)

def main(options):
    # store args
    model_type = options.model_type
    train_interval_cem = options.train_interval_cem
    batch_size_cem = options.batch_size_cem
    steps_cem = options.steps_cem
    batch_size_props = options.batch_size_props
    steps_props = options.steps_props
    trunc_thres = options.trunc_thres
    Lmax = options.Lmax
    delta = options.delta

    # CEM
    # init environment
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    
    model = initModel(model_type, nb_actions, env.observation_space.shape)
    memory = initMemory()
    
    cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=batch_size_cem, nb_steps_warmup=1000, train_interval=train_interval_cem, elite_frac=0.05)
    cem.compile()
    callback_cem = cem.fit(env, nb_steps=steps_cem, visualize=False, verbose=0)
    cem.save_weights('cem_dumps/cem_{}_{}_ti_{}_bs_{}_steps_{}.h5f'.format(ENV_NAME, model_type, train_interval_cem, batch_size_cem, steps_cem), overwrite=True)
    #cem.test(env, nb_episodes=1, visualize=False)

    # PROPS
    # init environment
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    
    model = initModel(model_type, nb_actions, env.observation_space.shape)
    memory = initMemory()

    bound_opts = {'analytic_jac' : True, 'normalize_weights' : True, 'truncate_weights' : True, 'truncate_thresh' : trunc_thres}

    props = PROPSAgent(model=model, nb_actions=nb_actions, memory=memory, Lmax=Lmax, delta=delta, bound_opts=bound_opts, batch_size=batch_size_props)
    props.compile()
    callback_props = props.fit(env, nb_steps=steps_props, visualize=False, verbose=0)
    props.save_weights('props_dumps/props_{}_{}_bs_{}_steps_{}_thres_{}_Lmax_{}_delta_{}.h5f'.format(ENV_NAME, model_type, batch_size_props, steps_props, trunc_thres, Lmax, delta), overwrite=True)
    #props.test(env, nb_episodes=1, visualize=False)

    df_cem = pd.DataFrame({'data': callback_cem.history['episode_reward']})
    #plt.plot(callback_cem.history['episode_reward'])
    plt.plot(df_cem.rolling(window=train_interval_cem).mean())

    df_props = pd.DataFrame({'data': callback_props.history['episode_reward']})
    #plt.plot(callback_props.history['episode_reward'])
    plt.plot(df_props.rolling(window=batch_size_props).mean())

    plt.legend(['cem', 'props'], loc='upper left')
    #plt.show()
    plt.savefig('plots/{}_{}_bs_{}_thres_{}_Lmax_{}_delta_{}.jpeg'.format(ENV_NAME, model_type, batch_size_props, trunc_thres, Lmax, delta))

def initMemory():
    memory = EpisodeParameterMemory(limit=1000, window_length=1)
    return memory

def initModel(model_type, nb_actions, obs_space_shape):
    model = Sequential()
    if model_type == "simple":
        model.add(Flatten(input_shape=(1,) + obs_space_shape))
        model.add(Dense(nb_actions))
        model.add(Activation('softmax'))
    else:
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + obs_space_shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('softmax'))
    return model

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  main(options)
