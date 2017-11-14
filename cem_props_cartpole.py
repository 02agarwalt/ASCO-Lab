import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
#from cem import CEMAgent
from props import PROPSAgent
from rl.memory import EpisodeParameterMemory

import matplotlib.pyplot as plt
import pandas as pd

ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
#model = Sequential()
#model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(nb_actions))
#model.add(Activation('softmax'))

memory = EpisodeParameterMemory(limit=1000, window_length=1)
train_interval_cem = 50
batch_size_cem = 50
cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=batch_size_cem, nb_steps_warmup=2000, train_interval=train_interval_cem, elite_frac=0.05)
cem.compile()
callback_cem = cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)
cem.test(env, nb_episodes=5, visualize=True)


# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
#model = Sequential()
#model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(nb_actions))
#model.add(Activation('softmax'))

memory = EpisodeParameterMemory(limit=1000, window_length=1)
bound_opts = {'analytic_jac' : True, 'normalize_weights' : True, 'truncate_weights' : True, 'truncate_thresh' : 1}
train_interval_props = 50
batch_size_props = 50

props = PROPSAgent(model=model, nb_actions=nb_actions, memory=memory, bound_opts=bound_opts, batch_size=batch_size_props, nb_steps_warmup=2000, train_interval=train_interval_props)
props.compile()
callback_props = props.fit(env, nb_steps=100000, visualize=False, verbose=2)
props.save_weights('props_{}_params.h5f'.format(ENV_NAME), overwrite=True)
props.test(env, nb_episodes=5, visualize=True)


df_cem = pd.DataFrame({'data': callback_cem.history['episode_reward']})
#plt.plot(callback_cem.history['episode_reward'])
plt.plot(df_cem.rolling(window=train_interval_cem).mean())

df_props = pd.DataFrame({'data': callback_props.history['episode_reward']})
#plt.plot(callback_props.history['episode_reward'])
plt.plot(df_props.rolling(window=batch_size_props).mean())

plt.legend(['cem', 'props'], loc='upper left')
plt.show()
