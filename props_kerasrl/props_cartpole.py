import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from props import PROPSAgent
from rl.memory import EpisodeParameterMemory

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
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))


print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=1)

bound_opts = {'analytic_jac' : True, 'normalize_weights' : True, 'truncate_weights' : True}

props = PROPSAgent(model=model, nb_actions=nb_actions, memory=memory, bound_opts=bound_opts, batch_size=250, nb_steps_warmup=2000, train_interval=50)
props.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
props.fit(env, nb_steps=200000, visualize=False, verbose=2)

# After training is done, we save the best weights.
props.save_weights('props_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
props.test(env, nb_episodes=5, visualize=True)
