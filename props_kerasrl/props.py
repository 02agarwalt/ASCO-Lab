from __future__ import division
from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K
from keras.models import Model

from rl.core import Agent
from rl.util import *
from props_util import *
from scipy.optimize import minimize
import numpy.matlib

class PROPSAgent(Agent):
    def __init__(self, model, nb_actions, memory, th_mean=None, batch_size=500,
                 delta=0.05, bound_opts={},
                 Lmax=10, initial_std=1.0, **kwargs):
        super(PROPSAgent, self).__init__(**kwargs)

        # Related objects
        self.empty_memory = memory
        self.memory = memory
        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.sizes = [w.size for w in model.get_weights()]
        self.num_weights = sum(self.sizes)
        
        # Parameters
        if th_mean is None:
            th_mean = np.zeros(self.num_weights)
        self.nb_actions = nb_actions
        self.batch_size = batch_size
        self.delta = delta
        self.Lmax = Lmax
        self.curr_th_mean = th_mean
        self.curr_th_std = np.ones_like(th_mean) * initial_std
        self.d = th_mean.size
        self.pk0 = NormalDist(th_mean, np.diag(np.power(self.curr_th_std, 2)))
        self.curr_pk = self.pk0
        self.pks = [self.pk0]
        self.yss = None
        self.thss = None
        self.a = .01
        self.tol = 1e-8
        self.min_var = 1e-3;
        self.bound_opts = bound_opts

        self.choose_weights()
        
        # store the best result seen during training, as a tuple (reward, flat_weights)
        self.best_seen = (-np.inf, np.zeros(self.num_weights))

        # State
        self.episode = 0
        self.compiled = False
        self.reset_states()

    def reset_memory(self):
        self.memory = deepcopy(self.empty_memory)
        
    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse')
        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def get_weights_flat(self,weights):
        weights_flat = np.zeros(self.num_weights)

        pos = 0
        for i_layer, size in enumerate(self.sizes):
            weights_flat[pos:pos+size] = weights[i_layer].flatten()
            pos += size
        return weights_flat
        
    def get_weights_list(self,weights_flat):
        weights = []
        pos = 0
        for i_layer, size in enumerate(self.sizes):
            arr = weights_flat[pos:pos+size].reshape(self.shapes[i_layer])
            weights.append(arr)
            pos += size
        return weights          

    def reset_states(self):
        self.recent_observation = None
        self.recent_action = None

    def select_action(self, state, stochastic=False):
        batch = np.array([state])
        if self.processor is not None:
            batch = self.processor.process_state_batch(batch)

        action = self.model.predict_on_batch(batch).flatten()
        if stochastic or self.training:
            return np.random.choice(np.arange(self.nb_actions), p=np.exp(action) / np.sum(np.exp(action)))
        return np.argmax(action)

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def choose_weights(self):
        ths = self.curr_th_mean + self.curr_th_std[None, :][0] * np.random.randn(1, self.curr_th_mean.size)[0]
        ths = self.get_weights_list(ths)
        self.model.set_weights(ths)

    @property
    def layers(self):
        return self.model.layers[:]

    def backward(self, reward, terminal):
	#print(self.episode)
        self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if terminal:
            params = self.get_weights_flat(self.model.get_weights())
            self.memory.finalize_episode(params)

            if self.episode % self.batch_size == 0 and self.episode > 0:
                params, reward_totals = self.memory.sample(self.batch_size)
		#print(params, reward_totals)
                ths = np.array(params)
                ys = np.array(reward_totals)
                #ys_trans = -np.array([ys])
		ys_trans = 200 - np.array([ys])
                ths_trans = np.array([ths]).transpose(2, 1, 0)

                if self.yss is None:
                    self.yss = ys_trans
                else:
                    self.yss = np.append(self.yss, ys_trans, axis=0)

                if self.thss is None:
                    self.thss = ths_trans
                else:
                    self.thss = np.append(self.thss, ths_trans, axis=2)

                # Setup box constraints on optimization vector        
                box_constraints = [(self.tol, None)] # alpha > 0
                # no constraints for mean
                for i in range(0, self.d):
                    box_constraints.append((None, None))
                # need covariance greater than 0 and upper bound which yields valid renyii terms
                j0 = max(0, len(self.pks) - self.Lmax)
                for i in range(0, self.d):
                    Sub_i = None
                    for j in range(j0, len(self.pks)):
                        if Sub_i is None:
                            Sub_i = 2*self.pks[j].S[i,i] - 100*self.tol
                        else:
                            Sub_i = min(Sub_i, 2*self.pks[j].S[i,i] - 100*self.tol)
                    box_constraints.append((self.min_var, Sub_i))

                # set initial guess to previous solution
                x0 = np.concatenate((np.array([self.a]), self.curr_pk.m,  np.diag(self.curr_pk.S)))

                # minimize using L-BFGS-B
                analytic_jac = self.bound_opts.get('analytic_jac')
                bound = lambda a_and_pk : dist_bound_robust_cost_func(a_and_pk, self.pks, self.yss, self.thss, self.delta, self.Lmax, self.bound_opts)
                res = minimize(bound, x0, method='L-BFGS-B', jac=analytic_jac, bounds=box_constraints, options={'disp' : False})

                # convert opt vector to variables
                self.a = res.x[0]
                self.curr_th_mean = res.x[1:(self.d+1)]
                th_cov = res.x[(self.d+1):(2*self.d+1)]
                self.curr_th_std = np.sqrt(th_cov)

                #print(self.curr_th_std[None, :][0])
                #print("hello")
                
                # store policy distribution for future bounds computation
                self.curr_pk = NormalDist(self.curr_th_mean, np.diag(th_cov))
                self.pks.append(self.curr_pk)

                metrics = [np.mean(np.array(reward_totals))]
                if self.processor is not None:
                    metrics += self.processor.metrics
                    
                self.reset_memory()
            self.choose_weights()
            self.episode += 1
        return metrics

    def _on_train_end(self):
        self.choose_weights()
	self.training = False

    @property
    def metrics_names(self):
        names = ['mean_reward']
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names
