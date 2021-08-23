# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 03:04:00 2021

@author: dasgu
"""
import numpy as np

class Agent():
    def __init__(self, lr , gamma , n_actions , state_space , eps_start , eps_end, eps_dec):
        self.lr =lr
        self.gamma =gamma
        self.epsilon = eps_start # epsilon is intialised
        self.eps_end = eps_end # point where the epsilon is decreament in reduced
        self.eps_dec = eps_dec # decreament level by which epsilon is decreased.
        self.n_actions = n_actions
        
        self.state_space = state_space
        self.actions = [i for i in range(self.n_actions)]
        
        self.Q = {}
        
        self.init_q()
    
    #All the Q values are then intialised to zero.    
    def init_q(self):
        for state in self.state.space:
            for action in self.actions:
                self.Q[(state,action)] = 0.0
    
    #this will choose the maximum action based upon their Q-Value
    def max_action(self,state):
        actions = np.array(self.Q[(state,a)] for a in self.action_space)
        action = np.argmax(actions)
        return action
    
    #this will choose the next action based upon e-greedy method
    def choose_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.max_action(state)
        return action
    
    #epsilon is decreased ...before it started with uniform random but after that it gets reduced
    ## to such a extend that it gets to schocastic form.
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
        if self.epsilon > self.eps_end else self.eps_end
            
    #the actions in the selected in the e-greedy method and after that in the learning process
    #the state we will select the action with the maximum Q value.
    def learn(self, state, action, reward , state_):
        a_max = self.max_action(state_)
        self.Q[(state,action)] = self.Q[(state,action)]+self.lr*(reward + self.gamma*self.Q[(state,a_max)] - self.Q[(state,action)])
         
        
        
        
        
        
        

