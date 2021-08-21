# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 03:04:00 2021

@author: dasgu
"""
import numpy as np

class agent():
    def __init__(self, lr gamma , n_actions , state_space , eps_start , eps_end, eps_dec):
        self.lr =lr
        self.gamma =gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        
        self.state_space = state_space
        self.actions = [i for i in range(self.n_actions)]
        
        
        

