# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 19:44:01 2021

@author: astonishing wolf

"""

import gym
from Control_blackjack import Agent
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    agent = Agent(eps=0.001)
    n_episodes = 200000
    win_lose_draw = {-1:0, 0:0 ,1:0 }
    win_rates = []
    for i in range(n_episodes):
        if i>0 and i%1000 ==0:
            pct = win_lose_draw[1]/ i
            

