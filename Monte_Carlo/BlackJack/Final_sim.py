# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 04:30:12 2021

@author: astonishing_wold


"""
import gym
from prediction_blackJack import Agent

if __name__=='__main__':
    env = gym.make('BlackJack-v0')
    agent = Agent()
    n_episodes = 500000
    for i in range(n_episodes):
        if i%50000 == 0:
            print('starting episodes')
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_,reward,done,info = env.step(action)
            agent.memory.append((observation, reward))
            observation = observation_
        agent.update_V()
    print(agent.V[(21,3,True)])
    
            
        