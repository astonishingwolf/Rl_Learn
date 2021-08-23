# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:05:59 2021

@author: astonishing_wolf

"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from pole_agent import Agent

class CartPoleState():
    def __init__(self,bounds =(2.4,4,0.209,4),n_bins=10):
        self.position_space = np.linspace(-1*bounds[0],bounds[0],n_bins)
        self.velocity_space = np.linspace(-1*bounds[1],bounds[1],n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2],bounds[2],n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3],bounds[3],n_bins)
        self.states = self.get_state_space()
    
    #intialises all the states it is used to define the Q value
    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append(i,j,k,l)
        return states
    
    #It is used to digitize the observations.Once we have the observations we can used
    # it to find the approximate state close to it.
    def digitize(self,observation):
        x,x_dot, theta , theta_dot =observation
        cart_x = int(np.digitize(x,self.position_space))
        cart_xdot = int(np.digitize(x_dot,self.velocity_space))
        cart_theta = int(np.digitize(theta,self.pole_angle_space))
        cart_thetadot = int(np.digitize(theta_dot,self.pole_velocity_space))
        return (cart_x,cart_xdot,cart_theta,cart_thetadot)
    


def plot_learning_curve(scores,x):
    run_avg = np.zeros(len(scores))
    for i in range(len(run_avg)):
        run_avg[i] = np.mean(scores[max(0,i-100):i+1])
        
    plt.plot(x,run_avg)
    plt.title('Running avg of previous 100 scores')
    plt.show()
    


if __name__ == '__name__':
    env = gym('CartPole-v0')
    n_games = 50000
    eps_dec = 2/n_games
    digitizer = CartPoleState()
    #The states are then passed on to agent class to make the ways.
    agent = Agent(lr=0.01 , gamma =0.99 , n_actions=2 , eps_start =1.0,eps_end=0.01,eps_dec=eps_dec,state_space=digitizer.states)
    
    scores=[]
    print('it works')
    #The episodes are then looked after one after one.
    for i in range(n_games):
        observation = env.reset()
        done = False
        score =0
        state =digitizer.digitize(observation)
        while not done :
            action = agent.choose_action(state)
            observation_,reward,done,info = env.step(action)
            state_ = digitizer.digitize(observation_)
            agent.learn(state,action,reward,state_)
            state = state_
            score+=reward   
        if i%5000 == 0:
            print('episode',i,'score %.if' %score,'epsilon %.2f'% agent.epsilon)    
        agent.decrement_eps()
        scores.append(score)   
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores,x)
        
            
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
