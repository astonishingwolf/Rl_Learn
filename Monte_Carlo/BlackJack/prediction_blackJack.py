# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 02:55:00 2021

@author: astonishing_wolf

"""
import numpy as np

class Agent():
    def __init__(self , gamma=0.99):
        #value function to store the value of each states
        self.V = {} 
        #to store the all possible sum of cards we can have with
        self.sum_space = [i for i in range(4,22)]
        #to store all the dealer cards
        self.dealer_show_card_space = [i+1 for i in range(10)]
        #whether we have usable ace or not
        self.ace_space = [False,True]
        #to know whether we are hitting or sticking
        self.action_space = [0,1] #stick or hit
        #this is a array which contains tuples of 
        #parameters of each states
        self.state_space = []
        #it is used to store the return of each states 
        #corresponding to each states.In terms of each tuples
        self.returns = {}
        #whether to know we visited that states 
        self.states_visited = {}
        # we will get to know it next time.
        # it will store observation and rewards
        self.memory = []
        #to know disccount factor
        self.gamma =gamma
        self.init_vals()
    # this function is used to intialise the state space to zero   
    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total,card,ace)] = 0
                    self.returns[(total,card,ace)] = []
                    self.states_visited[(total, card ,ace)] =0
                    self.state_space.append((total,card,ace))
    # for this set of experiment we are updating it for a fixed policy
    #hence the policy we are using is a policy which draws a card untill 
    # and unless the value of cards drawn doesn't exceed 20
    def policy(self,state):
        total, _ , _ =state
        action = 0 if total>=20 else 1
        return action
    
    #Here we are updating the value function for the above mentioned policy
    #
    # self,memory stores wverything from all the episodes  and in order to 
    # get a specific one we need to interate . Each episode has a 
    def update_V(self):
        for idt, (state, _) in enumerate(self.memory):
            G=0
            #self.memory comes with a iterator and states and rewards for each episodes
            #the control loop I think it uselesss just to make sure we haven't updated the list earlier
            # the above loop goes through all the points hit on the during a episodes
            if self.states_visited[state]==0:
                self.states_visited[state]+=1
                discount = 1
                # suppose if we are in 5th point in same spisodes then this will
                # calculte discount factor from there and store it
                #state id already intialised and then we are just storing the disounted reward
                # for each states
                for t, (_,reward) in enumerate(self.memory[idt:]):
                    G+=reward*discount
                    discount *=self.gamma
                    self.returns[state].append(G)
        #final V is stored by collecting the mean of all the rewards from a start.            
        for state,_ in self.memory:
            self.V[state] = np.mean(self.returns[state])
            
        for state in self.state_space:
            self.states_visited[state] = 0
                    
        self.memory = []
        
        
                    
        
        
        
        

