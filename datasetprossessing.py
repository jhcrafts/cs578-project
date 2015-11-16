# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 21:15:16 2015

@author: francois
tries to clean the dataset
"""
import enchant
from stemming.porter2 import stem
import sys
from math import *
import copy
import random
import json
import numpy as np

d=enchant.Dict("en_US")
def  checkspelling(recipe):
    for j,ingredient in enumerate(recipe['ingredients']):
        new_ingredient=str
        for i,word in enumerate(ingredient.split()):
            if(d.check(word)): # spelling ok
                new_ingredient+=word
            else: # spelling not ok
                try: # the word can be corrected
                    new_ingredient+=d.suggest(word)[0]
                except IndexError: # the word cannot
                    print(word)
                    new_ingredient+=word
            new_ingredient+=" "
            recipe['ingredients'][j]=str(new_ingredient)
    return(recipe)

def recipestem(recipe):
    for j,ingredient in enumerate(recipe['ingredients']):
        new_ingredient=str()
        for i,word in enumerate(ingredient.split()):
            print(word)
            new_ingredient+=stem(word)
            new_ingredient+=" "
        recipe['ingredients'][j]= new_ingredient
    return(recipe)




if __name__ == '__main__':

    for example in examples:
        example=recipestem(example)