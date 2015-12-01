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
import re
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
                    new_ingredient+=word
            new_ingredient+=" "
            recipe['ingredients'][j]=str(new_ingredient)
    return(recipe)

def recipestem(recipe):
    for j,ingredient in enumerate(recipe['ingredients']):
        new_ingredient=str()
        for i,word in enumerate(ingredient.split()):
            new_ingredient+=stem(word)
            new_ingredient+=" "
        recipe['ingredients'][j]= new_ingredient
    return(recipe)

def distance(ingredient1,ingredient2):
    similarity=0.
    part1=re.compile('[A-Za-z]+').findall(ingredient1)
    part2=re.compile('[A-Za-z]+').findall(ingredient2)
    for chunk in part1:
        if chunk in part2:
            similarity+=1.
    return(similarity*2/(len(part1)+len(part2)))        
def keeponlywords(recipe):
    for j,ingredient in enumerate(recipe['ingredients']):
        new_ingredient=str()
        new_ingredient=re.compile('[A-Za-z]+').findall(ingredient)
        new_ingredient=' '.join(new_ingredient)
        recipe['ingredients'][j]= new_ingredient
    return(recipe)
def gatheringredients(examples):
    allIngredients=str()
    for example in examples:
        for ingredient in examples['ingredients']:
            allIngredients
if __name__ == '__main__':
    trainingdata = open("train.json")
    examples = json.load(trainingdata)
    #for example in examples:
    #    example=recipestem(example)
