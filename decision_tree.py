# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:59:36 2015

@author: francois jordan

Build decision trees:
Does this recipe belong to that cuisine ?
"""
from copy import *
import numpy as np
import json
from cs578project import *
#%%
def predict_one(recipe,tree):
    currentnode=tree
    while not(isinstance(currentnode,float) or isinstance(currentnode,int) ):
        attribute=currentnode[0]
        if attribute in recipe['ingredients']:
            currentnode=currentnode[1]
        else:
            currentnode=currentnode[2]
    return(currentnode)
#%%
def entropy(featurefreq,featuresubset,cuisinedict):
    """
    Computes the entropy for every ingredient the smaller the entropy the more
    information gain for that ingredient
    inputs:
    -the feature freq per cuisine : featurefreq[cuisine][ingredient] (np array)
    = freq of that ingredient  indexed  by dictionnary cuisinedict
    (type of cuisine :chinese mexican ...)
    -dictionnary cuisinedict (type of cuisine :chinese mexican ...)
    -dictionnaries featuresubset[i]=  the dictionnary of ingredients for
    cuisine i  retunrs ingredient index
    outputs:
    - a numpy array entropy with numerical values (keep featuresubsert as index)
    - totalfreq indorder to take only frequent word
    """
    entropy=np.empty(20,dtype='object')
    frequency=np.empty(20,dtype='object')
    for i,subset in enumerate(featuresubset): # gather all the feature in one set
        entropy[i]=np.zeros(len(subset))
        frequency[i]=np.zeros(len(subset))
#    totalfeatures=listodict(list(totalfeatures)) # turn it into a dict for indexing
    totalfeatures=set()
    for i in range(20):
        totalfeatures=totalfeatures.union(featuresubset[i])
    for ingredient in totalfeatures:
        totalfreq=0.
        for cuis in cuisinedict:
            i=cuisinedict[cuis]
            try :
                f=featurefreq[i][featuresubset[i][ingredient]]
            except KeyError:
                f=0.
            totalfreq+=f
        count=0
        for cuis in cuisinedict:
            i=cuisinedict[cuis]
            try :
                p=featurefreq[i][featuresubset[i][ingredient]]/totalfreq
                entropy[i][featuresubset[i][ingredient]]+= -p*log(p)
                frequency[i][featuresubset[i][ingredient]]=totalfreq
            except KeyError:
                count+=1
        if (count==20):
            print('arg!')
    return(entropy,frequency)
#%%
def classify(decisionTree, example):
    currentnode=decisionTree
    while not(isinstance(currentnode,float) or isinstance(currentnode,int) ):
        attribute=currentnode[0]
        t=currentnode[3]
        score=example[attribute]
        if (score<=t):
            currentnode=currentnode[1]
        if (score>t):
            currentnode=currentnode[2]
    return(currentnode)
#%%
def decomposdata(examples):
    """
    Based on the examples, computes the frequency of every ingredient for
    for every type of cuisine. It can be useful in order to compute entropy
    the values are stored in numpy arrays
    -featurefreq[cuisine][ingredient] = freq of that ingredient
    indexed by
    -dictionnary cuisinedict (type of cuisine :chinese mexican ...) returns
    cuisine index
    -dictionnaries featuresubset[i]=  the dictionnary of ingredients for
    cuisine i  retunrs ingredient index
    """
    featuresubset=np.empty(20,dtype=object)
    #feature subset: np array of sets of the ingredient of each cuisine
    for i in range(len(featuresubset)):
        featuresubset[i]=set()
    cuisinedict={}
    labels = set()
    for example in examples:
        if not(example["cuisine"] in labels): # to build the dictionnary index on the fly
            labels.add(example["cuisine"])
            cuisinedict[example["cuisine"]]=len(labels)-1
        for ingredient in example["ingredients"]:
            featuresubset[cuisinedict[example["cuisine"]]].add(ingredient)
    featurefreq=np.empty(20,dtype=object)
    for i,subset in enumerate(featuresubset):
        featurefreq[i]=np.zeros(len(subset))
        featuresubset[i]=listodict(list(subset))
    for example in examples:
        cuisine=cuisinedict[example["cuisine"]]
        for ingredient in example["ingredients"]:
            featurefreq[cuisine][featuresubset[cuisine][ingredient]]+=1.
    return(featurefreq,featuresubset,cuisinedict)
#%%
def subtree(dataset,cuisine,pastingredients=[],max_depth=9,mingain=-1):
    """
    Fundamental dexision tree learning function
    inputs:
    -dataset to split
    outputs:
    -np array of the node with the attribute we split on
    -separated datasets

    """
    if (dataset==[]):
        #make sure the dataset is not empty (I can't imagine this happening in practise)
        return(0.)
    else:
        tree=np.empty(3,dtype='object') #node
        ingredient,gain=best_choice(dataset,pastingredients,cuisine)
        if ingredient==1:
            return(0)
        # choice of the ingredient
        tree[0]=ingredient
        #split the dataset
        datasetwith,datasetwithout=split(dataset,ingredient)
        pastingredients+=[ingredient]
        datasets=np.empty(2,dtype='object')
        datasets[0]=datasetwith
        datasets[1]=datasetwithout
        for i,data in enumerate(datasets):
            #stopping condition
            if(len(pastingredients)>max_depth):
                tree[i+1]=majorityvote(data,cuisine)
            elif (data==[]):
                tree[i+1]=majorityvote(dataset,cuisine)
            elif(gain<=mingain):
                tree[i+1]=majorityvote(dataset,cuisine)
            else:
                #extra iteration
                tree[i+1]=subtree(data,cuisine,pastingredients,max_depth)
    return(tree)

#%%
def searchdict(index,dic):
    for attribute,key in dic.iteritems():
        if key==index:
            return(attribute)
    return(1)
#%%

def best_choice(dataset,pastingredients,cuisine):
    """
    Chooses the best ingredient
    """
    featurefreq,featuresubset,cuisinedict=decomposdata(dataset) # make the data readable
    entropy_gain,frequency=entropy(featurefreq,featuresubset,cuisinedict)
    try:
        index=np.argmin(entropy_gain[cuisinedict[cuisine]][frequency[cuisinedict[cuisine]]>20])
    except (ValueError,KeyError):
        return(1,-10)
    best_attribute=searchdict(index,featuresubset[cuisinedict[cuisine]])
    print(best_attribute)
    return(best_attribute,index)

def random_choice(attributes,dataset,t):
    index_best=int(np.random.uniform(0,len(attributes)))
    best_attribute=attributes[index_best]
    return(best_attribute,index_best)

#%%
def threshold_optimize(dataset,attribute_index):
    # find the best threshold to optimze entropy gain
    t=np.mean(dataset[:,attribute_index])
    t=np.random.uniform(10)
  #  t=10.*(np.sum(dataset[:,-1])*1./len(dataset[:,-1]))
    t=5.
    #bissection method
    t1=5.
    i=0
#    print('new iteration')
#    print(entropygain(dataset,attribute_index,t1))
    while(i<5):
        r1=entropygain(dataset,attribute_index,t1)
        r2=entropygain(dataset,attribute_index,t1-1)
        r3=entropygain(dataset,attribute_index,t1+1)
        if r2==max(r1,r2,r3):
            t1=t1-1
        elif r3==max(r1,r2,r3):
            t1=t1+1
        else:
            pass
        i=i+1
#        print(entropygain(dataset,attribute_index,t1))
 #   print(t1)
    return(t1)
#%%
def split(dataset,ingred):
    datasetwith=list()
    datasetwithout=deepcopy(dataset)
    a=[1]
    for recipe in dataset:
       if ingred in recipe['ingredients']:
           a[0]=recipe
           datasetwith+=a
           datasetwithout.remove(recipe)
    return (datasetwith,datasetwithout)
 #%%
def majorityvote(dataset,cuisine):
    """
    takes the majority vote of a set
    1: the cuisine dominates
    0.: the other cuisines dominate
    """
    vote=0.
    for recipe in dataset:
        if recipe['cuisine']==cuisine:
            vote+=1
    if vote>len(dataset)/2:
        return(1.)
    else:
        return(0.)
#%%
trainingdata = open("train.json")
examples = json.load(trainingdata)
cuisine='italian'
tree=subtree(examples,cuisine)
error=0.
for recipe in examples:
    pred=predict_one(recipe,tree)
    if (int(pred)==1 and recipe['cuisine']!=cuisine):
        print('negative error' )
        error+=1
    if (int(pred)==0 and recipe['cuisine']==cuisine):
        print('positive error' )
        error+=1
print('error:',error*1./len(examples))
