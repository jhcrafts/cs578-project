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
def predict_one(recipe,tree,neveringredients=[]):
    currentnode=tree
    depth=0
    while not(isinstance(currentnode,float) or isinstance(currentnode,int) ):
        attribute=currentnode[0]
        if attribute in recipe['ingredients']:
            currentnode=currentnode[1]
        else:
            currentnode=currentnode[2]
        depth+=1
    return(currentnode,depth)
#%%
def predict_all(recipe,majortree,neveringredients=[]):
    result=np.zeros((2,20))
    for i, tree in enumerate(majortree):
        result[:,i]=predict_one(recipe,tree)
    if (np.sum(result[0,:])==1):
        return(np.argmax(result[0,:]))
    elif (np.sum(result[0,:])==0):
        #print('Nobody claimed it, return italian')
        return(17)
    elif (np.sum(result[0,:])>1.):
        return(np.argmin(result[1,result[0,:]==1]))

#%%
def generalentropy(featurefreq,featuresubset,cuisinedict):
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
def subtree(dataset,cuisine,pastingredients=[],max_depth=0,mingain=-1):
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
        count=0
        for recipe in dataset:
            if recipe['cuisine']==cuisine:
                count+=1
        if count==0:
            return(0.)
        if count==len(dataset):
            return(1.)
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
                if(max_depth>3):
                    tree[i+1]=majorityvote(data,cuisine)
                elif (data==[]):
                    tree[i+1]=majorityvote(dataset,cuisine)
                elif(gain<=mingain):
                    tree[i+1]=majorityvote(dataset,cuisine)
                else:
                    #extra iteration
                    tree[i+1]=subtree(data,cuisine,pastingredients,max_depth+1)
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
    bestgain=10000.
    best_ingredient=0.
    i=cuisinedict[cuisine]
    totalfeatures=set(featuresubset[i])
    for ingredient in pastingredients:
        try:
            totalfeatures.remove(ingredient)
        except KeyError:
            pass
    print(totalfeatures)
    for ingredient in totalfeatures:
        entropy_gen,entro=entropy(dataset,ingredient,cuisine)
        if (bestgain>entropy_gen):
            bestgain=entropy_gen
            best_ingredient=ingredient
    print(best_ingredient)
    return(best_ingredient,bestgain)
#%%
def firstpass(dataset,cuisine):
    """
    Gives the ingredient that very unlikely in this cuisine
    """
    featurefreq,featuresubset,cuisinedict=decomposdata(dataset) # make the data readable
    totalfeatures=set()
    pastingredients=[]
    for i in range(20):
        totalfeatures=totalfeatures.union(featuresubset[i])
    for ingredient in totalfeatures:
        entropy_gen,entro=entropy(dataset,ingredient,cuisine)
        print(entro)
        if (entro[0]==0):
            pastingredients+=[ingredient]
    return(pastingredients)
#%%
def entropy(dataset,attribute,cuisine):
    positivein=0.
    negativein=0.
    positiveout=0.
    negativeout=0.
    entro=np.zeros(2)
    if dataset==[]:
        return(0)
    else:
        for recipe in dataset:
            if (attribute in recipe['ingredients'] and recipe['cuisine']==cuisine):
                positivein+=1.
            if (attribute in recipe['ingredients'] and recipe['cuisine']!=cuisine):
                negativein+=1.
            if (not(attribute in recipe['ingredients']) and recipe['cuisine']==cuisine):
                positiveout+=1.
            if (not(attribute in recipe['ingredients']) and recipe['cuisine']!=cuisine):
                negativeout+=1.
        p_positive=positivein/(positivein+negativein)
        p_negative=negativein/(positivein+negativein)
        entro[0]= -p_positive*np.log(p_positive)-p_negative*np.log(p_negative)
        try:
            p_positive=positiveout/(positiveout+negativeout)
            p_negative=negativeout/(positiveout+negativeout)
        except ZeroDivisionError:
            p_positive=0.5
            p_negative=0.5
            print('WARNING: this ingredient is everywhere ! Might want to be more selective')
        entro[1]= -p_positive*np.log(p_positive)-p_negative*np.log(p_negative)
        entro[np.isnan(entro)]=0.
        print(entro)
        gen_entropy=entro[0]*(positivein+negativein)/len(dataset) \
        +entro[1]*(positiveout+negativeout)/len(dataset)
    return(gen_entropy,entro)

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
def makesubmission(majortree):
    testdata = open("test.json")
    for recipe in examples:
        pred=predict_all(recipe,majortree)
#%%

        ## Test
majortree=np.empty(20,dtype='object')
trainingdata = open("train.json")
examples = json.load(trainingdata)
featurefreq,featuresubset,cuisinedict=decomposdata(examples)
cuisines=list(cuisinedict)
comingredients=set()
for subset in featuresubset:
    comingredients=set.intersection(comingredients,subset)
for i,cuisine in enumerate(cuisines):
    # clean the dataset
    pastingredients=list(comingredients)

    for ingredient in featuresubset[cuisinedict[cuisine]]:
        if (featurefreq[cuisinedict[cuisine]][featuresubset[cuisinedict[cuisine]][ingredient]]<3) :
            pastingredients+=[ingredient]
    majortree[i]=subtree(examples,cuisine,pastingredients)
np.save("Treedeep.npy",majortree)
error=0.
missed=0.
for recipe in examples:
    pred=predict_all(recipe,majortree)
    if pred<22:
        if cuisines[pred]!=recipe['cuisine']:
            error+=1
    else:
        missed+=1
print(error)