# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:59:36 2015

@author: francois jordan

Build decision trees:
Does this recipe belong to that cuisine ?
"""

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
def subtree(dataset,max_depth=9,mingain=0):
    if (featuresubset==[]):
        #make sure the dataset is not empty (I can't imagine this happening in practise)
        return(featuresubset)
    else:
        tree=np.empty(4,dtype='object')
        featurefreq,featuresubset,cuisinedict=decomposdata(dataset)
        best_attribute,index_best=best_choice(featurefreq,featuresubset,cuisinedict,cuisine)
        tree[0]=best_attribute
        #split the dataset

        temp=np.delete(np.arange(dataset.shape[1]),index_best,0)
        datasets=np.empty(2,dtype='object')
        datasets[0]=dataset[dataset[:,index_best]<=t][:,temp]
        datasets[1]=dataset[dataset[:,index_best]>t][:,temp]
        tree[3]=t
        for i,data in enumerate(datasets):
            if(attributes.shape[0]<10-max_depth):
                tree[i+1]=majorityvote(data)
            elif (data.shape[0]==0):
                tree[i+1]=majorityvote(dataset)
            elif(gain<=mingain):
                tree[i+1]=majorityvote(dataset)
            else:
                tree[i+1]=subtree(data,attributes,max_depth)
    return(tree)

#%%
def searchdict(index,dic):
    for attribute,key in dic.iteritems():
        if key==index:
            return(attribute)
    return(1)
#%%

def best_choice(featurefreq,featuresubset,cuisinedict,cuisine):
    entropy_gain,frequency=entropy(featurefreq,featuresubset,cuisinedict)
    index=np.argmin(entropy_gain[cuisinedict[cuisine]][frequency[cuisinedict[cuisine]]>20])
    best_attribute=searchdict(index,featuresubset[cuisinedict[cuisine]])
    print(best_attribute)
    return(best_attribute,index,)

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
def split(featuresubset,cuisinedict,cuisine,ingredient):
    for i,cuis in enumerate(cusisinedict):
       del featuresubset[i][ingredient]
    return (featuresubset,cuisinedict,cuisine,ingredient)
 #%%
def majorityvote(featurefreq,featuresubset,cuisinedict,cuisine):
    #takes the majority vote of a set
    if (dataset.shape[1]<=1):
        if (np.sum(dataset)>len(dataset)/2):
            return(1.)
        else:
            return(0.)
    else:
        if (np.sum(dataset[:,-1])>len(dataset[:,-1])/2):
            return(1.)
        else:
            return(0.)