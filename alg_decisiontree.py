import algorithm
import numpy as np
from copy import deepcopy
class Hyperparameters:
    pass

class BinDecisionTree:
    "Binary Decision Tree"
    tree=np.array(3,dtype='object')
    name="bindecisiontree"
    cuisine=None

    def __init__(self,cuisine):
        self.tree=np.array(4,dtype='object')
        self.cuisine=cuisine

    def trainclassifier(self,dataset,maxdepth,savedname=""):
        if savedname!="":
            try:
                self.tree=np.load(savedname)
            except IOError:
                print("could not load previous tree, jus making a new one")
                self.tree=self.subtree(dataset,self.cuisine,max_depth=0,mingain=-1)
        else:
            self.tree=self.subtree(dataset,self.cuisine,max_depth=0,mingain=-1)

    def subtree(self,dataset,cuisine,pastingredients=[],max_depth=0,mingain=-1):
        """
        Fundamental dexision tree learning function
        inputs:
        -dataset to split
        outputs:
        -np array of the node with the attribute we split on
        -separated datasets
        """
        if (dataset==[]):
            #make sure the dataset is not empty
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
                ingredient,gain=self.best_choice(dataset,pastingredients,cuisine)
                if ingredient==1:
                        return(0)
                # choice of the ingredient
                tree[0]=ingredient
                #split the dataset
                datasetwith,datasetwithout=self.split(dataset,ingredient)
                pastingredients+=[ingredient]
                datasets=np.empty(2,dtype='object')
                datasets[0]=datasetwith
                datasets[1]=datasetwithout
                for i,data in enumerate(datasets):
                    #stopping condition
                    if(max_depth>3):
                        tree[i+1]=self.majorityvote(data,cuisine)
                    elif (data==[]):
                        tree[i+1]=self.majorityvote(dataset,cuisine)
                    elif(gain<=mingain):
                        tree[i+1]=self.majorityvote(dataset,cuisine)
                    else:
                        #extra iteration
                        tree[i+1]=self.subtree(data,cuisine,pastingredients,max_depth+1)
        return(tree)

    def searchdict(self,index,dic):
        for attribute,key in dic.iteritems():
            if key==index:
                return(attribute)
        return(1)

    def best_choice(self,dataset,pastingredients,cuisine):
        """
        Chooses the best ingredient
        """
        featurefreq,featuresubset,cuisinedict=self.decomposdata(dataset) # make the data readable
        bestgain=31415.
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
            entropy_gen,entro=self.entropy(dataset,ingredient,cuisine)
            if (bestgain>entropy_gen):
                bestgain=entropy_gen
                best_ingredient=ingredient
        print(best_ingredient)
        return(best_ingredient,bestgain)

    def entropy(self,dataset,attribute,cuisine):
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

    def split(self,dataset,ingred):
        datasetwith=list()
        datasetwithout=deepcopy(dataset)
        a=[1]
        for recipe in dataset:
           if ingred in recipe['ingredients']:
               a[0]=recipe
               datasetwith+=a
               datasetwithout.remove(recipe)
        return (datasetwith,datasetwithout)

    def majorityvote(self,dataset,cuisine):
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
    def predict_one(self,recipe,neveringredients=[]):
        currentnode=self.tree
        depth=0
        while not(isinstance(currentnode,float) or isinstance(currentnode,int) ):
            attribute=currentnode[0]
            if attribute in recipe['ingredients']:
                currentnode=currentnode[1]
            else:
                currentnode=currentnode[2]
            depth+=1
        return(currentnode,depth)

    def decomposdata(self,examples):
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
            featuresubset[i]=self.listodict(list(subset))
        for example in examples:
            cuisine=cuisinedict[example["cuisine"]]
            for ingredient in example["ingredients"]:
                featurefreq[cuisine][featuresubset[cuisine][ingredient]]+=1.
        return(featurefreq,featuresubset,cuisinedict)


    def listodict(self,liste):
        dic=dict()
        for i,word in enumerate(liste):
            dic[word]=i
        return(dic)

class AlgDecisionTree(algorithm.Algorithm):
    "Implementation of Decision Tree Classifier with base class 'algorithm'"
    binaryclassifiers=list()
    cuisines=dict()
    labels=dict()
    name='desiciontree'

    def extractfeatures(self,trainingexamples):
        cuisines=set()
        for example in trainingexamples:
            cuisines.add(example['cuisine'])
        for cuisine in cuisines:
            self.labels[cuisine]=cuisine
            self.cuisines[cuisine]=cuisine

    def formatexample(self,example):
        return(example)

    def train(self, trainingexamples):
        ## create a classifier for each cuisine
        for cuisine in AlgDecisionTree.cuisines.keys():
            AlgDecisionTree.binaryclassifiers.append(BinDecisionTree(AlgDecisionTree.cuisines[cuisine]))
        ## train each classifier on every example
        for classifier in AlgDecisionTree.binaryclassifiers:
            classifier.trainclassifier(trainingexamples,maxdepth=0)

    def predict(self, example):
        result=np.zeros((2,20))
        for i, tree in enumerate(self.binaryclassifiers):
            result[:,i]=tree.predict_one(example)
        if (np.sum(result[0,:])==1): # only one tree claimed it
            return(np.argmax(result[0,:]))
        elif (np.sum(result[0,:])==0): # nobody clamined
            #print('Nobody claimed it, return italian')
            return(17)
        elif (np.sum(result[0,:])>1.): # multiple claims -> the most direct lineage wins (most superficial node)
            return(np.argmin(result[1,result[0,:]==1]))


    def name(self):
        return "Decision Tree"

    def label(self, fmt_label):
        return(fmt_label)

    def description(self):
        pass

    def labelfromexample(self,fmt_example):
        return(fmt_example['cuisine'])

