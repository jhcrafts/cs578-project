import algorithm
import numpy as np
from copy import deepcopy

class Hyperparameters:
    pass

class BinAdaboost:
    """Binary Adaboost"""

    name="binadaboost"
    cuisine=None
    ElmClassi=None
    def __init__(self,cuisine):
        self.tree=np.array(4,dtype='object')
        self.cuisine=cuisine

    def trainclassifier(self,dataset,Nclassi):
        """
        Adaboost
        inputs:
        -dataset to split
        outputs:
        -np array of the nodes with their weights
        """
        if (dataset==[]):
            #make sure the dataset is not empty
            return(0.)
        else:
            distribution=np.zeros(len(dataset))+1./len(dataset)
            featurefreq,featuresubset,cuisinedict=self.decomposdata(dataset)
            self.ElmClassi=[None]*Nclassi
            for i in range(Nclassi):
                bestIngredient=[None]*2
                bestIngredient[1]=0.
                for ingredient in featuresubset[cuisinedict[self.cuisine]]:
                    accuracy=0
                    for j,recipe in enumerate(dataset):
                        if (ingredient in recipe['ingredients'] and recipe['cuisine']==self.cuisine):
                            accuracy+=distribution[j]
                        elif (not(ingredient in recipe['ingredients']) and not(recipe['cuisine']==self.cuisine)):
                            accuracy+=distribution[j]
                    if accuracy>bestIngredient[1]:
                        bestIngredient[0]=ingredient
                        bestIngredient[1]=accuracy
                ElmClassi[i]=bestIngredient
                alpha=0.5*log(accuracy/1-accuracy)
                ElmClassi[i][1]=alpha
                # Update the distribution
                distribution=distribution*exp(-alpha)
                for j,recipe in enumerate(dataset):
                    if (ingredient in recipe['ingredients'] and recipe['cuisine']==self.cuisine):
                        distribution[j]=distribution[j]*exp(2*alpha)
                    elif (not(ingredient in recipe['ingredients']) and not(recipe['cuisine']==self.cuisine)):
                        distribution[j]=distribution[j]*exp(2*alpha)

                # Normalize
                distribution=distribution/np.sum(distribution)
            return(0)
    def predict_one(self,recipe,neveringredients=[]):
        score=0.
        for Elm in self.ElmClassi:
            if Elm[0] in recipe['ingredient']:
                score+=Elm[1]
            else:
                score-=Elm[1]

        return(score)

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

class AlgAdaboost(algorithm.Algorithm):
    "Implementation of Adaboost Classifier with base class 'algorithm'"
    binaryclassifiers=list()
    cuisines=dict()
    labels=dict()
    name='Adaboost'

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
        for cuisine in AlgAdaboost.cuisines.keys():
            AlgAdaboost.binaryclassifiers.append(BinAdaboost(AlgAdaboost.cuisines[cuisine]))
        ## train each classifier on every example
        for classifier in AlgAdaboost.binaryclassifiers:
            classifier.trainclassifier(trainingexamples,100)

    def predict(self, example):
        result=np.zeros(20)
        for i,Adaboost  in enumerate(self.binaryclassifiers):
            result[i]=Adaboost.predict_one(example)
            return(np.argmax(result))


    def name(self):
        return "Adaboost"

    def label(self, fmt_label):
        return(fmt_label)

    def description(self):
        pass

    def labelfromexample(self,fmt_example):
        return(fmt_example['cuisine'])

