# CS578 Project
# Francois Sanson
# Jordan Crafts


import sys
import math
import copy
import random
import json
import numpy as np
### main
### ----
### The main program loop
### You should modify this function to run your experiments
##
##def parseArgs(args):
##  """Parses arguments vector, looking for switches of the form -key {optional value}.
##  For example:
##    parseArgs([ 'main.py', '-p', 5 ]) = {'-p':5 }"""
##  args_map = {}
##  curkey = None
##  for i in xrange(1, len(args)):
##    if args[i][0] == '-':
##      args_map[args[i]] = True
##      curkey = args[i]
##    else:
##      assert curkey
##      args_map[curkey] = args[i]
##      curkey = None
##  return args_map
##
##def validateInput(args):
##    pass
##    return
##
##class Example:
##    'Class to encapsulate data examples'
##    featureValues = dict()
##
##    def __init__(self, label):
##        self.label = label
##        self.featureValues = dict()
##
##    def label(self):
##        return self.label
##
##    def addfeature(self, feature,value):
##        self.featureValues[feature] = value
##
##    def remfeature(self, feature):
##        del self.featureValues[feature]
##
##    def value(self, feature):
##        if (feature in self.featureValues.keys()):
##            return self.featureValues[feature]
##        else:
##            return None
##
##    def features(self):
##        return self.featureValues
##
def debugprint(string):
    print("DEBUG: " + str(string))


def main():
    # Read in the data file
    trainingdata = open("train.json")
    examples = json.load(trainingdata)
    featuresubset=np.empty(20,dtype=object)
    for i in range(len(featuresubset)):
        featuresubset[i]=set()
    cuisinedict={}
    featureset = set()
    labels = set()
    examplecount = 0
    for example in examples:
        examplecount += 1
        if not(example["cuisine"] in labels):
            labels.add(example["cuisine"])
            cuisinedict[example["cuisine"]]=len(labels)-1
        for ingredient in example["ingredients"]:
            featuresubset[cuisinedict[example["cuisine"]]].add(ingredient)
    for dummy in featuresubset:
        featureset=featureset.union(dummy)
    print("Training Data Properties:")
    print("Number of Examples (Recipes):            " + str(examplecount))
    print("Number of Features (Unique Ingredients): " + str(len(featureset)))
    print("Number of Labels (Cuisines):             " + str(len(labels)))

    for cuisine in labels:
        print "Size of the",cuisine,"cuisine:",len(featuresubset[cuisinedict \
        [cuisine]])


    debugprint(labels)

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

if __name__ == '__main__':
    main()



