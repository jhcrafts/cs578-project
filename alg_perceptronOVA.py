import algorithm

class PerceptronBinaryClassifier:
    "Perceptron Binary Classifier"
    mytargetlabel = 0
    weights = None
    bias = 0.0
    cachedweights = None
    cachedbias = 0.0
    c = 1.0
    name = None    
    
    def __init__(self, targetlabel, weightvectorsize, name):
        self.mytargetlabel = targetlabel
        self.weights = [0.0]*weightvectorsize 
        self.cachedweights = [0.0]*weightvectorsize
        self.cachedbias = 0.0
        self.c = 1.0
        self.bias = 0.0
        self.name = name
    
    def trainclassifier(self, examplevectors, maxiterations):            
        iterations = 0
        for i in range(maxiterations):      
            iterations += 1
            exampleswithoutmistake = 0
            for example in examplevectors:
                sign = 1 if example[0] == self.mytargetlabel else -1
                dotproduct = 0.0
                for j in range(1,len(example),1):
                    dotproduct += self.weights[example[j]]
                if (sign*(dotproduct + self.bias) <= 0):
                    # mistake!
                    exampleswithoutmistake = 0
                    for k in range(1,len(example),1):
                        self.weights[example[k]] += sign
                        self.cachedweights[example[k]] += (sign*self.c)
                    self.bias += sign
                    self.cachedbias += self.c*sign
                else:
                    exampleswithoutmistake += 1
                self.c += 1.0
            if (exampleswithoutmistake >= len(examplevectors)):
                break         
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.cachedweights[i]/self.c
        self.bias = self.bias - (self.cachedbias/self.c)    

    def predict(self, example):        
        dotproduct = 0.0
        for i in range(1,len(example),1):
            dotproduct += self.weights[example[i]]
        dotproduct += self.bias
        return dotproduct

    def name(self):
        return self.name

    def mytargetlabel(self):
        return mytargetlabel

class AlgPerceptronOVA(algorithm.Algorithm):
    "Implementation of Perceptron One vs. All Classifier with base class 'algorithm'"
    binaryclassifiers = list()

    def train(self, trainingexamples):
        ## create a set of all of the labels
        ## create a set of all of the ingredients
        cuisines = dict()
        cuisineindex = 0
        ingredients = dict()
        ingredientindex = 0
        examplevectors = list()
        for example in trainingexamples:            
            examplevector = list()
            try: 
                cindex = cuisines[example["cuisine"]]
            except KeyError:
                cindex = cuisineindex
                cuisines[example["cuisine"]] = cindex                
                cuisineindex += 1
            examplevector.append(cindex)                
            for ingredient in example["ingredients"]: 
                try: 
                    iindex = ingredients[ingredient]
                except KeyError:
                    iindex = ingredientindex
                    ingredients[ingredient] = iindex
                    ingredientindex += 1
                examplevector.append(iindex)
            examplevectors.append(examplevector)
        ## create a classifier for each label
        for cuisine in cuisines.keys():
            AlgPerceptronOVA.binaryclassifiers.append(PerceptronBinaryClassifier(cuisines[cuisine], len(ingredients),cuisine))
        ## train each classifier on every example
        for classifier in AlgPerceptronOVA.binaryclassifiers:
            classifier.trainclassifier(examplevectors,10)        

        correct = 0.0
        total = 0.0
        for example in examplevectors:
            total += 1
            if (example[0] == AlgPerceptronOVA.predict(self,example)):
                correct += 1
        print("Accuracy:  " + str(correct/total))

    def predict(self, example):
        results = list()        
        for classifier in AlgPerceptronOVA.binaryclassifiers:
            result = classifier.predict(example)
            results.append([result,classifier.mytargetlabel])
        sortedresults = sorted(results, cmp=resultcompare)
        return sortedresults[0][1]

    def name(self):
        return "Perceptron One vs. All"

def resultcompare(x,y):
    return -1 if (x[0] - y[0]) > 0.0 else 1 