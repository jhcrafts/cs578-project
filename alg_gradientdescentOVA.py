import algorithm

class GradientDescentBinaryClassifier:
    "Gradient Descent Binary Classifier"
    mytargetlabel = 0
    weights = None
    bias = 0.0
    name = None
    lmbd = 0.01
    stepsize = 0.01   
    
    def __init__(self, targetlabel, weightvectorsize, name):
        self.mytargetlabel = targetlabel
        self.weights = [0.0]*weightvectorsize 
        self.bias = 0.0
        self.name = name
   
    def trainclassifier(self, examplevectors, maxiterations):        
        regularizer = GradientDescentBinaryClassifier.l1reg #GradientDescentBinaryClassifier.l2reg                        
        trainingaccuracy = list()
        nonzerolossexamples = list()
        for i in range(maxiterations):
            g = [0.0]*len(self.weights);
            gb = 0.0
            n = 0.0
            total = 0.0
            correct = 0.0
            for example in examplevectors:
                sign = 1 if example[0] == self.mytargetlabel else -1
                dotproduct = 0.0
                for index in example[1]:
                    dotproduct += self.weights[index]                
                if (sign*(dotproduct + self.bias)) <= 1.0:
                    n += 1
                    for index in example[1]:                
                        g[index] += sign
                    gb += sign    
                #Calculate Training Accuracy Inline - This saves computation effort          
                if (sign*(dotproduct + self.bias)) > 0.0:
                    correct += 1  
                total += 1                            
            trainingaccuracy.append(correct/total)
            nonzerolossexamples.append(n/total)
            for k in range(len(g)):
                g[k] = g[k] - regularizer(self,self.weights[k], GradientDescentBinaryClassifier.lmbd)
                self.weights[k] = self.weights[k] + GradientDescentBinaryClassifier.stepsize*g[k]
            self.bias = self.bias + GradientDescentBinaryClassifier.stepsize*gb        
        print(str(self.mytargetlabel) + "," + str(trainingaccuracy) + "," + str(nonzerolossexamples))        
        return    
    
    def l1reg(self,weight,lmbd):
        if (weight > 0.0):
            return lmbd
        elif (weight < 0.0):
            return -1.0*lmbd
        else:
            return 0

    def l2reg(self,weight,lmbd):
        return weight*lmbd 

    def predict(self, example):        
        dotproduct = 0.0
        for index in example[1]:
            dotproduct += self.weights[index]
        dotproduct += self.bias
        return dotproduct

    def name(self):
        return self.name

    def mytargetlabel(self):
        return mytargetlabel

class AlgGradientDescentOVA(algorithm.Algorithm):
    "Implementation of Gradient Descent One vs. All Classifier with base class 'algorithm'"
    binaryclassifiers = list()
    cuisines = dict()
    labels = dict()    
    ingredients = dict()
    
          
    def extractfeatures(self,trainingexamples):
        ingredientindex = 0 
        cuisineindex = 0
        for example in trainingexamples:
            try: 
                cindex = AlgGradientDescentOVA.cuisines[example["cuisine"]]                
            except KeyError:
                cindex = cuisineindex
                AlgGradientDescentOVA.cuisines[example["cuisine"]] = cindex                
                AlgGradientDescentOVA.labels[cindex] = example["cuisine"]
                cuisineindex += 1                            
            for ingredient in example["ingredients"]: 
                try: 
                    iindex = AlgGradientDescentOVA.ingredients[ingredient]
                except KeyError:
                    iindex = ingredientindex
                    AlgGradientDescentOVA.ingredients[ingredient] = iindex
                    ingredientindex += 1

    def exportfeatures(self):
        pass

    def loadfeatures(self,features):
        pass
        
    def formatexample(self,example):
        try:
            label = AlgGradientDescentOVA.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(AlgGradientDescentOVA.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector]

    def train(self, trainingexamples):        
        ## create a classifier for each label
        for cuisine in AlgGradientDescentOVA.cuisines.keys():
            AlgGradientDescentOVA.binaryclassifiers.append(GradientDescentBinaryClassifier(AlgGradientDescentOVA.cuisines[cuisine], len(AlgGradientDescentOVA.ingredients),cuisine))
        ## train each classifier on every example
        for classifier in AlgGradientDescentOVA.binaryclassifiers:
            classifier.trainclassifier(trainingexamples,1000)        

    def predict(self, example):
        results = list()        
        for classifier in AlgGradientDescentOVA.binaryclassifiers:
            result = classifier.predict(example)
            results.append([result,classifier.mytargetlabel])
        sortedresults = sorted(results, cmp=resultcompare)
        return sortedresults[0][1]

    def name(self):
        return "Gradient Descent One vs. All"

    def label(self, fmt_label):
        return AlgGradientDescentOVA.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])


def resultcompare(x,y):
    return -1 if (x[0] - y[0]) > 0.0 else 1 