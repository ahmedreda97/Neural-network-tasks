import random
import matplotlib.pyplot as plt
import numpy as np
import math


class iris:
    def __init__(self, rate, hasBias, epoch, numLayers, numNodesInLayers, activationFunction):

        # read dataset from file and divide it into 3 classes in divide each class into training and testing sets
        text_file = open("irisData.txt", "r")
        lines = text_file.read().split('\n')
        setosa=[]
        versicolor=[]
        virginica=[]
        labelSetosa = []
        labelVersicolor = []
        labelVirginica = []
        inputData = []  # a list of list that holds the features of ALL classes
        labels = []  # a list of list that holds the label of the corresponding sample in 'inputData' ([1,0,0] -> Setosa , [0,1,0] -> VersiColor , [0,0,1] -> Virginica )

        for i in range(len(lines)):
            if i != 0:  # skip first row that contains column's name
                data = lines[i].split(',')
                if "setosa" in data[4]:
                    del data[4]
                    data = list(map(float, data))  # convert type from string to float
                    if hasBias == True:
                        data = [1] + data
                    setosa.append(data)
                    labelSetosa.append([1, 0, 0])
                elif "versicolor" in data[4]:
                    del data[4]
                    data = list(map(float, data))  # convert type from string to float
                    if hasBias == True:
                        data = [1] + data
                    versicolor.append(data)
                    labelVersicolor.append([0, 1, 0])
                elif "virginica" in data[4]:
                    del data[4]
                    data = list(map(float, data))  # convert type from string to float
                    if hasBias == True:
                        data = [1] + data
                    virginica.append(data)
                    labelVirginica.append([0, 0, 1])
        trainingSet = []
        testingSet = []
        trainingLabels =[]
        testingLabels = []

        #randomize each class individually

        combined=list(zip(setosa,labelSetosa))
        random.shuffle(combined)
        setosa[:],labelSetosa[:]=zip(*combined)
        trainingSet.extend(setosa[:40])
        trainingLabels.extend(labelSetosa[:40])
        testingSet.extend(setosa[40:])
        testingLabels.extend(labelSetosa[40:])

        combined = list(zip(versicolor,labelVersicolor))
        random.shuffle(combined)
        versicolor[:], labelVersicolor[:] = zip(*combined)
        trainingSet.extend(versicolor[:40])
        trainingLabels.extend(labelVersicolor[:40])
        testingSet.extend(versicolor[40:])
        testingLabels.extend(labelVersicolor[40:])

        combined = list(zip(virginica, labelVirginica))
        random.shuffle(combined)
        virginica[:], labelVirginica[:] = zip(*combined)
        trainingSet.extend(virginica[:40])
        trainingLabels.extend(labelVirginica[:40])
        testingSet.extend(virginica[40:])
        testingLabels.extend(labelVirginica[40:])

        combined = list(zip(trainingSet, trainingLabels))
        random.shuffle(combined)
        trainingSet[:], trainingLabels[:] = zip(*combined)


        #print(labels)
        # trainingSet = inputData[:120]
        # testingSet = inputData[120:]
        # trainingLabels = labels[:120]
        # testingLabels = labels[120:]

        # build a dictionary key: # layer , value : matrix of random weights
        layerToMatrix = {}
        numOfInput = 4
        if hasBias == True:
            numOfInput = 5
        numNodesInLayers = [numOfInput] + numNodesInLayers + [
            3]  # add at the begining the number of input nodes and append at the end the number of outputs

        for i in range(len(numNodesInLayers) - 1):
            nodesInLayer = np.random.rand(numNodesInLayers[i], numNodesInLayers[i + 1])
            layerToMatrix[i] = nodesInLayer

        # assign values to attributes
        self.trainingSet = trainingSet
        self.testingSet = testingSet
        self.trainingLabels = trainingLabels
        self.testingLabels = testingLabels
        self.rate = rate
        self.hasBias = hasBias
        self.epoch = epoch
        self.layerToMatrix = layerToMatrix
        self.activationFunction = activationFunction
        self.grads = {}

    def sigmoid(self, W):
        return 1/(1+np.exp(-W))

    def tanh(self, W):
        return np.tanh(W)

    def sigmoid_backward(self, z):
        return z * (1 - z)

    def tanh_backward(self, z):
        return 1 - np.power(np.tanh(z), 2)

    def getLoss(self, output, expected):
        return expected-output

    def activation_function_backward(self, z):
        if self.activationFunction == 0:
            return self.sigmoid_backward(z)
        else:
            return self.tanh_backward(z)

    def backward(self, loss, outputCache, weights):

        noLayers = len(weights)
        # currentLayerWeights = weights[noLayers]
        currentLayerOutputs = outputCache[noLayers]
        outputLayerGradient = loss * self.activation_function_backward(currentLayerOutputs)
        self.grads[noLayers - 1] = outputLayerGradient
        for i in reversed(range(1, noLayers)):
            prevLayerWeights = weights[i]
            currentLayerOutputs = outputCache[i]
            outputLayerGradient = self.activation_function_backward(currentLayerOutputs) * np.dot(prevLayerWeights,
                                                                                                  self.grads[i])
            self.grads[i - 1] = outputLayerGradient

    def update_params(self,input):
        for i in range(len(self.layerToMatrix)):
            gradsShape=self.grads[i].shape[0]
            inputShape=input[i].shape[0]
            self.layerToMatrix[i]=self.layerToMatrix[i]+(self.rate*np.dot(self.grads[i].reshape((gradsShape,1)),input[i].reshape((1,inputShape))).T)
            if i==4:
                print(self.layerToMatrix[i])

    def train(self):
        for i in range(int(self.epoch)):
            cache = {}  # will hold the input of each layer, with the same keys of "layerToMatrix"
            for j in range(len(self.trainingSet)):
                label = np.array(self.trainingLabels[j])
                input = np.array(self.trainingSet[j], dtype=np.float64)
                cache[0] = input

                for k in range(len(self.layerToMatrix)):
                    weight = np.array(self.layerToMatrix[k], dtype=np.float64)
                    tempOutput = np.dot(weight.T, input)  # output from layer
                    output = np.array(self.tanh(tempOutput))  # activation function using 'tanh'
                    if self.activationFunction == 0:  # sigmoid function
                        output = np.array(self.sigmoid(tempOutput))  # activation function using 'sigmoid'

                    cache[k + 1] = output
                    input = output  # add the output of the current layer as input of the next layer
                #print(input, label, j)
                currentInputLoss = self.getLoss(input, label)

                self.backward(currentInputLoss, cache, self.layerToMatrix)
                self.update_params(cache)

    def getFinalRes(self, output):
        maxNum = max(output)
        newOutput = []
        for i in output:
            if i == maxNum:
                newOutput.append(1)
            else:
                newOutput.append(0)
        return np.array(newOutput)

    def test(self):
        truePositivesC1 = 0
        falseNegativeC1LabeledC2 = 0
        falseNegativeC1LabeledC3 = 0
        truePositivesC2 = 0
        falseNegativeC2LabeledC1 = 0
        falseNegativeC2LabeledC3 = 0
        truePositivesC3 = 0
        falseNegativeC3LabeledC1 = 0
        falseNegativeC3LabeledC2 = 0
        for i in range(len(self.testingSet)):
            label = np.array(self.testingLabels[i])
            input = np.array(self.testingSet[i], dtype=np.float64)

            for k in range(len(self.layerToMatrix)):
                weight = np.array(self.layerToMatrix[k], dtype=np.float64)
                tempOutput = np.dot(weight.T, input)  # output from layer
                output = np.array(self.tanh(tempOutput))  # activation function using 'tanh'
                if self.activationFunction == 0:  # sigmoid function
                    output = np.array(self.sigmoid(tempOutput))  # activation function using 'sigmoid'

                input = output  # add the output of the current layer as input of the next layer

            result = self.getFinalRes(input)  # give the function the final output as parameter

            if np.array_equal(label, result) == True:
                if result[0] == 1:  # C1
                    truePositivesC1 += 1
                if result[1] == 1:  # C2
                    truePositivesC2 += 1
                if result[2] == 1:  # C3
                    truePositivesC3 += 1
            else:
                if result[0] == 1 and label[1] == 1:  # C1 labeled C2
                    falseNegativeC1LabeledC2 += 1
                if result[0] == 1 and label[2] == 1:  # C1 labeled C3
                    falseNegativeC1LabeledC3 += 1
                if result[1] == 1 and label[0] == 1:  # C2 labeled C1
                    falseNegativeC2LabeledC1 += 1
                if result[1] == 1 and label[2] == 1:  # C2 labeled C3
                    falseNegativeC2LabeledC3 += 1
                if result[2] == 1 and label[0] == 1:  # C3 labeled C1
                    falseNegativeC3LabeledC1 += 1
                if result[2] == 1 and label[1] == 1:  # C3 labeled C2
                    falseNegativeC3LabeledC2 += 1
        accuracy = (truePositivesC1 + truePositivesC2 + truePositivesC3) / (
                (truePositivesC1 + truePositivesC2 + truePositivesC3) + (
                    falseNegativeC1LabeledC2 + falseNegativeC1LabeledC3 + falseNegativeC2LabeledC1 + falseNegativeC2LabeledC3 + falseNegativeC3LabeledC1 + falseNegativeC3LabeledC2))
        matrix = [[truePositivesC1, falseNegativeC2LabeledC1, falseNegativeC3LabeledC1],
                  [falseNegativeC1LabeledC2, truePositivesC2, falseNegativeC3LabeledC2],
                  [falseNegativeC1LabeledC3, falseNegativeC2LabeledC3, truePositivesC3]]
        self.accuracy = accuracy
        self.matrix = matrix


# # train a model: param (rate, hasBias, epocs, numLayers, numNodesInLayers, activationFunction)
T1 = iris(0.25, True, 50, 4, [2,3,4,8], 0)
T1.train()
T1.test()
# T1.plotting()
print(T1.accuracy)
print(T1.matrix)
