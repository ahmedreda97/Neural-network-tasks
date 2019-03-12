import random
import matplotlib.pyplot as plt
import numpy as np


class iris:
    def __init__(self, F1, F2, C1, C2, rate, hasBias, epoch, mseThres=0):

        # read dataset from file and divide it into 3 classes in divide each class into training and testing sets
        text_file = open("irisData.txt", "r")
        lines = text_file.read().split('\n')

        # 3 Lists for the three classes each one is a list of lists
        setosa = []
        versicolor = []
        virginica = []

        for i in range(len(lines)):
            if i != 0:  # skip first row that contains column's name
                data = lines[i].split(',')
                if "setosa" in data[4]:
                    del data[4]
                    setosa.append(data)
                elif "versicolor" in data[4]:
                    del data[4]
                    versicolor.append(data)
                elif "virginica" in data[4]:
                    del data[4]
                    virginica.append(data)

        trainSetSetosa = setosa[:30]
        testSetSetosa = setosa[30:]

        trainSetVersicolor = versicolor[:30]
        testSetVersicolor = versicolor[30:]

        trainSetVirginica = virginica[:30]
        testSetVirginica = virginica[30:]

        # put input in a single variable
        # Features: 0 -> x1 , 1 -> x2 , 2 -> x3 , 3 -> x4
        # Classes: setosa -> 0 , versicolor -> 1 , virginica -> 2
        inputData = []  # a list of lists each inner list consists of 3 elements (x, y, label)
        testSet = []  # have the same format of input var, will be used in testing phase

        # lists for the whole dataset for the 2 choices of user
        firstChoice = []
        secondChoice = []
        gotFirstClass = False  # to check if the first class is '0' or not

        # Add data to input && user's choices
        if C1 == 0 or C2 == 0:  # setosa
            for i in range(len(trainSetSetosa)):
                x = float(trainSetSetosa[i][F1])
                y = float(trainSetSetosa[i][F2])
                inputData.append([x, y, -1])
                firstChoice.append([x, y, -1])
                gotFirstClass = True
        if C1 == 1 or C2 == 1:  # versicolor
            for i in range(len(trainSetVersicolor)):
                x = float(trainSetVersicolor[i][F1])
                y = float(trainSetVersicolor[i][F2])
                if gotFirstClass == False:
                    inputData.append([x, y, -1])
                    firstChoice.append([x, y, -1])
                else:
                    inputData.append([x, y, 1])
                    secondChoice.append([x, y, 1])
        if C1 == 2 or C2 == 2:  # virginica
            for i in range(len(trainSetVirginica)):
                x = float(trainSetVirginica[i][F1])
                y = float(trainSetVirginica[i][F2])
                inputData.append([x, y, 1])
                secondChoice.append([x, y, 1])

        gotFirstClass = False  # to check if the first class is '0' or not
        # Add data to test
        if C1 == 0 or C2 == 0:  # setosa
            for i in range(len(testSetSetosa)):
                x = float(testSetSetosa[i][F1])
                y = float(testSetSetosa[i][F2])
                testSet.append([x, y, -1])
                gotFirstClass = True
        if C1 == 1 or C2 == 1:  # versicolor
            for i in range(len(testSetVersicolor)):
                x = float(testSetVersicolor[i][F1])
                y = float(testSetVersicolor[i][F2])
                if gotFirstClass == False:
                    testSet.append([x, y, -1])
                else:
                    testSet.append([x, y, 1])
        if C1 == 2 or C2 == 2:  # virginica
            for i in range(len(testSetVirginica)):
                x = float(testSetVirginica[i][F1])
                y = float(testSetVirginica[i][F2])
                testSet.append([x, y, 1])

        # assign values to attributes
        self.input = inputData
        self.testSet = testSet
        self.epoch = epoch
        self.rate = rate
        self.hasBias = hasBias
        self.firstChoice = firstChoice
        self.secondChoice = secondChoice
        self.Setosa = setosa
        self.Virginica = virginica
        self.Versicolor = versicolor
        self.F1 = F1
        self.F2 = F2
        self.mseThres = mseThres
        # assign flowe's name to label to be used in the plotting function
        if C1 == 0:
            self.label1 = "Setosa"
        if C1 == 1:
            self.label1 = "Versicolor"
        if C1 == 2:
            self.label1 = "Virginica"
        if C2 == 0:
            self.label2 = "Setosa"
        if C2 == 1:
            self.label2 = "Versicolor"
        if C2 == 2:
            self.label2 = "Virginica"

    def signum(self, W):
        ReturnList = []
        for i in W:
            if i < 0:
                ReturnList.append(-1)
            else:
                ReturnList.append(1)

        return ReturnList


    def plotting(self):
        XFirst = []
        YFirst = []
        XSecond = []
        YSecond = []

        MinXC1 = 100
        MinXC2 = 100
        MaxXC1 = 0
        MaxXC2 = 0

        for i in range(len(self.firstChoice)):
            XFirst.append(self.firstChoice[i][0])
            YFirst.append(self.firstChoice[i][1])
            MinXC1 = min(MinXC1, self.firstChoice[i][0])
            MaxXC1 = max(MaxXC1, self.firstChoice[i][1])

        for i in range(len(self.secondChoice)):
            XSecond.append(self.secondChoice[i][0])
            YSecond.append(self.secondChoice[i][1])
            MinXC2 = min(MinXC2, self.secondChoice[i][0])
            MaxXC2 = max(MaxXC2, self.secondChoice[i][1])

        # compute values to draw the line
        if self.hasBias:
            b = self.weight[0]
            w1 = self.weight[1]
            w2 = self.weight[2]
        else:
            b = 0
            w1 = self.weight[0]
            w2 = self.weight[1]

        xMin = min(MinXC1, MinXC2)
        xMax = max(MaxXC1, MaxXC2)
        yMin = (-(w1 * xMin) - b) / w2
        yMax = (-(w1 * xMax) - b) / w2

        # Define the known points
        x = [xMin, xMax]
        y = [yMin, yMax]

        # Calculate the coefficients. This line answers the initial question.
        coefficients = np.polyfit(x, y, 1)

        # Print the findings
        # print
        # 'a =', coefficients[0]
        # print
        # 'b =', coefficients[1]

        # Let's compute the values of the line...
        coefficients = np.squeeze(coefficients)
        polynomial = np.poly1d(coefficients)
        x_axis = np.linspace(0, 10, 10)
        y_axis = polynomial(x_axis)

        plt.figure(self.label1 + " VS. " + self.label2)
        plt.scatter(XFirst, YFirst)
        plt.scatter(XSecond, YSecond)
        plt.plot(x_axis, y_axis)
        plt.plot(x[0], y[0], 'go')
        plt.plot(x[1], y[1], 'go')
        plt.grid(True)
        plt.xlabel(self.label1)
        plt.ylabel(self.label2)
        plt.show()

    def train(self):
        if self.hasBias == True:
            weight = np.random.rand(3, 1)
        else:
            weight = np.random.rand(2, 1)
        for i in range(int(self.epoch)):
            for j in range(len(self.input)):
                x = self.input[j][0]  # add x
                y = self.input[j][1]  # add y
                label = self.input[j][2]

                if self.hasBias == True:
                    input = np.array([1, x, y], dtype=np.float64)
                else:
                    input = np.array([x, y], dtype=np.float64)

                fw1 = np.dot(weight.T, input)  # input to hidden layer 1
                Output = np.array(self.signum(fw1))  # activation function for hidden layer 1
                if not Output == label:  # False prediction
                    loss = label - Output
                    # recompute weights
                    tempInput = input.reshape(weight.shape[0], 1)
                    weight = weight + (float(self.rate) * loss * tempInput)
            print("---- Epoch Number " + str(i) + " Loss = " + str(round(np.sum(loss), 3)) + " ----")

        self.weight = weight

    def trainAda(self):
        """Task 2 function
            Train single layer network
            with Adaline algorithm
            """
        if self.hasBias == True:
            weight = np.random.rand(3, 1)
        else:
            weight = np.random.rand(2, 1)
        for i in range(int(self.epoch)):
            for j in range(len(self.input)):
                x = self.input[j][0]  # add x
                y = self.input[j][1]  # add y
                label = self.input[j][2]

                if self.hasBias == True:
                    input = np.array([1, x, y], dtype=np.float64)
                else:
                    input = np.array([x, y], dtype=np.float64)

                fw1 = np.dot(weight.T, input)  # input to hidden layer 1
                if fw1 == [float('inf')]:
                    print('Weight error, please reduce learning rate')
                    return
                loss = label - fw1
                # recompute weights
                tempInput = input.reshape(weight.shape[0], 1)
                weight = weight + (float(self.rate) * loss * tempInput)
            mse = 0
            for j in range(len(self.input)):    #loop on the input data to calculate the mse with the updated weights
                x = self.input[j][0]  # add x
                y = self.input[j][1]  # add y
                label = self.input[j][2]

                if self.hasBias:
                    input = np.array([1, x, y], dtype=np.float64)
                else:
                    input = np.array([x, y], dtype=np.float64)
                fw1 = np.dot(weight.T, input)
                loss = label - np.array(fw1)
                mse = mse + (loss * loss)
            mse = mse / len(self.input)
            print("---- Epoch Number "+str(i)+" Loss = "+str(round(np.sum(loss),3))+" MSE = "+str(mse)+" ----")

            self.weight = weight
            if mse < float(self.mseThres):
                return

    def test(self):
        truePositivesC1 = 0
        falseNegativeC1 = 0
        truePositivesC2 = 0
        falseNegativeC2 = 0
        for i in range(len(self.testSet)):
            x = self.testSet[i][0]  # add x
            y = self.testSet[i][1]  # add y
            label = self.testSet[i][2]

            if self.hasBias == True:
                input = np.array([1, x, y], dtype=np.float64)
            else:
                input = np.array([x, y], dtype=np.float64)

            fw1 = np.dot(self.weight.T, input)  # input to hidden layer 1
            Output = np.array(self.signum(fw1))  # activation function for hidden layer 1
            if Output == label:
                if label == -1:
                    # print(1)
                    truePositivesC1 += 1
                else:
                    # print(2)
                    truePositivesC2 += 1
            else:
                if label == -1:
                    # print(3)
                    falseNegativeC1 += 1
                else:
                    # print(4)
                    falseNegativeC2 += 1

        # print(truePositivesC1)
        # print(truePositivesC2)
        # print(falseNegativeC1)
        # print(falseNegativeC2)
        accuracy = (truePositivesC1 + truePositivesC2) / (
                    (truePositivesC1 + truePositivesC2) + (falseNegativeC1 + falseNegativeC2))
        matrix = [[truePositivesC1, falseNegativeC1], [falseNegativeC2, truePositivesC2]]
        self.accuracy = accuracy
        self.matrix = matrix

    def plottingAll(self):
        XFirst = []
        YFirst = []
        XSecond = []
        YSecond = []
        XThird = []
        YThird = []

        for i in range(len(self.Setosa)):
            XFirst.append(self.Setosa[i][self.F1])
            YFirst.append(self.Setosa[i][self.F2])
            XSecond.append(self.Versicolor[i][self.F1])
            YSecond.append(self.Versicolor[i][self.F2])
            XThird.append(self.Virginica[i][self.F1])
            YThird.append(self.Virginica[i][self.F2])

        plt.figure("Compare All")
        plt.scatter(XFirst, YFirst)
        plt.scatter(XSecond, YSecond)
        plt.scatter(XThird, YThird)
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

# # train a model: param (F1, F2, C1, C2, rate, hasBias, epoch)
# T1 = iris(3,0,0,1,0.3,False, 10)
# T1.train()
# T1.test()
# T1.plotting()
# print(T1.accuracy)
# print(T1.matrix)
