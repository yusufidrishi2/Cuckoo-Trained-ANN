"""
INTRODUCTION:

THIS IS A MODEL WHICH IS USING AS A REGRESSION MODEL.
THIS MODEL USES TRAIN DATA SET OF ADDITION, IN WHICH IT ADDS THREE VALUES AND GIVE OUTPUTS.
WE ARE USING A RATIO OF 70:30 FOR TRAIN DATASET AND TEST DATASET RESPECTIVELY.
AT THE END, WHEN PROGRAM RAN SUCCESSFULLY, THE OUTPUT YOU WILL GET WILL BE BASED ON THE GIVEN DATA SET,YOU WILL GET YOUR:-
I)   DESIRED WEIGHTS
II)  ROOT MEAN SQUARE ERROR (RMSE) VALUE
III) ACCURACY (between 0 to 1) 
IV)  THREE GRAPHS SHOWING INITIAL POPULATION, FINAL POPULATION AND FITNESS(RMSE) CURVE.
"""


"""  
IN THIS CODE MOST OF THE THINGS WILL BE SAME AS CUCKOO SEARCH,
JUST DIFFERENCES ARE THE FOLLOWING:
1. NO. OF WEIGHTS (WHICH IS EQUAL TO NO. OF INPUTS IN AN ANN) WILL BE EQUAL TO NO. OF DIMENSIONS.
2. IN Individual CLASS FOR EACH CUCKOO FITNESS WILL BE CALULATED BY rmse (ROOT
   MEAN SQUARE ERROR) FUNC OF ANN, INSTEAD OF OBJECTIVE FUNC.
3. ONE ACCURACY FUNCTION IS USED TO CALCULATE ACCURACY.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
import random
from math import sqrt
import sklearn.metrics as metrics

# INITIALIZATION
population_size = 50
max_generation = 1000
_lambda = 1.5
dimension = 3
max_domain = 500
min_domain = -500
step_size_cons = 0.01
Pa = 0.25
x = []
y = []

# CALCULATING ACCURACY USING r2 SCORE (since this model is currently getting used as a regression model)
def accuracy(array, data, labels, data_size):
    y_predicted = []
    for i in range(data_size):
        y_predicted.append(np.dot(data[i], array))
    x_actual = labels
    acc = metrics.r2_score(x_actual, y_predicted)
    return acc

# OBJECTIVE FUNCTION
def rmse(array, data, labels, data_size):  # CONVERTING THIS FUNCTION INTO rmse FUNC OF ANN
    y_predicted = []
    for i in range(data_size):
        y_predicted.append(np.dot(data[i], array))
    x_actual = labels
    fitness = sqrt(metrics.mean_squared_error(x_actual, y_predicted))
    return fitness

# LEVY FLIGHT
def levy_flight(Lambda):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=dimension)
    v = np.random.normal(0, sigma2, size=dimension)
    step = u / np.power(np.fabs(v), 1 / Lambda)

    return step

class Individual:
    def __init__(self, data, labels, data_size):
        self.__position = np.random.rand(dimension) * (max_domain - min_domain) + min_domain
        self.__fitness = rmse(self.__position, data, labels, data_size)

    def get_position(self):
        return self.__position

    def get_fitness(self):
        return self.__fitness

    def set_position(self, position):
        self.__position = position

    def set_fitness(self, fitness):
        self.__fitness = fitness

    def abandon(self, data, labels, data_size):
        # abandon some variables
        for i in range(dimension):
            p = np.random.rand()
            if p < Pa:
                self.__position[i] = np.random.rand() * (max_domain - min_domain) + min_domain
        self.__fitness = rmse(self.__position, data, labels, data_size)

def main():

    """
    #WRITING RANDOM TRAIN DATA
    train_data_size = 100
    row = [[random.randint(0,100), random.randint(0,100), random.randint(0, 100), 0] for i in range(train_data_size+1)]
    row[0] = ['INPUT1', 'INPUT2', 'INPUT3', 'OUTPUT']
    for i in range(1, train_data_size+1):
        row[i][3] = row[i][0]+row[i][1]+row[i][2]

    with open('train_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(train_data_size+1):
            writer.writerow(row[i])

    csvFile.close()
    """

    # READING TRAIN DATA
    t_data = pd.read_csv('train_data.csv')
    phase1 = t_data.drop(['OUTPUT'], axis=1) # DROPING OUTPUT COLUMN FROM DATA SET
    phase2 = phase1.values                   # CONVERTING THE WHOLE DATA INTO NUMPY ARRAY IN ROW MAJOR
    train_data = phase2.tolist()             # CONVERTING THE 2D NUMPY ARRAY INTO LIST
    labels = t_data['OUTPUT'].tolist()       # EXTRACTING LABELS FROM DATA SET
    train_data_size = len(t_data)
    # print(train_data)

    # RANDOMLY CREATING HOSTS
    cs_list = []
    for i in range(population_size):
        cs_list.append(Individual(train_data, labels, train_data_size))


    # SORT TO GET THE BEST FITNESS
    cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

    best_fitness = cs_list[0].get_fitness()
    best_position = cs_list[0].get_position()

    fig = plt.figure()

    # INITIAL POPULATION DISTRIBUTION
    ax1 = fig.add_subplot(131)
    for i in range(population_size):
        ax1.scatter([cs_list[i].get_position()[0]], [cs_list[i].get_position()[1]])
    ax1.set_title('Initial Population Distributtion')
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')

    ax3 = fig.add_subplot(133)

    t = 1
    while t < max_generation:

        # GENERATING NEW SOLUTIONS
        for i in range(population_size):

            # CHOOSING A RANDOM CUCKOO (say i)
            i = np.random.randint(low=0, high=population_size)

            # SETTING ITS POSITION USING LEVY FLIGHT
            position = (cs_list[i].get_position())+(step_size_cons*levy_flight(_lambda))

            # Simple Boundary Rule
            for i in range(dimension):
                if position[i] > max_domain:
                    position[i] = max_domain
                if position[i] < min_domain:
                    position[i] = min_domain

            cs_list[i].set_position(position)
            cs_list[i].set_fitness(rmse(cs_list[i].get_position(), train_data, labels, train_data_size))

            # CHOOSING A RANDOM HOST (say j)
            j = np.random.randint(0, population_size)
            while j == i:  # random id[say j] ≠ i
                j = np.random.randint(0, population_size)

            # RELAXATION
            if cs_list[j].get_fitness() > cs_list[i].get_fitness():
                cs_list[j].set_position(cs_list[i].get_position())
                cs_list[j].set_fitness(cs_list[i].get_fitness())

        # SORT (to Keep Best)
        cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

        # ABANDON SOLUTION (exclude the best)
        for a in range(1, population_size):
            r = np.random.rand()
            if (r < Pa):
                cs_list[a].abandon(train_data, labels, train_data_size)

        # RANKING THE CS LIST
        cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

        # FIND THE CURRENT BEST
        if cs_list[0].get_fitness() < best_fitness:
            best_fitness = cs_list[0].get_fitness()
            best_position = cs_list[0].get_position()

        # PRINTING SOLUTION IN EACH ITERATION
        print("iteration =", t, " best_fitness =", best_fitness)

        # FITNESS ARRAY
        x.append(t)
        y.append(best_fitness)

        t += 1

    # FITNESS PLOTTING
    ax3.plot(x, y)

    # OPTIMIZED WEIGHTS
    print("\nOptimized weights are ", *best_position)

    """
    #WRITING RANDOM TEST DATA
    test_data_size = 30
    row = [[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 0] for i in range(test_data_size+1)]
    row[0] = ['INPUT1', 'INPUT2', 'INPUT3', 'OUTPUT']
    for i in range(1, test_data_size+1):
        row[i][3] = row[i][0]+row[i][1]+row[i][2]

    with open('test_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(test_data_size+1):
            writer.writerow(row[i])

    csvFile.close()
    """

    # READING TEST DATA
    test_data = pd.read_csv('test_data.csv')
    td = test_data.drop(['OUTPUT'], axis=1).values.tolist()
    test_labels = test_data['OUTPUT'].tolist()
    test_data_size = len(test_data)
    # print(test_data['INPUT1'][0])


    print("RMSE of of final ANN is ", rmse(best_position, td, test_labels, test_data_size))
    print("Accuracy of final ANN is ", accuracy(best_position, td, test_labels, test_data_size))

    # WRITE PREDICTED OBSERVATION DATA
    row = [['INPUT1', 'INPUT2', 'INPUT3', 'ACTUAL OUTPUT', 'PREDICTED OUTPUT']]

    for i in range(test_data_size):
        predicted_output = np.dot(td[i], best_position)
        row.append([td[i][0], td[i][1], td[i][2], test_data['OUTPUT'][i], predicted_output])

    with open('predicted_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(test_data_size+1):
            writer.writerow(row[i])

    csvFile.close()

    # GRAPH FOR FITNESS
    ax3.set_title('Fitness Curve')
    ax3.set_xlabel('x-axis')
    ax3.set_ylabel('y-axis')

    # FINAL POPULATION DISTRIBUTION
    ax2 = fig.add_subplot(132)
    for i in range(population_size):
        ax2.scatter([cs_list[i].get_position()[0]], [cs_list[i].get_position()[1]])
    ax2.set_title('Final Population Distributtion after '+str(t)+' iterations')
    ax2.set_xlabel('x-axis')
    ax2.set_ylabel('y-axis')

    # SHOWING GRAPH
    plt.show()


if __name__ == "__main__":
    main()