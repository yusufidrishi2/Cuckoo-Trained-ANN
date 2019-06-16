"""
IN THIS CODE MOST OF THE THINGS WILL BE SAME AS CUCKOO SEARCH,
JUST DIFFERENCES ARE THE FOLLOWING:
1. NO. OF WEIGHTS (WHICH IS EQUAL TO NO. OF INPUTS IN AN ANN) WILL BE EQUAL TO NO. OF DIMENSIONS.
2. IN INDIVISUAL CLASS FOR EACH CUCKOO FITNESS WILL BE CALULATED BY RMSE (ROOT
   MEAN SQUARE ERROR) FUNC OF ANN, INSTEAD OF OBJECTIVE FUNC.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import confusion_matrix, precision_score,f1_score,recall_score

#INITIALIZATION
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


#OBJECTIVE FUNCTION
def RMSE(array, data, labels):  #CONVERTING THIS FUNCTION INTO RMSE FUNC OF ANN
    data_size = len(data)
    predicted = []
    for i in range(data_size):
        predicted.append(np.dot(data[i], array))
    x_actual = labels
    y_predicted = predicted
    fitness = sqrt(mean_squared_error(x_actual, y_predicted))
    return fitness

#LEVY FLIGHT
def levy_flight(Lambda):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=dimension)
    v = np.random.normal(0, sigma2, size=dimension)
    step = u / np.power(np.fabs(v), 1 / Lambda)

    return step

class Indivisual:
    def __init__(self, data, labels):
        self.__position = np.random.rand(dimension) * (max_domain - min_domain) + min_domain
        self.__fitness = RMSE(self.__position, data, labels)

    def get_position(self):
        return self.__position

    def get_fitness(self):
        return self.__fitness

    def set_position(self, position):
        self.__position = position

    def set_fitness(self, fitness):
        self.__fitness = fitness

    def abandon(self, data, labels):
        # abandon some variables
        for i in range(dimension):
            p = np.random.rand()
            if p < Pa:
                self.__position[i] = np.random.rand() * (max_domain - min_domain) + min_domain
        self.__fitness = RMSE(self.__position, data, labels)

def main():

    '''
    #WRITING RANDOM TRAIN DATA
    row = [[random.randint(0,100), random.randint(0,100), random.randint(0, 100), 0] for i in range(101)]
    row[0] = ['INPUT1', 'INPUT2', 'INPUT3', 'OUTPUT']
    for i in range(1, 101):
        row[i][3] = row[i][0]+row[i][1]+row[i][2]

    with open('train_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(101):
            writer.writerow(row[i])

    csvFile.close()
    '''
    # READING TRAIN DATA
    t_data = pd.read_csv(r'C:\Users\Kingshuk\Desktop\Activity_Dataset.csv')
    phase1 = t_data.drop(['Activity'], axis=1) #DROPING OUTPUT COLUMN FROM DATA SET
    phase2 = phase1.values                         #CONVERTING THE WHOLE DATA INTO NUMPY ARRAY IN ROW MAJOR
    train_data = phase2.tolist()             #CONVERTING THE 2D NUMPY ARRAY INTO LIST
    labels = t_data['Activity'].tolist()       #EXTRACTING LABELS FROM DATA SET
    #print(train_data)


    #RANDOMLY CREATING HOSTS
    cs_list = []
    for i in range(population_size):
        cs_list.append(Indivisual(train_data, labels))


    #SORT TO GET THE BEST FITNESS
    cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

    best_fitness = cs_list[0].get_fitness()
    best_position = cs_list[0].get_position()

    fig = plt.figure()

    '''
    #INITIAL POPULATION DISTRIBUTION
    ax1 = fig.add_subplot(131)
    for i in range(population_size):
        ax1.scatter([cs_list[i].get_position()[0]], [cs_list[i].get_position()[1]])
    ax1.set_title('Initial Population Distributtion')
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')
    '''
    #ax3 = fig.add_subplot(133)

    t = 1
    while t < max_generation:

        #GENERATING NEW SOLUTIONS
        for i in range(population_size):

            #CHOOSING A RANDOM CUCKOO (say i)
            i = np.random.randint(low=0, high=population_size)

            #SETTING ITS POSITION USING LEVY FLIGHT
            position = (cs_list[i].get_position())+(step_size_cons*levy_flight(_lambda))

            # Simple Boundary Rule
            for i in range(dimension):
                if position[i] > max_domain:
                    position[i] = max_domain
                if position[i] < min_domain:
                    position[i] = min_domain

            cs_list[i].set_position(position)
            cs_list[i].set_fitness(RMSE(cs_list[i].get_position(), train_data, labels))

            #CHOOSING A RANDOM HOST (say j)
            j = np.random.randint(0, population_size)
            while j == i:  # random id[say j] ≠ i
                j = np.random.randint(0, population_size)

            #RELAXATION
            if cs_list[j].get_fitness() > cs_list[i].get_fitness():
                cs_list[j].set_position(cs_list[i].get_position())
                cs_list[j].set_fitness(cs_list[i].get_fitness())

        #SORT (to Keep Best)
        cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

        #ABANDON SOLUTION (exclude the best)
        for a in range(1, population_size):
            r = np.random.rand()
            if (r < Pa):
                cs_list[a].abandon(train_data, labels)

        #RANKING THE CS LIST
        cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

        #FIND THE CURRENT BEST
        if cs_list[0].get_fitness() < best_fitness:
            best_fitness = cs_list[0].get_fitness()
            best_position = cs_list[0].get_position()

        #PRINTING SOLUTION IN EACH ITERATION
        print("iteration =", t, " best_fitness =", best_fitness, )

        #FITNESS ARRAY
        x.append(t)
        y.append(best_fitness)

        t += 1

    #FITNESS PLOTTING
    #ax3.plot(x, y)

    #OPTIMIZED WEIGHTS
    print("\nOptimized weights are ", end='')
    for i in range(dimension):
        print(best_position[i], end=' ')

    print()

    '''
    #WRITING RANDOM TEST DATA
    row = [[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 0] for i in range(51)]
    row[0] = ['INPUT1', 'INPUT2', 'INPUT3', 'OUTPUT']
    for i in range(1, 51):
        row[i][3] = row[i][0]+row[i][1]+row[i][2]


    with open('test_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(51):
            writer.writerow(row[i])

    csvFile.close()
    '''

    # READING TEST
    # DATA
    test_data = pd.read_csv(r'C:\Users\Kingshuk\Desktop\testcsv.csv')
    td = test_data.drop(['Activity'], axis=1).values.tolist()
    test_labels = test_data['Activity'].tolist()
    # print(test_data['INPUT1'][0])


    print("RMSE of of final ANN is ", RMSE(best_position, td, test_labels))



    #WRITE PREDICTED OBSERVATION DATA
    row = [['ACCELEROMETERX', 'ACCELEROMETERY', 'ACCELEROMETERZ', 'Activity', 'PREDICTED OUTPUT']]

    for i in range(61):
        predicted_output = np.dot(td[i], best_position)
        row.append([td[i][0], td[i][1], td[i][2], test_data['Activity'][i], int(round(predicted_output))])

    with open('predicted_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(61):
            writer.writerow(row[i])

    csvFile.close()
    predict = pd.read_csv('predicted_data.csv')
    Predicted = predict['PREDICTED OUTPUT'].tolist()
    Actual = predict['Activity'].tolist()

    """Confusion Matrix"""

    cm = confusion_matrix(Actual, Predicted, labels = [1, 2, 3])
    sum = cm.sum()
    print(cm)
    print("Accuarcy is ", (cm.trace())/sum)
    cm = precision_score(Actual,Predicted)
    print("Precision score is ", cm)
    cm = recall_score(Actual, Predicted)
    print("Recall score is ", cm)
    cm = f1_score(Actual, Predicted)
    print("f1 score is ", cm)

    '''
    #GRAPH FOR FITNESS
    ax3.set_title('Fitness Curve')
    ax3.set_xlabel('x-axis')
    ax3.set_ylabel('y-axis')
    
    
    '''

    '''
    # FINAL POPULATION DISTRIBUTION
    ax2 = fig.add_subplot(132)
    for i in range(population_size):
        ax2.scatter([cs_list[i].get_position()[0]], [cs_list[i].get_position()[1]])
    ax2.set_title('Final Population Distributtion after '+str(t)+' iterations')
    ax2.set_xlabel('x-axis')
    ax2.set_ylabel('y-axis')

    #SHOWING GRAPH
    plt.show()
    '''

if __name__ == "__main__":
    main()
