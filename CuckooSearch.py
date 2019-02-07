import matplotlib.pyplot as plt
import numpy as np
import math

#INITIALIZATION
population_size = 50
max_generation = 122
_lambda = 1.5
dimension = 2
max_domain = 500
min_domain = -500
step_size_cons = 0.01
Pa = 0.25

#OBJECTIVE FUNCTION
def schwefel(array):
    sum = 0
    for x in array:
        sum = sum + x * np.sin(np.sqrt(np.abs(x)))
    fitness = 418.9829 * len(array) - sum
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
    def __init__(self):
        self.__position = np.random.rand(dimension) * (max_domain - min_domain) + min_domain
        self.__fitness = schwefel(self.__position)

    def get_position(self):
        return self.__position

    def get_fitness(self):
        return self.__fitness

    def set_position(self, position):
        self.__position = position

    def set_fitness(self, fitness):
        self.__fitness = fitness

    def abandon(self):
        # abandon some variables
        for i in range(len(self.__position)):
            p = np.random.rand()
            if p < Pa:
                self.__position[i] = np.random.rand() * (max_domain - min_domain) + min_domain
        self.__fitness = schwefel(self.__position)

def main():

    #RANDOMLY CREATING HOSTS
    cs_list = []
    for i in range(population_size):
        cs_list.append(Indivisual())


    #SORT TO GET THE BEST FITNESS
    cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

    best_fitness = cs_list[0].get_fitness()

    fig = plt.figure()

    #INITIAL POPULATION DISTRIBUTION
    ax1 = fig.add_subplot(131)
    for i in range(population_size):
        ax1.scatter([cs_list[i].get_position()[0]], [cs_list[i].get_position()[1]])
    ax1.set_title('Initial Population Distributtion')
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')

    ax3 = fig.add_subplot(133)

    t = 1
    while(best_fitness > 0.009):

        #GENERATING NEW SOLUTIONS
        for i in range(population_size):

            #CHOOSING A RANDOM CUCKOO (say i)
            i = np.random.randint(low=0, high=population_size)

            #SETTING ITS POSITION USING LEVY FLIGHT
            position = cs_list[i].get_position()+(step_size_cons*levy_flight(_lambda))

            # Simple Boundary Rule
            for i in range(len(position)):
                if position[i] > max_domain:
                    position[i] = max_domain
                if position[i] < min_domain:
                    position[i] = min_domain

            cs_list[i].set_position(position)
            cs_list[i].set_fitness(schwefel(cs_list[i].get_position()))

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
                cs_list[a].abandon()

        #RANKING THE CS LIST
        cs_list = sorted(cs_list, key=lambda ID: ID.get_fitness())

        #FIND THE CURRENT BEST
        if cs_list[0].get_fitness() < best_fitness:
            best_fitness = cs_list[0].get_fitness()

        #PRINTING SOLUTION IN EACH ITERATION
        print("iteration =", t, " best_fitness =", best_fitness)

        #FITNESS PLOTTING
        ax3.scatter(t, best_fitness)

        t += 1

    #GRAPH FOR FITNESS
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

    #SHOWING GRAPH
    plt.show()


if __name__ == "__main__":
    main()