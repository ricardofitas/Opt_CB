import random
import numpy as np
from deap import base, creator, tools, algorithms
import os
import csv

####### TEST CASE STUDY: G-VRP

X = [1]
i = 1
flag = True
p = 0.4

a = 1
b = 100

X.append(np.random.uniform(a,b))

def generate_speed_cost(speed): return 0.05 * speed + 1

def z12(X, dd, ll, Q):
    iii = 2
    for i in range(len(X)):
        if i % 2 == 1:
            iii += X[i]
            X[i] = iii

    fact_aux = (N_total_nodes-1)/iii

    for i in range(len(X)):
        if i % 2 == 1: X[i] = int(np.ceil(X[i]*fact_aux))
    indp = 1
    ind = 1
    velp = X[0]
    sm = 0
    vh_load = Q
    z2 = 0
    cck = True
    if vh_load < 0 or vh_load > 2*Q: cck = False
    for ii, elem in enumerate(X[1:]):
        if ii % 2 == 1:
            dist = 0
            if dist * vh_load * generate_speed_cost(velp) < 0:
                print(indp)
                print(ind)
                print(dist)
                print(vh_load)

            for k in range(indp, ind):
                dist += dd[k-1]
            if vh_load < 0 or vh_load > 2*Q: cck = False
            if dist != 0:
                sm += (velp / dist) * abs(ll[ind - 1])
            kkkkk = float("inf") if vh_load < 0 else dist * vh_load * generate_speed_cost(velp)
            z2 += kkkkk

            if vh_load <0 and cck: print(cck)
            velp = elem
            vh_load += ll[ind-1]
            indp = ind

            if vh_load < 0 or vh_load > 2*Q: cck = False

        else:
            ind = elem

    if indp < N_total_nodes:
        dist = 0
        for k in range(indp, N_total_nodes):
            dist += dd[k-1]
        if vh_load < 0 or vh_load > 2*Q: cck = False
        sm += (dist/velp) * (1/abs(ll[N_total_nodes-1]))
        kkkkk = 10**10 if vh_load < 0 else dist * vh_load * generate_speed_cost(velp)
        z2 += kkkkk
        if vh_load < 0 and cck: print(cck)
        if vh_load < 0 or vh_load > 2*Q: cck = False



    return sm, z2*10**-9, cck


def cc(point, Q):
    xx = z12(point, dd, ll, Q)
    return -1 if not xx[2] else 1

#list1=[f1,f2,f3,f4]
list1=[cc]


def z1(position,Q):
    qq = z12(position, dd, ll, Q)[0]
    return qq

def z2(position,Q):
    qq = z12(position, dd, ll, Q)[1]
    return qq

objetivos = [z1, z2, cc]

# Create Fitness and Individual classes
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Attribute generator
def random_attribute(i):
    lower, upper = bounds[i]
    if i % 2 == 1:  # Discrete variables
        # Ensure the bounds are integers
        lower, upper = int(lower), int(upper)
        return random.randint(lower, upper)
    else:  # Continuous variables
        return random.uniform(lower, upper)

# Individual generator
def create_individual():
    return [random_attribute(i) for i in range(2 * N - 1)]

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def evaluate(individual):
    return z1(individual,Q), z2(individual,Q)

# Constraint function
def feasible(individual):
    return cc(individual, Q) == 1


def evaluate_individual(individual, Q):
    objective1 = z1(individual, Q)  # Replace with your objective function
    objective2 = z2(individual, Q)  # Replace with your objective function
    is_feasible = cc(individual, Q)    # Replace with your constraint check function
    return (objective1, objective2), is_feasible


# Register functions in toolbox
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("feasible", feasible)


def save_iteration_data(folder_name, iteration, xline_all_values, yline_all_values, X_all_values, inc):
    """
    Saves the iteration data to a CSV file.

    Parameters:
    - folder_name (str): The name of the folder to save the files in.
    - iteration (int): The current iteration number.
    - xline_all_values (list): List of xline values.
    - yline_all_values (list): List of yline values.
    - X_all_values (list of lists): List of X values (design vectors).
    - inc: The 'self.inc' variable.
    """

    # Ensure the directory exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the filename
    filename = os.path.join(folder_name, f'Iteration_{iteration}.csv')

    # Save data to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['xline_all_values', 'yline_all_values', 'X_all_values', 'inc'])

        # Assuming all lists are of the same length
        for xline, yline, X in zip(xline_all_values, yline_all_values, X_all_values):
            writer.writerow([xline, yline, X, inc])

def main(fn, trials):
    random.seed(64)
    NGEN = 100   # Number of generations
    MU = 50      # Number of individuals to select for the next generation
    LAMBDA = 100 # Number of children to produce at each generation
    CXPB = 0.7   # Crossover probability
    MUTPB = 0.2  # Mutation probability

    Q = 200 if fn == 1 else 2000

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Directory to save Pareto fronts
    directory = "GVRP3_" + str(fn) + "/NSGA_" + str(trials)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Begin the evolutionary process
    for gen in range(NGEN):
        # Select and clone the next generation individuals
        offspring = algorithms.varOr(pop, toolbox, lambda_=LAMBDA, cxpb=CXPB, mutpb=MUTPB)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            (obj1, obj2), is_feasible = toolbox.evaluate(ind,Q)
            if is_feasible:
                ind.fitness.values = (obj1, obj2)
            else:
                ind.fitness.values = (10**10, 10**10)  # Assuming two objectives

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select the next generation population
        pop[:] = toolbox.select(pop + offspring, MU)

        # Update the statistics with the new population
        record = stats.compile(pop)
        print(f"Generation {gen}: {record}")

        # Save the Pareto front after each generation
        pareto_front = [(ind.fitness.values) for ind in hof]
        save_pareto_front(gen, pareto_front, directory)
        # Extract data for saving
        obj1_values, obj2_values = zip(*pareto_front)
        X_all_values = [ind for ind in pop]  # Assuming 'ind' represents the design vector
        inc = 0

        # Call the function to save iteration data
        save_iteration_data(directory, gen, obj1_values, obj2_values, X_all_values, inc)

    return pop, stats, hof

import matplotlib.pyplot as plt


def save_pareto_front(iteration, pareto_front, directory):
    obj1_values, obj2_values = zip(*pareto_front)
    plt.figure()
    plt.scatter(obj1_values, obj2_values)
    plt.title(f'Pareto Front at Iteration {iteration}')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')

    plt.savefig(f"{directory}/Pareto_Front_{iteration}.png")
    plt.close()


def import_arrays_from_file(file_path):
    """
    Imports the dd and ll arrays from a text file with multiline arrays.

    Parameters:
    - file_path (str): Path to the file containing the arrays.

    Returns:
    - tuple: (dd array, ll array, dd_total)
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line numbers where dd and ll arrays start
    dd_start = lines.index("dd array:\n") + 1
    ll_start = lines.index("ll array:\n") + 1
    dd_lines = []
    ll_lines = []

    # Extract dd array lines
    for line in lines[dd_start:]:
        if line.startswith('ll array:'):
            break
        dd_lines.append(line.strip())

    # Extract ll array lines
    for line in lines[ll_start:]:
        if line.strip() == '':
            break
        ll_lines.append(line.strip())

    # Convert lines to strings and then to arrays
    dd_str = ' '.join(dd_lines).replace('\n', '').replace('[', '').replace(']', '')
    ll_str = ' '.join(ll_lines).replace('\n', '').replace('[', '').replace(']', '')
    dd = np.fromstring(dd_str, sep=',')
    ll = np.fromstring(ll_str, sep=',')
    dd_total = np.sum(dd)

    return dd, ll, dd_total

nnn = 100
for trials in range(0, nnn):

    print("TRIALS COMPLETED: " + str(trials))

    for problem_ind in range(1, 3):

        # Example usage
        file_path = 'PG' + str(problem_ind) + '.txt'  # Replace with your actual file path
        dd, ll, dd_total = import_arrays_from_file(file_path)

        if problem_ind == 1:
            N = 10
            N_total_nodes = 1000
        else:
            N = 100
            N_total_nodes = 1000000

        k = N_total_nodes/(N+1)

        list_aux = []

        for _ in range(N):

            random_value = random.random()*k
            i += random_value + 1
            list_aux.append(i)
            i += random_value + 1

        fact_aux = (N_total_nodes-1)/i

        X = []

        for kk in list_aux:
            X.append(int(np.ceil(kk*fact_aux)))
            X.append(np.random.uniform(a,b))

        bounds = []

        for ii in range(2*N - 1):
            if ii % 2 == 1: bounds.append([1, k])
            else: bounds.append([a, b])

        if __name__ == "__main__":
            pop, stats, hof = main(problem_ind, trials)

            # Print the Pareto front
            print("Pareto Front:")
            pareto_front = []
            for ind in hof:
                print(ind.fitness.values)
                pareto_front.append(ind.fitness.values)

            # Unzip the objectives
            obj1_values, obj2_values = zip(*pareto_front)
