# -*- coding: utf-8 -*-
# COM ELITISMO --- TOP_Nparticles


# GENERALIZADO:
    # VARIAVEIS DE ENTRADA ILIMITADAS AUTOMATICO --- ATRAVES DE LEN(BOUNDS), BASTA ADICIONAR AS BOUNDS DESSA VARIAVEL ADICIONAL À LISTA DE BOUNDS ---> CRIADA NO FIM
    # FUNCOES OBJETIVO ILIMITADAS --- ATRAVES DA LISTA OBJETIVOS ---> CRIADA NO FIM, SAO EM FUNCAO DO VETOR QUE ENGLOBA TODAS AS VARIAVEIS DE ENTRADA
    # RESTRICOES ILIMITADAS --- ATRAVES DA LISTA RESTRICOES NA FUNCAO DE CALCULO DO SCORE ---> CRIADA MANUALMENTE CONSOANTE O QUE FAÇA SENTIDO


import random
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import os
from copy import deepcopy
import csv
import multiprocessing
import time
import copy
import Optimization_CB_v2_prod

def run_trial(trials):
    def save_iteration_data(folder_name, iteration, xline_all_values, yline_all_values, X_all_values, o1l, o2l, o3l, inc):
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
            writer.writerow(['xline_all_values', 'yline_all_values', 'X_all_values', 'obj1', 'obj2', 'obj3', 'inc'])

            # Assuming all lists are of the same length
            for xline, yline, X, o1, o2, o3 in zip(xline_all_values, yline_all_values, X_all_values, o1l, o2l, o3l):
                writer.writerow([xline, yline, X, o1,o2,o3, inc])

    # ONDE DIZ AQUIIIIIIIIIIII SÃO OS SITIOS NECESSÁRIOS A ALTERAR CASO SE QUEIRA TER MAIS FUNÇOES (F1, F2, F3, F4, ...), PARA ALEM DE ADICIONAR ESSAS TAIS FUNCOES NOS INITS
    # O PLOT É QUE NÃO DÁ PARA FAZER AQUI ACHO EU. EM BAIXO ESTÁ APENAS PARA 2 FUNCOES
    # BASTA FAZER CTRL+F DE "AQUIIIIII" PARA VER OS SITIOS DE MUDANÇA. SAO 16 LOCAIS


    # A DIMENSAO DAS FUNCOES ESTA AUTOMATIZADA. O NUMERO DE FUNCOES PARA CONSTITUIR OS "PAIRS" É QUE NAO


    class Particle:

        def __init__(self, bounds): #, dim): AQUIIIIIIIIIIIIIIIIIIIII

            #self.dim=len(bounds) # MELHOT QUE TER XMAX E XMIN
            # bounds=[[xmin,xmax],[ymin,ymax],...]  ESTÁ GENERALIZADO!!!!!!

            # ESTE SELF.POSITION NAO VAI ESTAR A SER ALTERADO

            self.position=[0]*len(bounds)
            self.velocity=[0]*len(bounds)
            for i in range(len(bounds)):
                self.position[i] = (random.uniform(bounds[i][0], bounds[i][1]))
                self.velocity[i] = random.uniform(-1, 1)

            self.best_position = self.position
            self.best_fitness = -float('inf') # NOTE-SE QUE COM O SCORE TEMOS UM PROBLEMA DE MAXIMIZAÇÃO
            self.fitness = -float('inf')

            # NOVO

            # AQUIIIIIIII

            #    self.pairs=[[self.f1(self.position), self.f2(self.position)]]  # AQUIIIIIIIIIIIIIIIIIIIII. PARA UMA PARTICULA, A CADA (X,Y) QUE VÁ TENDO VAI CORRESPONDER UM VALOR DE F1 E DE F2 QUE FORMAM UM PAR. ESTES PARES TODOS ESTÃO A SER ARMAZENADOS AQUI

            # self.best_pair = None # MELHOR PAR PARA CADA PARTICULA. CUIDADO COM O NONE ..... SEM USO

            #self.history = [[self.position], [self.fitness], [self.velocity], []]   #[pos, fit/score, vel, iteracao dessa particula]

            #    self.history = [self.position]

    def update_position(position, velocity, bounds, iiii = True):

        new_position = [0]*len(bounds)

        if iiii:
            for i in range(len(position)):
                new_position[i] = (max(bounds[i][0], min(bounds[i][1],position[i] + velocity[i])))

        else:
            for i in range(len(bounds)):
                new_position[i] = (random.uniform(bounds[i][0], bounds[i][1]))

        return list(new_position)

    def update_velocity(position, velocity, particle_base, global_best_position, w, c1, c2, best):

        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        new_velocity=[0]*len(bounds)

        phi = c1+c2
        K = 2/abs(2-phi-np.sqrt(phi**2-4*phi))
        for i in range(len(velocity)):
            new_velocity[i] = K*(w * velocity[i] + c1 * r1 * (best[i] - position[i]) + c2 * r2 * (global_best_position[i] - position[i]))

        return list(new_velocity)

    class ParticleData:
        def __init__(self):
            self.best = None
            self.best_f = None

    class PSO:
        def __init__(self, function, bounds, w, c1, c2, n_particles, n_iterations, funcoes_objetivo, folder, problem_ind): #AQUIIIIIIIIIIIIIIII
            self.function = function
            self.bounds = bounds
            self.n_particles = n_particles
            self.n_iterations = n_iterations
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.global_best_pareto = []
            self.folder = folder
            self.problem_ind = problem_ind

            self.global_best_position = []
            self.global_best_fitness = -float('inf')
            self.inc = 0
            # NOVO

            self.global_best_pair = []

            # self.f1=f1
            # self.f2=f2
            # AQUIIIIIIIIII
            self.funcoes_objetivo = funcoes_objetivo


            self.all_pairs_global= []
            self.dict_pair_rank = dict()


            self.history_all = []
            self.index_dict = dict()

            # GERAÇÃO DE PARTICULAS INICIAIS

            #self.particles = [Particle(bounds, f1, f2) for i in range(n_particles)] # AQUIIIIIIIIIIIIIIII

            for i in range(n_particles):
                particle=Particle(bounds)

                # OBJECTIVES = [f1(vetor), f2(vetor), f3(vetor), ...]
                obj1, obj2, obj3, obj4 = Optimization_CB_v2_prod.opt_calc_prod(particle.position)
                
                objectives = [obj1, obj2, obj4]
                #objectives = [obj1*obj2, obj2/obj3, obj4]
                
                self.history_all.append([i, particle.position, particle.velocity, objectives, particle.fitness, [obj1, obj2, obj3]])
                if i not in self.index_dict:
                    self.index_dict[i] = ParticleData()

                self.index_dict[i].best = list(particle.position)
                self.index_dict[i].best_f = particle.fitness

                # a particula inicial faz parte do historico pq assim conseguimos aceder a esta e atualizar os particles bests para a velocidade e assim e desta maneira nao faz mal se houverem varias particulas com a mm particula base

        def search(self):
            for i in range(self.n_iterations):
                print(i)
                self.history_all = self.function(self.n_iterations, self.n_particles, self.history_all)

                pareto = [k[3] for k in self.history_all if k[6] == 1]
                conter_1 = 0
                for kk in pareto:
                    if kk in self.global_best_pareto: conter_1 += 1
                if conter_1 == len(pareto):
                    self.inc += 1
                else:
                    self.inc = 0

                self.global_best_pareto = pareto

                for j in range(len(self.history_all)):
                    if self.history_all[j][4] > self.global_best_fitness:
                        self.global_best_position = list(self.history_all[j][1])
                        self.global_best_fitness = self.history_all[j][4]
                        self.global_best_pair = list(self.history_all[j][3])

                    jj = self.history_all[j][0]
                    if self.history_all[j][4] > self.index_dict[jj].best_f:
                        self.index_dict[jj].best_f = self.history_all[j][4]
                        self.index_dict[jj].best = list(self.history_all[j][1])


                # FAZER O PLOT A CADA ITERAÇÃO PARA VER A CURVA A APROXIMAR DO ZERO,0 ----- NAO PARECE ESTAR A CONVERGIR PARA 0

                # NO PLOT PROJETAMOS EM 2D OS 2 OBJETIVOS QUE QUEREMOS, BASTA ALTERAR XLINE E YLINE

                xline_all_values=[]
                yline_all_values=[]
                X_all_values = []
                o1l = []
                o2l = []
                o3l = []

                for elem in self.history_all:
                    rank = elem[6]
                    objectives = elem[3]
                    cp = elem[9]
                    X = elem[1]
                    obj_elem = elem[5]
                    print(objectives)
                    if rank == 1 and objectives[2]:
                        xline_all_values.append(objectives[0])
                        yline_all_values.append(objectives[1])
                        X_all_values.append(X)
                        o1l.append(obj_elem[0])
                        o2l.append(obj_elem[1])
                        o3l.append(obj_elem[2])

                #plt.plot(xline_all_values, yline_all_values, 'o')

                save_iteration_data(self.folder, i, xline_all_values, yline_all_values, X_all_values, o1l, o2l, o3l, self.inc)

                #plt.xlim([0.001, 2])
                #plt.ylim([0, 2])

                xline_all_values=[]
                yline_all_values=[]

                for rank, objectives, cp, obj_elem in zip([k[6] for k in self.history_all],[k[3] for k in self.history_all], [k[9] for k in self.history_all], [k[5] for k in self.history_all]):
                    if rank == 1 and objectives[2]:
                        xline_all_values.append(objectives[0])
                        yline_all_values.append(objectives[1])

                plt.plot(xline_all_values, yline_all_values, 'x')

                # Save the plot in the GVRP folder with the iteration count in the filename
                folder_name = self.folder

                # Example usage
                filename = os.path.join(folder_name, f'Iteration_{i}.png')
                plt.savefig(filename)
                plt.close()  # Close the plot to free up memory

                for indk in range(len(self.history_all)):
                    self.history_all[indk] = self.history_all[indk][0:6]

                # TIRAR PONTOS QUE POR ALGUMA RAZAO TÊM A MM POSIÇÃO

                historico_2 = []

                positions_all = []

                for particle in self.history_all:
                    if particle[1] not in positions_all:
                        positions_all.append(particle[1])
                        historico_2.append(particle)

                self.history_all = historico_2

                # TOP_Nparticles

                all_fitnesses = []

                for particle in self.history_all:
                    all_fitnesses.append(particle[4])

                fitness_aux = np.flip(sorted(all_fitnesses))

                historico_3 = []

                for fit in fitness_aux:
                    for particle in self.history_all:
                        if particle[4] == fit:
                            historico_3.append(particle)
                            break

                self.history_all = historico_3[0:self.n_particles]


                # UPDATE DO TOP30

                if i == 1:
                    n = len(self.history_all)       # SE SE REMOVEREM ITENS POR ESTAREM DUPLICADOS NA PRIMEIRA ITERAÇÃO PODEMOS TER MENOS PARTICLES QUE AS 30
                else:
                    n = self.n_particles

                rdms = np.random.choice(n, self.inc, replace=False)

                for k in range(n):          # CRIAÇÃO DAS NOVAS PARTICULAS
                    velocity = update_velocity(self.history_all[k][1], self.history_all[k][2], self.history_all[k][0], self.global_best_position, self.w, self.c1, self.c2, self.index_dict[self.history_all[k][0]].best)
                    position = update_position(self.history_all[k][1], velocity, self.bounds, self.inc == 0 or k in rdms)

                    #pair = [self.f1(position), self.f2(position)] = objectives

                    # OBJECTIVES = [f1(vetor), f2(vetor), f3(vetor), ...]
                    obj1, obj2, obj3, obj4 = Optimization_CB_v2_prod.opt_calc_prod(position)
                    
                    objectives = [obj1, obj2, obj4]
                    #objectives = [obj1*obj2, obj2/obj3, obj4]

                    fitness = None

                    self.history_all.append([self.history_all[k][0], position, velocity, objectives, fitness,  [obj1, obj2, obj3]])

            return self.global_best_position, self.global_best_pair

        def remove_duplicates(self, history_all):
            unique_positions = set()
            unique_particles = []

            for particle in history_all:
                position = tuple(particle[1])  # Assuming particle[1] is the position

                if position not in unique_positions:
                    unique_positions.add(position)
                    unique_particles.append(particle)

            return unique_particles

        def get_top_particles(self, n):
            """
            Retrieve the top N particles based on fitness.
            """
            # Create a heap with the negative fitness (for max-heap) and particle index
            heap = [(-particle[4], idx) for idx, particle in enumerate(self.history_all)]
            heapq.heapify(heap)

            top_particles = []
            for _ in range(min(n, len(heap))):
                _, idx = heapq.heappop(heap)
                top_particles.append(self.history_all[idx])

            return top_particles

        def pareto_front_pair_rank(self):
            # Using a set to handle unique pairs efficiently
            unique_pairs = {tuple(particle[3]) for particle in self.history_all}

            # Initialize an empty dictionary for pair ranks
            self.dict_pair_rank = {}

            # Process each pair to determine its rank
            while unique_pairs:
                pareto_front = set()
                for pair in unique_pairs:
                    if all(not self.is_dominated(pair, other) for other in unique_pairs - {pair}):
                        pareto_front.add(pair)

                # Assign rank and remove the current Pareto front from consideration in the next round
                rank = len(self.dict_pair_rank) + 1
                for pair in pareto_front:
                    self.dict_pair_rank[pair] = rank
                unique_pairs -= pareto_front

            return self.dict_pair_rank

        def is_dominated(self, pair, other):
            # Check if 'pair' is dominated by 'other'
            return all(x <= y for x, y in zip(other, pair)) and any(x < y for x, y in zip(other, pair))


    # -------- FUNÇOES SCORE E AUXILIARES -----------------------------------------------------------------------------------



    def score(n_iterations, n_particles, history):
    #-----------------------------RANK-----------------------------------------

        Nu = sum([len(k)==6 for k in history])

        n = 1
        pairs_bk = [(ind, k[3][0:2], k[3][2]) for ind, k in enumerate(history)]

        while n <= n_iterations * n_particles and Nu > 0:
            Nu = sum([len(k)==6 for k in history])
            pareto=[]
            list_to_remove = []
            for ind, pair, const in pairs_bk:
                dominant = True
                for ind2, pair_2, const2 in pairs_bk:
                    is_dominated = True
                    for i in range(len(pair)):
                        is_dominated = is_dominated and pair_2[i] < pair[i] and const and const2
                    if is_dominated:
                        dominant = False
                        break

                if dominant:
                    pareto.append(pair)
                    history[ind].append(n)
                    Nu = sum([len(k)==6 for k in history])
                    list_to_remove.append((ind, pair, const))

            for elee in list_to_remove:
                pairs_bk.remove(elee)

            n += 1

        list11 = [k[6] for k in history]

        mmm = max(list11)


        list2 = [tuple(k[3][0:2]) for k in history]
        for indk, k in enumerate(history):
            point = deepcopy(k[3][0:2])
            count = 1 if not k[3][2] else 0
            history[indk].append(mmm - k[6])

            #-------------------------CDA--------------------------------------------------------------

            #rank_1 = rank_point
            #dict_pairs = dict_pair_rank
            #dict_pairs = dict_rank(point, list_points, n_iterations, n_particles)   # PRECISO DO DICIONARIO PORQUE SO COMPARO AS DISTANCIAS NUM MESMO RANK


            # POINT = (f1(pos), f2(pos), f3(pos), ...)

            same_rank=[]

            for rank_2,pair_2 in zip(list11,list2):
                if rank_2 == k[6] and pair_2 != tuple(point):
                    same_rank.append(pair_2)

            dist_min=math.inf          # NESTE CONTEXTO FAZ SENTIDO FAZER ISTO?? CASO NO ÚLTIMO RANK SÓ HAJA UM PAIR ESTA DIST É 0??
            dist_max = -math.inf
            dist = 1
            for i in range(len(point)):
                try:
                    min_val = np.log10(np.min([pair[i] for pair in same_rank]))
                    max_val = np.log10(np.max([pair[i] for pair in same_rank]))

                    if np.log10(point[i]) < min_val:
                        dist *= (min_val - np.log10(point[i]))/((max_val - min_val) + (min_val - np.log10(point[i])))
                        if dist > 1:
                            print(1, dist)

                    elif np.log10(point[i]) > max_val:
                        dist *= (np.log10(point[i]) - max_val) / ((max_val - min_val) + (np.log10(point[i]) - max_val))
                        if dist > 1:
                            print(2, dist)

                    else:
                        btm = np.max([np.log10(pair[i]) - np.log10(point[i]) for pair in same_rank if np.log10(pair[i]) - np.log10(point[i]) < 0])
                        top = np.min([np.log10(pair[i]) - np.log10(point[i]) for pair in same_rank if
                                      np.log10(pair[i]) - np.log10(point[i]) > 0])
                        dist *= (top - btm)/(max_val - min_val)
                        if dist > 1:
                            print(3, dist)
                except:
                    dist -= 1
            #print(cda_1)

            history[indk].append(dist)


            #------------------RESTRICOES--------------------------------------

            history[indk].append(count)
            history[indk][4] = history[indk][7] + history[indk][8] - n_iterations * n_particles * history[indk][9]




    #----------------FINAL-RETURN---------------------------------

        return history

    #------ FUNCOES PARA A RESTRICAO <---> APENDICE DA PARTE RESTRICOES DO SCORE ------------------

    # def f1(point):
    #     return point[0] + point[1]

    # def f2(point):
    #     return 3 * point[0] - point[1]**2

    # def f3(point):
    #     return point[0]**2 + point[1] - 1

    def f4(point):
        return point[0]




    # ---------------  FUNÇÔES TESTE OBJETIVO --------------------------------------------------------------------------------------------------------------------

    # A ENTRADA DESTAS FUNCOES É UM VETOR, NESTE CASO É O POSITION, MAS SERÁ O "D" NO CASO DAS ENGRENAGENS

    def sphere_function(position):
        return np.sum(np.power(position[0], 2))


    def auckley(position):
        x=position[0]
        y=position[1]
        return -20 * np.exp(-0.2 * np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20


    def beale(position):
        x=position[0]
        y=position[1]
        return ((1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2)


    def matyas(position):
        x=position[0]
        y=position[1]
        return 0.26*(x**2 + y**2) - 0.48 * x * y


    def rastrigin(position):
        A=10
        return A*2 + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in position])

    def module(f1, f2):
        return np.sqrt(f1**2+f2**2)





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

    '''kkk_bol = False

    if not kkk_bol:


        # Example usage
        file_path = 'PG1.txt'  # Replace with your actual file path
        dd, ll, dd_total = import_arrays_from_file(file_path)

    else:
        dd = []
        dd_total = 0

        ll = []
        dd_min = 10
        dd_max = 1000

        ll_min = -50
        ll_max = 50

        for _ in range(N_total_nodes):
            daux = np.random.uniform(dd_min, dd_max)
            dd.append(daux)
            ll.append(np.random.uniform(ll_min, ll_max))
            dd_total += daux

    '''
    # ----------  TESTE  -----------------------------------------------------------------------------------------------------------------------

    # USAR AUCKLEY E MATYAS NAO SIGNIFICA MT, APENAS TENDE PARA 0 POIS AMBAS AS FUNCOES TÊM O MINIMO, 0, NAS COORDENADAS (0,0)
    # BEALE nao tende para (0,0) por isso é mais interessante para testar com as outras, pois dá um range de solucoes, frente de pareto

    
    bounds = [[0.1, 10], [0.1, 10], [0.1, 10], [0.1, 10], [0.1, 10], [0, 1], [0, 1]] # [thickness liner; thickness flute; wavelength; height

    for problem_ind in range(1):
        print(problem_ind)

        pso = PSO(function=score, bounds=bounds, n_particles=50, n_iterations=100, w=0.7, c1=2.05, c2=2.05,
                  funcoes_objetivo=True,
                  folder="prob_CB_prod_fix_v5_WC/EPSO" + str(problem_ind) + "/EPSO_" + str(trials),
                  problem_ind=problem_ind)

        best_position, best_pair = pso.search()

        # pso.graph()

        pareto = pso.pareto_front_pair_rank()
        # Print the best position and fitness AND THE PAIRS + RANK

        print('Best position = {}'.format(best_position))
        print('Best pair = {}'.format(best_pair))
        print('Pareto solutions = ' + str(pareto))


if __name__ == '__main__':
    nnn = 4
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=4)
    pool.map(run_trial, range(nnn))
    pool.close()
    pool.join()
    #run_trial(1)
    print("TRIALS COMPLETED: " + str(nnn))