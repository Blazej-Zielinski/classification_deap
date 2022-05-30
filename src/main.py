import multiprocessing
import sys
import os
import csv

from matplotlib import pyplot as plt

from utlls import prepare_data
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import *
from src.algorithms.svc import *
from src.algorithms.random_forest import *
from src.algorithms.decision_tree import *
from src.algorithms.logistic_regression import *
from src.algorithms.k_neighbours import *
from config import *
import random
import time

best_members = []
avg_arr = []
std_arr = []

sub_folder = sys.argv[0][:-7] + f'output/{config["classifier"]}{"_with_selection" if with_selection else ""}/'
if not os.path.isdir(sub_folder):
    os.mkdir(sub_folder)

output_file_path = sub_folder + 'output.csv'
plot_path_fv = sub_folder + 'fitness_value.png'
plot_path_avg = sub_folder + 'average.png'
plot_path_stdev = sub_folder + 'standard_deviation.png'


def write_to_file(best, avg, std):
    header = ['Epoch', 'Individual', 'Fitness_value', 'Average', 'Standard_deviation']

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for epoch in range(len(best)):
            writer.writerow([
                epoch + 1,
                best[epoch][0],
                best[epoch][1],
                avg[epoch],
                std[epoch]
            ])


def create_plots(best, avg, std):
    x = range(1, config['number_iteration'] + 1)

    plt.xlabel('Epoch')
    plt.ylabel('Fitness value')
    plt.title("Chart of fitness value by epoch")

    plt.plot(x, [el[1] for el in best])
    plt.savefig(plot_path_fv)
    plt.cla()

    plt.xlabel('Epoch')
    plt.ylabel('Average')
    plt.title("Chart of average by epoch")

    plt.plot(x, [a for a in avg])
    plt.savefig(plot_path_avg)
    plt.cla()

    plt.xlabel('Epoch')
    plt.ylabel('Standard deviation')
    plt.title("Chart of standard deviation by epoch")

    plt.plot(x, [s for s in std])
    plt.savefig(plot_path_stdev)
    plt.cla()


if __name__ == '__main__':
    toolbox = base.Toolbox()
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)

    x, y = prepare_data("data/SouthGermanCredit.csv")
    number_of_attributes = len(x.columns)

    toolbox = base.Toolbox()
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(x)
    clf = None

    match config['classifier']:
        case Classifier.SVC:
            toolbox.register('mutate', SVC_mutation)
            toolbox.register('individual', SVC_parameters, number_of_attributes, creator.Individual)
            toolbox.register('evaluate', SVC_parameters_fitness, y, x, number_of_attributes)
            clf = SVC()
        case Classifier.RANDOM_FOREST:
            toolbox.register('mutate', rf_mutation)
            toolbox.register('individual', rf_parameters, number_of_attributes, creator.Individual)
            toolbox.register('evaluate', rf_parameters_fitness, y, x, number_of_attributes)
            clf = RandomForestClassifier()
        case Classifier.DECISION_TREE:
            toolbox.register('mutate', dt_mutation)
            toolbox.register('individual', dt_parameters, number_of_attributes, creator.Individual)
            toolbox.register('evaluate', dt_parameters_fitness, y, x, number_of_attributes)
            clf = DecisionTreeClassifier()
        case Classifier.K_NEIGHBOURS:
            toolbox.register('mutate', kn_mutation)
            toolbox.register('individual', kn_parameters, number_of_attributes, creator.Individual)
            toolbox.register('evaluate', kn_parameters_fitness, y, x, number_of_attributes)
            clf = KNeighborsClassifier()
        case _:
            toolbox.register('mutate', LR_mutation)
            toolbox.register('individual', LR_parameters, number_of_attributes, creator.Individual)
            toolbox.register('evaluate', LR_parameters_fitness, y, x, number_of_attributes)
            clf = LogisticRegression()

    scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f'\nWynik klasyfikatora {clf.__class__.__name__} : {scores.mean()}\n')

    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    match config['select']:
        case Selection.TOURNAMENT:
            toolbox.register("select", tools.selTournament, tournsize=config['tournament_size'], k=config['select_size'])
        case Selection.WORST:
            toolbox.register("select", tools.selWorst, k=config['select_size'])
        case Selection.RANDOM:
            toolbox.register("select", tools.selRandom, k=config['select_size'])
        case Selection.ROULETTE_WHEEL:
            toolbox.register("select", tools.selRoulette, k=config['select_size'])
        case Selection.STOCH_UNIVERSAL_SAMPLING:
            toolbox.register("select", tools.selStochasticUniversalSampling, k=config['select_size'])
        case _:
            toolbox.register("select", tools.selBest, k=config['select_size'])

    pop = toolbox.population(n=config['size_population'])

    # fitness_values = list(map(toolbox.evaluate, pop))
    fitness_values = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness_values):
        ind.fitness.values = fit

    g = 0
    start_time = time.time()
    while g < config['number_iteration']:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        best_individual = tools.selBest(pop, 1)[0]
        offspring.remove(best_individual)
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < config['probability_mutate']:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitness_values = map(toolbox.evaluate, invalid_ind)
        fitness_values = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness_values):
            ind.fitness.values = fit

        # print("  Evaluated %i individuals" % len(invalid_ind))
        offspring.append(best_individual)
        pop[:] = offspring
        # Gather all the fitness_values in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]

        best_members.append((best_ind, best_ind.fitness.values[0]))
        avg_arr.append(mean)
        std_arr.append(std)
        # for mem in pop:
        #     print("%s, Value: %s" % (mem, mem.fitness.values[0]))
        print("Best individual is %s, Value: %s" % (best_ind, best_ind.fitness.values[0]))
    execution_time = round(time.time() - start_time, 2)
    print(f"-- End of evolution, execution time: {execution_time} --")
    print("Best individual is %s, Value: %s" % (best_ind, best_ind.fitness.values[0]))
    write_to_file(best_members, avg_arr, std_arr)
    create_plots(best_members, avg_arr, std_arr)
