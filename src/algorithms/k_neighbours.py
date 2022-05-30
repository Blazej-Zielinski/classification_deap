import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from src.config import with_selection
from sklearn import model_selection


def kn_parameters(number_features, icls):
    genome = list()

    # n_neighbors
    genome.append(random.randint(1, 10))

    # weights
    list_weights = ["uniform", "distance"]
    genome.append(list_weights[random.randint(0, 1)])

    # algorithm
    list_algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    genome.append(list_algorithm[random.randint(0, 3)])

    # leaf_size
    genome.append(random.randint(20, 40))

    # p
    genome.append(random.randint(2, 5))

    if with_selection:
        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

    return icls(genome)


def kn_mutation(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    match number_parameter:
        case 0:
            # n_neighbors
            individual[0] = random.randint(1, 10)
        case 1:
            # weights
            list_weights = ["uniform", "distance"]
            individual[1] = list_weights[random.randint(0, 1)]
        case 2:
            # algorithm
            list_algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
            individual[2] = list_algorithm[random.randint(0, 3)]
        case 3:
            # leaf_size
            individual[3] = random.randint(20, 40)
        case 4:
            # p
            individual[4] = random.randint(2, 5)
        case _:  # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0


def kn_parameters_fitness(y, df, number_of_attributes, individual):
    if with_selection:
        list_of_columns_to_drop = []  # lista cech do usuniecia
        for i in range(number_of_attributes, len(individual)):
            if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
                list_of_columns_to_drop.append(i - number_of_attributes)

        df = df.drop(df.columns[list_of_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = KNeighborsClassifier(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2],
                                     leaf_size=individual[3], p=individual[4])

    scores = model_selection.cross_val_score(estimator, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean(),
