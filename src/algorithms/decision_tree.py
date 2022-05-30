import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from src.config import with_selection
from sklearn import model_selection


def dt_parameters(number_features, icls):
    genome = list()

    # criterion
    list_criterion = ["gini", "entropy", "log_loss"]
    genome.append(list_criterion[random.randint(0, 2)])

    # splitter
    list_splitter = ["best", "random"]
    genome.append(list_splitter[random.randint(0, 1)])

    # max_depth
    genome.append(random.randint(2, 8))

    # min_samples_split
    genome.append(random.uniform(0.01, 1))

    # min_samples_leaf
    genome.append(random.uniform(0.01, 5))

    if with_selection:
        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

    return icls(genome)


def dt_mutation(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    match number_parameter:
        case 0:
            # criterion
            list_criterion = ["gini", "entropy", "log_loss"]
            individual[0] = list_criterion[random.randint(0, 2)]
        case 1:
            # splitter
            list_splitter = ["best", "random"]
            individual[1] = list_splitter[random.randint(0, 1)]
        case 2:
            # max_depth
            individual[2] = random.randint(2, 8)
        case 3:
            # min_samples_split
            individual[3] = random.uniform(0.01, 1)
        case 4:
            # min_samples_leaf
            individual[4] = random.uniform(0.01, 5)
        case _:  # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0


def dt_parameters_fitness(y, df, number_of_attributes, individual):
    if with_selection:
        list_of_columns_to_drop = []  # lista cech do usuniecia
        for i in range(number_of_attributes, len(individual)):
            if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
                list_of_columns_to_drop.append(i - number_of_attributes)

        df = df.drop(df.columns[list_of_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = DecisionTreeClassifier(criterion=individual[0], splitter=individual[1], max_depth=individual[2],
                                       min_samples_split=individual[3],
                                       min_samples_leaf=individual[4], random_state=101)

    scores = model_selection.cross_val_score(estimator, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean(),
