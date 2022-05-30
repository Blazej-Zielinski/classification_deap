import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from src.config import with_selection
from sklearn import model_selection


def LR_parameters(number_features, icls):
    genome = list()

    # solver
    list_solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    genome.append(list_solver[random.randint(0, 4)])

    # c
    genome.append(random.uniform(0.1, 5))

    # fit_intercept
    genome.append(random.randint(0, 1))

    # max_iter
    genome.append(random.randint(100, 1000))

    if with_selection:
        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

    return icls(genome)


def LR_mutation(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    match number_parameter:
        case 0:
            list_solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            individual[0] = list_solver[random.randint(0, 4)]
        case 1:
            # C
            individual[1] = random.uniform(0.1, 5)
        case 2:
            # fit_intercept
            individual[2] = random.randint(0, 1)
        case 3:
            # max_iter
            individual[3] = random.randint(100, 1000)
        case _: #genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0


def LR_parameters_fitness(y, df, number_of_attributes, individual):

    if with_selection:
        list_of_columns_to_drop = []  # lista cech do usuniecia
        for i in range(number_of_attributes, len(individual)):
            if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
                list_of_columns_to_drop.append(i - number_of_attributes)

        df = df.drop(df.columns[list_of_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = LogisticRegression(solver=individual[0], C=individual[1], fit_intercept=individual[2], max_iter=individual[3],
                                   random_state=101)

    scores = model_selection.cross_val_score(estimator, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean(),
