import random

from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import with_selection


def SVC_parameters(number_features, icls):
    genome = list()

    # kernel
    list_kernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(list_kernel[random.randint(0, 3)])

    # c
    k = random.uniform(0.1, 5)
    genome.append(k)

    # degree
    genome.append(random.uniform(0.1, 5))

    # gamma
    gamma = random.uniform(0.001, 2)
    genome.append(gamma)

    # coefficient
    coefficient = random.uniform(0.01, 1)
    genome.append(0.00631)

    if with_selection:
        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

    return icls(genome)


def SVC_mutation(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    match number_parameter:
        case 0:
            # kernel
            list_kernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = list_kernel[random.randint(0, 3)]
        case 1:
            # C
            k = random.uniform(0.1, 5)
            individual[1] = k
        case 2:
            # degree
            individual[2] = random.uniform(0.1, 5)
        case 3:
            # gamma
            gamma = random.uniform(0.01, 2)
            individual[3] = gamma
        case 4:
            # coefficient
            coefficient = random.uniform(0.1, 1)
            individual[4] = coefficient
            pass
        case _: #genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0


def SVC_parameters_fitness(y, df, number_of_attributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    if with_selection:
        list_of_columns_to_drop = []  # lista cech do usuniecia
        for i in range(number_of_attributes, len(individual)):
            if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
                list_of_columns_to_drop.append(i - number_of_attributes)

        df = df.drop(df.columns[list_of_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                    coef0=individual[4], random_state=101)

    result_sum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (
                tp + fp + tn + fn)  # w oparciu o macierze pomyłek https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
        result_sum = result_sum + result  # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej

    scores = model_selection.cross_val_score(estimator, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean(),
    # return result_sum / split,

