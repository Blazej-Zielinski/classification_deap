from enum import Enum


class Selection(Enum):
    BEST = 'selBest'
    ROULETTE_WHEEL = 'selRoulette'
    WORST = 'selWorst'
    TOURNAMENT = 'selTournament'
    RANDOM = 'selRandom'
    # Additional
    STOCH_UNIVERSAL_SAMPLING = 'selStochasticUniversalSampling'


class Classifier(Enum):
    RANDOM_FOREST = 'RANDOM_FOREST'
    LOGISTIC_REGRESSION = 'LOGISTIC_REGRESSION'
    SVC = 'SVC'
    K_NEIGHBOURS = 'K_NEIGHBOURS'
    DECISION_TREE = 'DECISION_TREE'


with_selection = False

config = {
    'classifier': Classifier.SVC,
    'select': Selection.BEST,

    'size_population': 5,
    'select_size': 5,
    'tournament_size': 1,
    'probability_mutate': 1,

    'number_iteration': 50,
}
