
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class Optimizer:
    def __init__(self, labels=['bad', 'good']):
        self.labels = labels
        self.set_classifier()
        self.set_params()
        self.set_grid_search()

    def set_classifier(self):
        self.clf = MLPClassifier(
            random_state=42
        )

    def __neuron_layers(self, base_neurons, n_layers):
        layers = []
        l = [base_neurons ** i for i in range(3, n_layers+1)]
        size = [l] * n_layers
        for h_layer in list(product(*size)):
            if h_layer not in layers:
                layers.append(h_layer)
        return layers

    def set_params(
        self,
        activation=['tanh'],
        solver=['sgd', 'adam'],
        n_neurons=2,
        n_layer_sizes=4
    ):
        self.optimizer_params = {
            'hidden_layer_sizes': self.__neuron_layers(n_neurons, n_layer_sizes),
            'activation': activation,
            'solver': solver,
            'alpha': [1/(10**i) for i in range(4, 7)],
        }

    def set_grid_search(self, cv=5):
        self.optimizer = GridSearchCV(
            self.clf,
            self.optimizer_params,
            n_jobs=-1,
            cv=cv
        )
