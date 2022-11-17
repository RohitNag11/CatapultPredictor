import pandas as pd
import numpy as np
import itertools
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle


class ModelOptimiser:
    def __init__(self, data_name, data_path, target_column_name, test_size=0.3):
        self.data_name = data_name
        self.df = pd.read_csv(data_path)
        X, y_bin = self.__get_features_and_binary_target_arrays(
            target_column_name)
        self.input_dim, self.output_dim = 6, 2  # note: need to make a function
        self.scArray = self.__get_scaling_array(X)
        X_scaled = self.__get_scaled_features_array(X, self.scArray)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_bin, test_size=test_size)

    def __get_features_and_binary_target_arrays(self, target_column_name):
        """ Get features array, X and target binary array, y in self.df """
        X = pd.get_dummies(self.df.drop(
            [target_column_name], axis=1)).to_numpy()
        y = self.df[target_column_name].apply(lambda n: int(n)).to_numpy()
        y_bin = np.array([[abs(n-1), n] for n in y])
        return X, y_bin

    def __get_scaling_array(self, X):
        return np.array([[np.mean(x), np.std(x)] for x in X.T]).T

    def __get_scaled_features_array(self, X, scArray):
        return (X - scArray[0, :]) / scArray[1, :]

    def __generate_model(self, n_layers, units_with_activations, input_dim):
        model = Sequential()
        for i in range(0, n_layers):
            units, activations = units_with_activations[i]
            if (i == 0):
                model.add(
                    Dense(units=units, activation=activations, input_dim=input_dim))
            else:
                model.add(Dense(units=units, activation=activations))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd')
        return model

    def __get_all_permuations(self, all_values, n, output_val):
        permutations = []
        for permutation in itertools.permutations(all_values.flatten(), n):
            if permutation[-1] == output_val:
                permutations.append((permutation))
        return permutations

    def __get_all_units_permutations(self, min_units, max_units, output_dim, n_layers):
        all_digits = np.array(
            [[i]*n_layers for i in range(min_units, max_units+1)])
        return self.__get_all_permuations(all_digits, n_layers, output_dim)

    def __get_all_activations_permutations(self, possible_activations, n_layers):
        all_activations = np.array(
            [[activation]*n_layers for activation in possible_activations])
        return self.__get_all_permuations(all_activations, n_layers, 'softmax')

    def __get_all_activation_with_units_permutations(self, min_units, max_units, output_dim, possible_activations, n_layers):
        units_permutations = self.__get_all_units_permutations(
            min_units, max_units, output_dim, n_layers)
        activations_permutations = self.__get_all_activations_permutations(
            possible_activations, n_layers)
        visit_permuations = set()
        for activations in activations_permutations:
            for units in units_permutations:
                permuation = tuple([(units[i], activations[i])
                                    for i in range(n_layers)])
                visit_permuations.add(permuation)
        return np.array(list(visit_permuations))

    def get_model_id(self, epochs, layers, units_activations):
        id = f'{epochs}_{layers}'
        for item in units_activations:
            unit, activation = item
            id += f'_{unit}-{activation}'
        return id

    def __save_model_perf(self):
        with open(f'results/{self.data_name}_results/sorted_model_perf_data.p', 'wb') as fp:
            pickle.dump(self.sorted_models_perf, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

    def run(self, no_runs, epochs_range, layers_range, units_range, possible_activations):
        models_perf = {}
        for i in range(1, no_runs+1):
            for epochs in range(epochs_range[0], epochs_range[1] + 1, 50):
                for n_layers in range(layers_range[0], layers_range[1] + 1):
                    units_with_activations_permutations = self.__get_all_activation_with_units_permutations(
                        units_range[0], units_range[1], self.output_dim, possible_activations, n_layers)
                    for units_with_activations in units_with_activations_permutations:
                        model_id = self.get_model_id(
                            epochs, n_layers, units_with_activations)
                        model = self.__generate_model(
                            n_layers, units_with_activations, self.input_dim)
                        model.fit(self.X_train, self.y_train,
                                  epochs=epochs, batch_size=32)
                        y_pred = model.predict(self.X_test)
                        # print(f'run{i}, model: {model_id}')
                        if not np.isnan(y_pred).any():
                            y_pred_binary = np.array(
                                [np.around(y) for y in y_pred])
                            try:
                                accuracy = accuracy_score(
                                    self.y_test, y_pred_binary)
                                if model_id in models_perf:
                                    models_perf[model_id]['accuracies'] += (
                                        accuracy, )
                                    models_perf[model_id]['avg_accuracy'] = sum(
                                        models_perf[model_id]['accuracies'])/len(models_perf[model_id]['accuracies'])
                                else:
                                    models_perf[model_id] = {}
                                    models_perf[model_id]['accuracies'] = (
                                        accuracy,)
                                    models_perf[model_id]['avg_accuracy'] = accuracy
                            except ValueError:
                                pass
        self.sorted_models_perf = sorted(
            models_perf.items(), key=lambda x: sum(x[1]['accuracies']), reverse=True)
        self.__save_model_perf()
        return

    def read_model_perf(data_name):
        with open(f'results/{data_name}_results/sorted_model_perf_data.p', 'rb') as fp:
            sorted_model_perf = pickle.load(fp)
            fp.close()
        return sorted_model_perf
