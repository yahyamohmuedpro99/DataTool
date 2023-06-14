import numpy as np
from sklearn import datasets

class DataLoader:
    def __init__(self):
        self.X = None
        self.y = None

    def load_dataset_classification(self, dataset_name):
        if dataset_name == "iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        else:
            data = datasets.load_wine()

        self.X = data.data
        self.y = data.target

    def load_dataset_regression(self, dataset_name):
        if dataset_name == 'california_housing':
            dataset = datasets.fetch_california_housing()
        elif dataset_name == 'boston_house':
            dataset = datasets.load_boston()
        else:
            dataset = datasets.load_diabetes()

        self.X = dataset.data
        self.y = dataset.target
