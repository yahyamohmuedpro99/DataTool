import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, LinearSVR
from sklearn import tree
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self):
        self.params = {}

    def add_parameters_classification(self, classifier_name):
        self.params = {}

        if classifier_name == "KNN":
            self.params["n_neighbors"] = st.sidebar.slider("Number of neighbors (K)", 1, 15, 5)
        elif classifier_name == "SVM":
            self.params["C"] = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
            self.params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        elif classifier_name == "Decision_Tree":
            self.params["max_depth"] = st.sidebar.slider("Max depth", 1, 10, 5)

    def get_classifier_classification(self, classifier_name):
        if classifier_name == "KNN":
            return KNeighborsClassifier(n_neighbors=self.params["n_neighbors"])
        elif classifier_name == "SVM":
            return SVC(C=self.params["C"], kernel=self.params["kernel"])
        elif classifier_name == "Decision_Tree":
            return tree.DecisionTreeClassifier(max_depth=self.params["max_depth"])

    def train_and_evaluate_classifier(self, clf):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

    def visualize_iris_dataset(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df["target"] = self.y
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="target", palette="viridis")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Iris Dataset Visualization")
        st.pyplot()

    def add_parameters_regression(self, regressor_name):
        self.params = {}

        if regressor_name == 'LinearRegressor':
            self.params["normalize"] = st.sidebar.checkbox("Normalize input features", value=True)
        elif regressor_name == 'SVM':
            self.params["C"] = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
            self.params["epsilon"] = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)
        elif regressor_name == 'KNeighborsRegressor':
            self.params["n_neighbors"] = st.sidebar.slider("Number of neighbors (K)", 1, 15, 5)
        elif regressor_name == 'Decision_Tree':
            self.params["max_depth"] = st.sidebar.slider("Max depth", 1, 10, 5)

    def get_regressor(self, regressor_name):
        if regressor_name == 'LinearRegressor':
            return LinearSVR(C=self.params["C"], epsilon=self.params["epsilon"])
        elif regressor_name == 'SVM':
            return SVC(C=self.params["C"], kernel=self.params["kernel"])
        elif regressor_name == 'KNeighborsRegressor':
            return KNeighborsRegressor(n_neighbors=self.params["n_neighbors"])
        elif regressor_name == 'Decision_Tree':
            return tree.DecisionTreeRegressor(max_depth=self.params["max_depth"])

    def train_and_evaluate_regressor(self, regressor):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        st.write("Mean Absolute Percentage Error:", mape)


class MachineLearningApp:
    def __init__(self):
        self.dataset_name = ""
        self.vised = ""
        self.method = ""
        self.classifier_name = ""
        self.regressor_name = ""
        self.X = None
        self.y = None
        self.table = None

    def run(self):
        st.write("Machine Learning is concerned with using algorithms that automatically improve through iteration to produce a model that can predict outcomes of new data.")
        self.sections = st.selectbox("#choose sections", ['Data Wrangling', 'ML Algorithms', 'Application in Stock Prices'])

        if self.sections == 'ML Algorithms':
            self.vised = st.selectbox("Choose", ['supervised', 'unsupervised'], 1)

            if self.vised == "supervised":
                self.method = st.selectbox("Choose the method", ['Classification', 'Regression'])

                if self.vised == 'supervised' and self.method == 'Classification':
                    self.dataset_name = st.sidebar.selectbox("Dataset", ("iris", "Breast Cancer", 'wine'))
                    self.classifier_name = st.sidebar.selectbox("Classifier", ("KNN", "SVM", "Decision_Tree"))
                    self.X, self.y = self.get_dataset_classification(self.dataset_name)
                    self.table = self.get_data_frame(self.dataset_name)
                    self.show_data_summary()

                    model_trainer = ModelTrainer()
                    model_trainer.add_parameters_classification(self.classifier_name)
                    clf = model_trainer.get_classifier_classification(self.classifier_name)

                    model_trainer.train_and_evaluate_classifier(clf)

                    if self.dataset_name == 'iris':
                        model_trainer.visualize_iris_dataset()

                elif self.vised == 'supervised' and self.method == 'Regression':
                    self.dataset_name = st.sidebar.selectbox("Dataset", ['california_housing', 'boston_house', 'diabetes'])
                    self.regressor_name = st.sidebar.selectbox("Regressor", ("LinearRegressor", "SVM", "KNeighborsRegressor", "Decision_Tree"))
                    self.X, self.y = self.get_dataset_regression(self.dataset_name)
                    self.table = self.get_data_frame(self.dataset_name)
                    self.show_data_summary()

                    model_trainer = ModelTrainer()
                    model_trainer.add_parameters_regression(self.regressor_name)
                    regressor = model_trainer.get_regressor(self.regressor_name)

                    model_trainer.train_and_evaluate_regressor(regressor)

    def get_dataset_classification(self, dataset_name):
        if dataset_name == "iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        else:
            data = datasets.load_wine()

        X = data.data
        y = data.target
        return X, y

    def get_data_frame(self, dataset_name):
        if dataset_name == "iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        else:
            data = datasets.load_wine()

        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    def get_dataset_regression(self, dataset_name):
        if dataset_name == 'california_housing':
            dataset = datasets.fetch_california_housing()
        elif dataset_name == 'boston_house':
            dataset = datasets.load_boston()
        else:
            dataset = datasets.load_diabetes()

        X = dataset.data
        y = dataset.target
        return X, y

    def show_data_summary(self):
        st.write(self.dataset_name, "dataset:")
        st.write("Number of instances:", self.X.shape[0])
        st.write("Number of features:", self.X.shape[1])
        st.write("Number of classes:", len(np.unique(self.y)))
        st.write("Sample data:")
        st.write(self.table.head())


if __name__ == '__main__':
    app = MachineLearningApp()
    app.run()
