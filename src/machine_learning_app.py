import streamlit as st
from data_loader import DataLoader
from model_trainer import ModelTrainer

class MachineLearningApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer()

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
                    self.data_loader.load_dataset_classification(self.dataset_name)
                    self.show_data_summary()

                    self.model_trainer.add_parameters_classification(self.classifier_name)
                    clf = self.model_trainer.get_classifier_classification(self.classifier_name)

                    self.model_trainer.train_and_evaluate_classifier(clf)

                    if self.dataset_name == 'iris':
                        self.model_trainer.visualize_iris_dataset()

                elif self.vised == 'supervised' and self.method == 'Regression':
                    self.dataset_name = st.sidebar.selectbox("Dataset", ['california_housing', 'boston_house', 'diabetes'])
                    self.regressor_name = st.sidebar.selectbox("Regressor", ("LinearRegressor", "SVM", "KNeighborsRegressor", "Decision_Tree"))
                    self.data_loader.load_dataset_regression(self.dataset_name)
                    self.show_data_summary()

                    self.model_trainer.add_parameters_regression(self.regressor_name)
                    regressor = self.model_trainer.get_regressor(self.regressor_name)

                    self.model_trainer.train_and_evaluate_regressor(regressor)

    def show_data_summary(self):
        st.write(self.dataset_name, "dataset:")
        st.write("Number of instances:", self.data_loader.X.shape[0])
        st.write("Number of features:", self.data_loader.X.shape[1])
        st.write("Number of classes:", len(np.unique(self.data_loader.y)))


if __name__ == '__main__':
    app = MachineLearningApp()
    app.run()
