# Machine Learning Application

This is a Python-based machine learning application built using Streamlit. It demonstrates the implementation of various supervised machine learning algorithms for classification and regression tasks.

## Introduction

Machine Learning is concerned with using algorithms that automatically improve through iteration to produce a model that can predict outcomes of new data. This application allows users to explore different datasets, choose algorithms, and evaluate their performance.

## Features

- Data Wrangling: The application provides options to select and visualize different datasets, such as Iris, Breast Cancer, Wine, California Housing, Boston House, and Diabetes.
- ML Algorithms: Users can choose between supervised and unsupervised learning. In the supervised learning section, they can select between classification and regression tasks and further specify the algorithm to use.
- Classification: For classification tasks, algorithms like K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees are available. The application allows users to tune hyperparameters and evaluates the model's accuracy.
- Regression: For regression tasks, algorithms like Linear Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Decision Trees are available. Users can tune hyperparameters and evaluate the model's performance using Mean Absolute Percentage Error (MAPE).

## Installation and Usage

1. Clone the repository:

```shell
git clone https://github.com/your-username/machine-learning-app.git
```

2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

3. Run application :

```shell
streamlit run machine_learning_app.py
```
4. Open the application in your web browser:

```shell 
Local URL: http://localhost:8501
Network URL: http://192.168.X.X:8501
```
5. Interact with the application by selecting different datasets, algorithms, and hyperparameters to train and evaluate machine learning models.

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## File Structure
The code is organized into the following files:

- machine_learning_app.py: The main file that runs the machine learning application and handles user interactions.
- data_loader.py: Contains the DataLoader class for loading and preprocessing datasets.
- model_trainer.py: Contains the ModelTrainer class for training and evaluating machine learning models.

## License
This project is licensed under the MIT License.

Feel free to explore and use this machine learning application to gain insights into different datasets and algorithms!

## ==>
this project was collaboriation with 3 students as a graduation project it was has alot more but it deleted because the dependeinces was not stable