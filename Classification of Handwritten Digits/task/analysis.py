# write your code here
import keras.datasets.mnist
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
# the function


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    predicted_value = model.predict(features_test)
    score = accuracy_score(target_test, predicted_value)
    # here you fit the model
    # make a prediction
    # calculate accuracy and save it to score
    print(f'Model: {model}\nAccuracy: {score:.3f}\n')


if __name__ == "__main__":
    # First Stage

    (X_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    X_train_reshaped = X_train.reshape(60000, 784)
    X_train_minimum = X_train_reshaped.min()
    X_train_maximum = X_train_reshaped.max()
    '''
    print("Classes:", np.unique(y_train))
    print("Features' shape:", X_train_reshaped.shape)
    print("Target's shape:", y_train.shape)
    print("min: " + str(float(X_train_minimum)), "max: " + str(float(X_train_maximum)))
    '''

    # Second Stage
    X_feat_train, X_feat_test, y_feat_train, y_feat_test = train_test_split(
        X_train_reshaped[:6000],
        y_train[:6000],
        test_size=0.3,
        random_state=40
    )
    # print("x_train shape:", X_feat_train.shape)
    # print("x_test shape:", X_feat_test.shape)
    # print("y_train shape:", y_feat_train.shape)
    # print("y_test shape:", y_feat_test.shape)
    # print("Proportion of samples per class in train set:", pd.Series(y_feat_train).value_counts(normalize=True), sep="\n")

    # Third Stage
    '''
    models_tuple = (KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression(random_state=40), RandomForestClassifier(random_state=40))
    for model in models_tuple:
        fit_predict_eval(model, X_feat_train, X_feat_test, y_feat_train, y_feat_test)
    print("The answer to the question: RandomForestClassifier -", 0.939)
    '''

    # Fourth Stage

    x_train_norm, x_test_norm = Normalizer().transform(X_feat_train), Normalizer().transform(X_feat_test)
    '''
    models_tuple = (KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression(random_state=40), RandomForestClassifier(random_state=40))
    for model in models_tuple:
        fit_predict_eval(model, x_train_norm, x_test_norm, y_feat_train, y_feat_test)
    print("The answer to the 1st question: yes\n")
    print("The answer to the 2nd question: KNeighborsClassifier-0.953, RandomForestClassifier-0.937")
    '''

    # Fifth Stage
    k_neigh_class_model = KNeighborsClassifier()
    k_neigh_params = {"n_neighbors": [3, 4], "weights": ['uniform', 'distance'], "algorithm": ['auto', 'brute']}
    clf_neigh_params = GridSearchCV(k_neigh_class_model, param_grid=k_neigh_params, scoring='accuracy', n_jobs=-1)

    rand_for_class_model = RandomForestClassifier(random_state=40)
    rand_for_params = {"n_estimators": [300, 500], "max_features": ['sqrt', 'log2'], "class_weight": ['balanced', 'balanced_subsample'],
                       "bootstrap": [True, False]}
    clf_rand_for_params = GridSearchCV(rand_for_class_model, param_grid=rand_for_params, scoring='accuracy', n_jobs=-1)

    clf_neigh_params.fit(x_train_norm, y_feat_train)
    clf_rand_for_params.fit(x_train_norm, y_feat_train)

    print("K-nearest neighbours algorithm")
    fit_predict_eval(model=clf_neigh_params.best_estimator_, features_train=x_train_norm, features_test=x_test_norm, target_train=y_feat_train, target_test=y_feat_test)

    print("Random forest algorithm")
    fit_predict_eval(model=clf_rand_for_params.best_estimator_, features_train=x_train_norm, features_test=x_test_norm, target_train=y_feat_train, target_test=y_feat_test)
