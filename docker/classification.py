from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, plot_confusion_matrix
import pandas as pd
import numpy as np
import plotly.express as px

def tune_classification_model_hyperparameters(model_class, X_train, y_train, param_grid):
    model = model_class()
    class_name = model.__class__.__name__
    clf = GridSearchCV(
        model, 
        cv = 10, 
        scoring = ['accuracy'], 
        param_grid = param_grid, 
        refit = 'accuracy')
    clf.fit(X_train, y_train)
    return clf, clf.best_params_, clf.best_score_, class_name