from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, plot_confusion_matrix
import pandas as pd
import numpy as np
import plotly.express as px

def tune_classification_model_hyperparameters(model_class, X_train, y_train):
    model = model_class()
    param_grid = {
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
    }
    clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid,
        refit = 'accuracy'
    )
    clf.fit(X_train, y_train)
    return clf

def tune_classification_model_hyperparameters(model_class, X_train, y_train):
    model = model_class()
    param_grid = {    
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
        }
    param_grid_2 = {    
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
        }
    param_grid_3 = {    
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
        }
    try:
        clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid,
        refit = 'accuracy'
        )
        clf.fit(X_train, y_train)
    except ValueError:
        clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid_2,
        refit = 'accuracy'
        )
        clf.fit(X_train, y_train)
    except ValueError:
        clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid_3,
        refit = 'accuracy'
        )
        clf.fit(X_train, y_train)
    return clf