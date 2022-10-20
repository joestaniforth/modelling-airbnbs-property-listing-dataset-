from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
import xgboost as xgb

class Models:
    def __init__(self) -> None:
        self.classifier_params = {
            DecisionTreeClassifier:{
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': range(5, 11, 1),
                'max_leaf_nodes': [100, 250],
                'max_depth': range(1, 11)
            },
            RandomForestClassifier:{
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': range(5, 11, 1),
                'max_leaf_nodes': [100, 250],
                'max_depth': range(1, 11)
            },
            GradientBoostingClassifier:{
                'learning_rate': np.arange(0.1, 1.3, 0.2),
                'n_estimators': [100, 250, 500],
                'subsample': np.arange(0.1, 1.3, 0.3),
                'criterion': ['squared_error'],
                'max_depth': range(1, 11)
            },
            LogisticRegression:{
                'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
                'penalty': ['l2'],
                'C': np.arange(0.2, 1.2, 0.2),
            }
        }
        self.regressor_params = {
            DecisionTreeRegressor:{
                'max_depth': [1, 5, 10],
                'min_samples_split': [2, 5, 7],
                'min_weight_fraction_leaf': [0, 0.25, 0.5]
            },
            RandomForestRegressor:{
                'n_estimators': [100, 250, 500],
                'max_depth': [1, 5, 10],
                'min_weight_fraction_leaf': [0, 0.25, 0.5],
                'max_samples': range(10, 60, 10)
            },
            GradientBoostingRegressor:{
                'max_depth': [1, 5, 10],
                'min_samples_split': [2, 5, 7],
                'min_weight_fraction_leaf': [0, 0.25, 0.5]
            },
            xgb.XGBRegressor:{
                'gamma': [0],
                'learning_rate':[0.05, 0.25, 0.05],
                'n_estimators': [100],
                'min_child_weight': range(1, 11),
                'max_depth': range(1, 6)
            }       
        }
        self.nn_params = {
            'learning_rate': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
            'hidden_layer_width': [16, 11, 8, 5, 1],
            'depth': [2],
            'epochs': [20, 40, 60, 80, 100],
            'dropout': [0.0, 0.2, 0.3, 0.4, 0.5],
            'training_batch_size': [32, 64, 128, 256]
        }