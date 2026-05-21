import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utlis import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):  
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            params={
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_intercept': [True, False]
                },
                "Lasso": {
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_intercept': [True, False]
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5]
                },
                "CatBoost": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'depth': [3, 4, 5]
                }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]  
            
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )
            predicted =best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square 

        except Exception as e:
            raise CustomException(e,sys)
            