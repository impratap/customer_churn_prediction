from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from src.exception import CustomException
from src.logger import logging
import sys


def train_models(X_train,y_train,X_test,y_test):
    try:
        logging.info("Model training Started")
        """Train multiple models with hyperparameter tuning and select the best model"""

        #Define models and their hyperparameter grids.

        models_params={
            "Logistics regression":{
                'model':LogisticRegression(random_state=42, max_iter=1000),
                'params':{
                    'C':[0.01,0.1,1,10],
                    'penalty':['l2'],
                    'solver':['lbfgs']
                }
            },

            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
            }
        },
            "AddaBoost": {
                "model": AdaBoostClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
            }
        },

            "DecisionTree":{
                'model': DecisionTreeClassifier(random_state=42),
                'params':{
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
                
            },
        }

        }


        best_model=None
        best_score=0
        best_name=""


        # Train and tune each model


        for name, config in models_params.items():
            print(f"Training {name}...")
            grid_search=GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            grid_search.fit(X_train,y_train)


            # Print best parameters

            print(f'{name} best Params: {grid_search.best_params_}')


            # Evaluate on the test set
            y_pred=grid_search.best_estimator_.predict(X_test)
            score=accuracy_score(y_test,y_pred)
            print(f"{name} Test Accuracy: {score:.4f}")


            # Update best model if current model is better

            if score>best_score:
                best_score=score
                best_model=grid_search.best_estimator_
                best_name=name


        print(f"Best Model Selected: {best_name} with Accuracy: {best_score:.4f}")
        return best_model, best_name
    

    except Exception as e:
        raise CustomException(e,sys)
    

def save_model(model, path):

    try:
        """Save the trained model"""
        logging.info('Train model Saved')
        joblib.dump(model,path)

    except Exception as e:
        raise CustomException(e,sys)

        



