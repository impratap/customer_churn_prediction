from src.data_preprocessing import preprocess_data
from src.model_training import train_models, save_model
from src.evalution import evaluate_model
from src.utils import load_model
from src.exception import CustomException
from src.logger import logging
import sys
import joblib


#preprocess data
X_train,X_test, y_train, y_test, scaler = preprocess_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


#Train Models
best_model,best_name=train_models(X_train,y_train,X_test,y_test)
print(f"Best Model: {best_name}")


#Evaluate
evaluate_model(best_model, X_test, y_test)

#save the model

save_model(best_model, "models/best_model.pkl")


def predict_churn(data, model_path, scaler):
    try:
        logging.info('Predict churn for new data.')
        model=load_model(model_path)
        data_scaled=scaler.transform(data)
        return model.predict(data_scaled)
    except Exception as e:
        raise CustomException(e,sys)
    


# Example: Predict for a single sample

sample=X_test[0:1]
prediction=predict_churn(sample,'models/best_model.pkl',scaler)
print(f"Prediction for sample:{'churn' if prediction[0]==1 else 'no Churn'}")


joblib.dump(scaler, "models/scaler.pkl")