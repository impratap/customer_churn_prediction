from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.exception import CustomException
from src.logger import logging
import sys
import numpy as np


def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Model Evaluation started")
        """Evaluate the model with multiple metrics."""
        y_pred = model.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred,pos_label=1),
            "Recall": recall_score(y_test, y_pred,pos_label=1),
            "F1 Score": f1_score(y_test, y_pred,pos_label=1)
        }
        
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        logging.info('Model Evalaution completed')
        return metrics
    except Exception as e:
        raise CustomException(e,sys)