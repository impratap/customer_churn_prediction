import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import sys

def load_data(file_path):
    try: 
        """Load the dataset from a csv file."""
        logging.info("Dataset Loaded")
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e,sys)



def clean_data(df):
    try:
        """Handle missing values and data inconsistencies"""
        logging.info("Data Cleaned")
        # Convert TotalCharges to numeric, fill missing with median
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)
    except Exception as e:
        raise CustomException(e,sys)


    # Drop unnecessary columns (e.g., customerID)
    df=df.drop('customerID',axis=1)
    return df


def encode_categorical(df):
    try:
        """Encode Categorical Variables"""
        logging.info("Encoding performed")
        le=LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col]=le.fit_transform(df[col])
        return df
    except Exception as e:
        raise CustomException(e,sys)


def scale_features(X):
    try:
        """Scale numerical features."""
        scaler=StandardScaler()
        return scaler.fit_transform(X), scaler
    except Exception as e:
        raise CustomException (e,sys)



def preprocess_data(file_path, test_size=0.2, random_state=42):
    try:
        """Full preprocessing pipeline"""
        logging.info("Pipeline executed")

        # Load and clean
        df=load_data(file_path)
        df=clean_data(df)


        # Separate features and target

        X=df.drop('Churn',axis=1)
        y=df['Churn']


        # Encode categorical variables 
        X=encode_categorical(X)


        # Split data

        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=test_size, random_state=random_state)


        # Scale features
        X_train_scaled, scaler =scale_features(X_train)
        X_test_scaled=scaler.transform(X_test)


        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    
    except Exception as e:
        raise CustomException (e,sys)
