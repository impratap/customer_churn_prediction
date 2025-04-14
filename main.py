from src.data_preprocessing import preprocess_data

X_train,X_test, y_train, y_test, scaler = preprocess_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(X_train.shape, X_test.shape)