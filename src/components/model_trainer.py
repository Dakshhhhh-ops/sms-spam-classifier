import os
import sys
import pandas as pd
from dataclasses import dataclass
import mlflow
import mlflow.sklearn

# 1. Changed import from Naive Bayes to Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, classification_report

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array_path, test_array_path, vectorizer_path):
        try:
            logging.info("Loading transformed train and test data")
            train_df=pd.read_csv(train_array_path)
            test_df=pd.read_csv(test_array_path)

            train_df['transformed_text'] = train_df['transformed_text'].fillna('')
            test_df['transformed_text'] = test_df['transformed_text'].fillna('')

            logging.info("loading the saved vectorizer")

            vectorizer = load_object(vectorizer_path)

            X_train = vectorizer.transform(train_df['transformed_text'])
            X_test = vectorizer.transform(test_df['transformed_text'])

            y_train=train_df['target'].values
            y_test = test_df['target'].values

            logging.info("Training the Random Forest model with MLflow tracking")
            mlflow.set_experiment("SMS_Classifier")
            mlflow.sklearn.autolog()

            with mlflow.start_run():

                model = MultinomialNB()

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)

                # custom metrics
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)

            logging.info(f"Random Forest Training Completed. Accuracy: {accuracy}, Precision: {precision}")

            print("\nClassification Report:\n", classification_report(y_test, predictions))
            if accuracy < 0.6:
                raise CustomException("Model accuracy is too low!", sys)

            logging.info("Saving the trained Random Forest model object")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(
                    self.model_trainer_config.trained_model_file_path,
                    model
            )
 
            return accuracy, precision
                
        except Exception as e:
                raise CustomException(e, sys)
