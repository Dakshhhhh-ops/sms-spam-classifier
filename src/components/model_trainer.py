import os
import sys
import pandas as pd
import pickle
from dataclasses import dataclass

# 1. Changed import from Naive Bayes to Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

from src.logger import logging
from src.exception import CustomException

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

                with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)

                X_train = vectorizer.transform(train_df['transformed_text'])
                X_test = vectorizer.transform(test_df['transformed_text'])

                y_train=train_df['target'].values
                y_test = test_df['target'].values

                logging.info("Training the Random Forest model")
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions) 
                precision = precision_score(y_test, predictions)

                logging.info(f"Random Forest Training Completed. Accuracy: {accuracy}, Precision: {precision}")

                print("\nClassification Report:\n", classification_report(y_test, predictions))
                if accuracy < 0.6:
                    raise CustomException("Model accuracy is too low!", sys)

                logging.info("Saving the trained Random Forest model object")
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
                with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                    pickle.dump(model, f)

                return accuracy, precision
                
            except Exception as e:
                    raise CustomException(e, sys)
