import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logging
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("--- Training Pipeline Started ---")

            # 1. Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            transformation = DataTransformation()
            (
                transformed_train_path,
                transformed_test_path,
                feature_encoder_path,
                vectorizer_path
            ) = transformation.initiate_data_transformation(train_path, test_path)

            # 3. Model Training (Isme hi accuracy print ho jayegi)
            trainer = ModelTrainer()
            trainer.initiate_model_trainer(
                train_array_path=transformed_train_path,
                test_array_path=transformed_test_path,
                vectorizer_path=vectorizer_path
            )

            logging.info("--- Training Pipeline Completed Successfully! ---")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()