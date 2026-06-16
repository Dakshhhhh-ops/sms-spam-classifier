import os
import sys

from src.utils import load_object, transform_text
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.vectorizer_path = os.path.join("artifacts", "vectorizer.pkl")

    def predict(self, text):
        try:
            # Load trained model
            model = load_object(self.model_path)

            # Load vectorizer
            vectorizer = load_object(self.vectorizer_path)

            # Preprocess text
            transformed_text = transform_text(text)

            # Convert text to numerical features
            vector_input = vectorizer.transform([transformed_text])

            # Prediction
            prediction = model.predict(vector_input)[0]

            if prediction == 1:
                return "Spam"
            else:
                return "Ham"

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = PredictPipeline()

    text =  "Claim your reward NOW"

    result = predictor.predict(text)

    print(f"Message: {text}")
    print(f"Prediction: {result}")