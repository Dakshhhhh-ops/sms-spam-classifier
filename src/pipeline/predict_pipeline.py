import os
import sys

from src.utils import load_object, transform_text
from src.exception import CustomException


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.vectorizer_path = os.path.join(
            "artifacts",
            "vectorizer.pkl"
        )

        # Load once
        self.model = load_object(self.model_path)

        self.vectorizer = load_object(
            self.vectorizer_path
        )

    def predict(self, text):

        try:

            transformed_text = transform_text(text)

            vector_input = self.vectorizer.transform(
                [transformed_text]
            )

            prediction = self.model.predict(
                vector_input
            )[0]

            probabilities = self.model.predict_proba(
                vector_input
            )[0]

            ham_probability = probabilities[0]
            spam_probability = probabilities[1]

            return {
                "prediction": (
                    "Spam"
                    if prediction == 1
                    else "Ham"
                ),
                "spam_probability": float(
                    spam_probability
                ),
                "ham_probability": float(
                    ham_probability
                )
            }

        except Exception as e:
            raise CustomException(e, sys)