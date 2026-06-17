from src.pipeline.predict_pipeline import PredictPipeline


class PredictionService:

    def __init__(self):
        self.predictor = PredictPipeline()

    def predict(self, text):

        return self.predictor.predict(text)