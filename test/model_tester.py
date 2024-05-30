import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler

class ModelTester:
    """
    Class for testing recommendation models.
    """

    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("MovieRecommendation") \
            .config("spark.driver.memory", "16g") \
            .config("spark.executor.memory", "16g") \
            .config("spark.executor.cores", "4") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", "800") \
            .config("spark.default.parallelism", "800") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.4") \
            .config("spark.executor.heartbeatInterval", "20s") \
            .config("spark.network.timeout", "300s") \
            .getOrCreate()

    def test_als_model(self, model, test_data):
        """
        Test a recommendation model on the ratings data.

        :param model: Trained model to be tested.
        :param ratings: Spark DataFrame containing ratings data.
        :return: DataFrame containing predictions.
        """
        predictions = model.transform(test_data)
        return predictions

    def test_baseline_model(self, model, test_data):
        """
        Test a recommendation model on the ratings data.

        :param model: Trained model to be tested.
        :param ratings: Spark DataFrame containing ratings data.
        :return: DataFrame containing predictions.
        """
        predictions = test_data.join(model, "movieId").select("userId", "movieId", "avg(rating)").withColumnRenamed("avg(rating)", "prediction")
        return predictions

    def test_lr_model(self, model, test_data):
        """
        Test a recommendation model on the ratings data.

        :param model: Trained model to be tested.
        :param ratings: Spark DataFrame containing ratings data.
        :return: DataFrame containing predictions.
        """
        assembler = VectorAssembler(inputCols=["userId", "movieId"], outputCol="features")
        test_data = assembler.transform(test_data)
        predictions = model.transform(test_data)
        return predictions

    def test_svd_model(self, model, test_data):
        """
        Test a recommendation model on the ratings data.

        :param model: Trained model to be tested.
        :param ratings: Spark DataFrame containing ratings data.
        :return: DataFrame containing predictions.
        """
        predictions = model.transform(test_data)
        return predictions

    def run(self, als_model, baseline_model, lr_model, svd_model):
        """
        Run the model testing tasks.

        :param als_model: Trained ALS model.
        :param baseline_model: Trained baseline model.
        :param lr_model: Trained linear regression model.
        :param svd_model: Trained SVD model.
        """
        ratings_path = os.getenv("PROCESSED_RATINGS_PATH", "data/processed_ratings.parquet")
        als_predictions_path = os.getenv("ALS_PREDICTIONS_PATH", "predictions/als_predictions.parquet")
        baseline_predictions_path = os.getenv("BASELINE_PREDICTIONS_PATH", "predictions/baseline_predictions.parquet")
        lr_predictions_path = os.getenv("LR_PREDICTIONS_PATH", "predictions/lr_predictions.parquet")
        svd_predictions_path = os.getenv("SVD_PREDICTIONS_PATH", "predictions/svd_predictions.parquet")

        ratings = self.spark.read.parquet(ratings_path)
        test_data = ratings.sample(fraction=0.2, seed=42)

        als_predictions = self.test_als_model(als_model, test_data)
        als_predictions.write.parquet(als_predictions_path)

        baseline_predictions = self.test_baseline_model(baseline_model, test_data)
        baseline_predictions.write.parquet(baseline_predictions_path)

        lr_predictions = self.test_lr_model(lr_model, test_data)
        lr_predictions.write.parquet(lr_predictions_path)

        svd_predictions = self.test_svd_model(svd_model, test_data)
        svd_predictions.write.parquet(svd_predictions_path)

if __name__ == "__main__":
    from models.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    als_model, baseline_model, lr_model, svd_model = trainer.run()

    tester = ModelTester()
    tester.run(als_model, baseline_model, lr_model, svd_model)
