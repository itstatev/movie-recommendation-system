import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

class ModelEvaluator:
    """
    Class for evaluating recommendation models.
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

        self.spark.sparkContext.setLogLevel("WARN")

    def evaluate_model(self, predictions):
        """
        Evaluate a recommendation model using RMSE metric.

        :param predictions: DataFrame containing model predictions.
        :return: RMSE value.
        """
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        return rmse

    def run(self):
        """
        Run the model evaluation tasks.
        """
        ratings_path = os.getenv("PROCESSED_RATINGS_PATH", "data/processed_ratings.parquet")
        ratings = self.spark.read.parquet(ratings_path)

        als_predictions_path = os.getenv("ALS_PREDICTIONS_PATH", "predictions/als_predictions.parquet")
        baseline_predictions_path = os.getenv("BASELINE_PREDICTIONS_PATH", "predictions/baseline_predictions.parquet")
        lr_predictions_path = os.getenv("LR_PREDICTIONS_PATH", "predictions/lr_predictions.parquet")
        svd_predictions_path = os.getenv("SVD_PREDICTIONS_PATH", "predictions/svd_predictions.parquet")

        als_predictions = self.spark.read.parquet(als_predictions_path)
        baseline_predictions = self.spark.read.parquet(baseline_predictions_path)
        lr_predictions = self.spark.read.parquet(lr_predictions_path)
        svd_predictions = self.spark.read.parquet(svd_predictions_path)

        baseline_predictions = baseline_predictions.join(ratings, on=["userId", "movieId"], how="inner") \
            .select(baseline_predictions["userId"], baseline_predictions["movieId"], baseline_predictions["prediction"], ratings["rating"].alias("actual_rating"))

        lr_predictions = lr_predictions.join(ratings, on=["userId", "movieId"], how="inner") \
            .select(lr_predictions["userId"], lr_predictions["movieId"], lr_predictions["prediction"], ratings["rating"].alias("actual_rating"))

        svd_predictions = svd_predictions.join(ratings, on=["userId", "movieId"], how="inner") \
            .select(svd_predictions["userId"], svd_predictions["movieId"], svd_predictions["prediction"], ratings["rating"].alias("actual_rating"))

        # "actual_rating" to "rating" for evaluation
        baseline_predictions = baseline_predictions.withColumnRenamed("actual_rating", "rating")
        lr_predictions = lr_predictions.withColumnRenamed("actual_rating", "rating")
        svd_predictions = svd_predictions.withColumnRenamed("actual_rating", "rating")

        als_rmse = self.evaluate_model(als_predictions)
        baseline_rmse = self.evaluate_model(baseline_predictions)
        lr_rmse = self.evaluate_model(lr_predictions)
        svd_rmse = self.evaluate_model(svd_predictions)

        return {
            'ALS': als_rmse,
            'Baseline': baseline_rmse,
            'Linear Regression': lr_rmse,
            'SVD': svd_rmse
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run()
    print(results)