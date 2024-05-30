import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

class ModelTrainer:
    """
    Class for training recommendation models.
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

        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
        self.spark.conf.set("spark.sql.shuffle.partitions", "800")

    def train_als_model(self, ratings):
        """
        Train an ALS (Alternating Least Squares) model on the ratings data.

        :param ratings: Spark DataFrame containing ratings data.
        :return: Trained ALS model.
        """
        als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", maxIter=5, regParam=0.1, coldStartStrategy="drop")
        model = als.fit(ratings)
        return model

    def train_baseline_model(self, ratings):
        """
        Train a baseline model (mean rating per movie) on the ratings data.

        :param ratings: Spark DataFrame containing ratings data.
        :return: Trained baseline model (mean ratings).
        """
        average_ratings = ratings.groupBy("movieId").avg("rating")
        return average_ratings

    def train_linear_regression_model(self, ratings):
        """
        Train a linear regression model on the ratings data.

        :param ratings: Spark DataFrame containing ratings data.
        :return: Trained linear regression model.
        """
        assembler = VectorAssembler(inputCols=["userId", "movieId"], outputCol="features")
        ratings = assembler.transform(ratings)
        lr = LinearRegression(featuresCol="features", labelCol="rating")
        model = lr.fit(ratings)
        return model

    def train_svd_model(self, ratings):
        """
        Train an SVD (Singular Value Decomposition) model on the ratings data.

        :param ratings: Spark DataFrame containing ratings data.
        :return: Trained SVD model.
        """
        als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", maxIter=5, regParam=0.1, coldStartStrategy="drop")
        model = als.fit(ratings)
        return model

    def run(self):
        """
        Run the model training tasks.
        """
        ratings_path = os.getenv("PROCESSED_RATINGS_PATH", "data/processed_ratings.parquet")
        als_model_path = os.getenv("ALS_MODEL_PATH", "models/als_model")
        baseline_model_path = os.getenv("BASELINE_MODEL_PATH", "models/baseline_model")
        lr_model_path = os.getenv("LR_MODEL_PATH", "models/lr_model")
        svd_model_path = os.getenv("SVD_MODEL_PATH", "models/svd_model")

        ratings = self.spark.read.parquet(ratings_path)

        als_model = self.train_als_model(ratings)
        baseline_model = self.train_baseline_model(ratings)
        lr_model = self.train_linear_regression_model(ratings)
        svd_model = self.train_svd_model(ratings)

        als_model.save(als_model_path)
        baseline_model.write.parquet(baseline_model_path)
        lr_model.save(lr_model_path)
        svd_model.save(svd_model_path)

        return als_model, baseline_model, lr_model, svd_model

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
