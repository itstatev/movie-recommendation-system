import os
from pyspark.sql import SparkSession

class DataProcessor:
    """
    Class for processing movie and rating data.
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

    def load_data(self, movies_path, ratings_path):
        """
        Process the movies CSV file and save it as a parquet file.

        :param movies_path: Path to the movies CSV file.
        """
        movies = self.spark.read.csv(movies_path, header=True, inferSchema=True)
        ratings = self.spark.read.csv(ratings_path, header=True, inferSchema=True)
        return movies, ratings

    def process_data(self, movies, ratings):
        """
        Process the ratings CSV file and save it as a parquet file.

        :param ratings_path: Path to the ratings CSV file.
        """
        ratings = ratings.dropna()
        movies = movies.dropna()
        return movies, ratings

    def save_data(self, movies, ratings, movies_path, ratings_path):
        movies.write.parquet(movies_path)
        ratings.write.parquet(ratings_path)

    def run(self):
        """
        Run the data processing tasks.
        """
        movies_path = os.getenv("MOVIES_PATH", "data/movies.csv")
        ratings_path = os.getenv("RATINGS_PATH", "data/ratings.csv")
        processed_movies_path = os.getenv("PROCESSED_MOVIES_PATH", "data/processed_movies.parquet")
        processed_ratings_path = os.getenv("PROCESSED_RATINGS_PATH", "data/processed_ratings.parquet")

        movies, ratings = self.load_data(movies_path, ratings_path)
        movies, ratings = self.process_data(movies, ratings)
        self.save_data(movies, ratings, processed_movies_path, processed_ratings_path)

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()
