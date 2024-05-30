import os
from data_processing.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from test.model_tester import ModelTester
from evaluation.model_evaluator import ModelEvaluator
from visualizations.visualizer import ResultsVisualizer

def main():
    os.environ["MOVIES_PATH"] = "dataset/movies.csv"
    os.environ["RATINGS_PATH"] = "dataset/ratings.csv"
    os.environ["PROCESSED_MOVIES_PATH"] = "processed_data/processed_movies.parquet"
    os.environ["PROCESSED_RATINGS_PATH"] = "processed_data/processed_ratings.parquet"
    os.environ["ALS_MODEL_PATH"] = "models/als_model"
    os.environ["BASELINE_MODEL_PATH"] = "models/baseline_model"
    os.environ["LR_MODEL_PATH"] = "models/lr_model"
    os.environ["SVD_MODEL_PATH"] = "models/svd_model"
    os.environ["ALS_PREDICTIONS_PATH"] = "predictions/als_predictions.parquet"
    os.environ["BASELINE_PREDICTIONS_PATH"] = "predictions/baseline_predictions.parquet"
    os.environ["LR_PREDICTIONS_PATH"] = "predictions/lr_predictions.parquet"
    os.environ["SVD_PREDICTIONS_PATH"] = "predictions/svd_predictions.parquet"

    os.makedirs("dataset", exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    processor = DataProcessor()
    processor.run()

    trainer = ModelTrainer()
    als_model, baseline_model, lr_model, svd_model = trainer.run()

    tester = ModelTester()
    tester.run(als_model, baseline_model, lr_model, svd_model)

    evaluator = ModelEvaluator()
    results = evaluator.run()

    visualizer = ResultsVisualizer(results)
    visualizer.plot_rmse()
    visualizer.save_plot()

    movies_path = os.getenv("MOVIES_PATH", "dataset/movies.csv")
    spark = tester.spark 
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)

    user_id = 1  
    recommendations = als_model.recommendForAllUsers(10)
    visualizer.plot_recommendations(recommendations, movies, user_id)
    visualizer.save_recommendations_plot(recommendations, movies, user_id)

if __name__ == "__main__":
    main()