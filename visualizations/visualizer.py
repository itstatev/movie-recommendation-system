import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import explode, col

class ResultsVisualizer:
    """
    Class for visualizing results and recommendations.
    """

    def __init__(self, results):
        self.results = results

    def plot_rmse(self):
        """
        Plot a bar chart of RMSE values for different models.
        """
        models = list(self.results.keys())
        rmse_values = list(self.results.values())

        plt.figure(figsize=(10, 6))
        plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple'])
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model RMSE Comparison')
        plt.ylim(0, max(rmse_values) + 0.1)  
        plt.show()

    def save_plot(self, filename="visualizations/rmse_comparison.png"):
        """
        Save a bar chart of RMSE values for different models to a file.

        :param filename: Path to save the plot image.
        """
        models = list(self.results.keys())
        rmse_values = list(self.results.values())

        plt.figure(figsize=(10, 6))
        plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple'])
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model RMSE Comparison')
        plt.ylim(0, max(rmse_values) + 0.1)  
        plt.savefig(filename)

    def plot_recommendations(self, recommendations, movies, user_id):
        """
        Plot a horizontal bar chart of the top 10 movie recommendations for a specific user.

        :param recommendations: DataFrame containing recommendations.
        :param movies: DataFrame containing movie details.
        :param user_id: ID of the user for whom to plot recommendations.
        """
        recommendations = recommendations.withColumn("recommendation", explode("recommendations"))
        recommendations = recommendations.select("userId", col("recommendation.movieId"), col("recommendation.rating").alias("prediction"))
        
        recommendations_df = recommendations.filter(recommendations.userId == user_id).toPandas()
        print("Recommendations DataFrame after exploding:", recommendations_df.columns)

        movie_titles = movies.toPandas().set_index('movieId').loc[recommendations_df['movieId']]['title']
        recommendations_df['title'] = movie_titles.values
        recommendations_df = recommendations_df.sort_values(by='prediction', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        plt.barh(recommendations_df['title'], recommendations_df['prediction'], color='skyblue')
        plt.xlabel('Predicted Rating')
        plt.title(f'Top 10 Movie Recommendations for User {user_id}')
        plt.gca().invert_yaxis()
        plt.show()

    def save_recommendations_plot(self, recommendations, movies, user_id, filename="visualizations/recommendations.png"):
        """
        Save a horizontal bar chart of the top 10 movie recommendations for a specific user to a file.

        :param recommendations: DataFrame containing recommendations.
        :param movies: DataFrame containing movie details.
        :param user_id: ID of the user for whom to plot recommendations.
        :param filename: Path to save the plot image.
        """
        recommendations = recommendations.withColumn("recommendation", explode("recommendations"))
        recommendations = recommendations.select("userId", col("recommendation.movieId"), col("recommendation.rating").alias("prediction"))

        recommendations_df = recommendations.filter(recommendations.userId == user_id).toPandas()

        movie_titles = movies.toPandas().set_index('movieId').loc[recommendations_df['movieId']]['title']
        recommendations_df['title'] = movie_titles.values
        recommendations_df = recommendations_df.sort_values(by='prediction', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        plt.barh(recommendations_df['title'], recommendations_df['prediction'], color='skyblue')
        plt.xlabel('Predicted Rating')
        plt.title(f'Top 10 Movie Recommendations for User {user_id}')
        plt.gca().invert_yaxis()
        plt.savefig(filename)