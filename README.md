# Movie Recommendation System with PySpark

This project implements a movie recommendation system using PySpark. The system processes movie and ratings data, trains various recommendation models, tests and evaluates the models, and visualizes the results and recommendations.

## Project Structure

.
├── data_processing
│ ├── init.py
│ └── data_processor.py
├── models
│ ├── init.py
│ └── model_trainer.py
├── test
│ ├── init.py
│ └── model_tester.py
├── evaluation
│ ├── init.py
│ └── model_evaluator.py
├── visualization
│ ├── init.py
│ └── results_visualization.py
├── dataset
│ ├── movies.csv
│ └── ratings.csv
├── processed_data
├── models
├── predictions
├── visualizations
└── main.py


## Requirements

- Python 3.7 or later
- PySpark
- matplotlib
- pandas

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2. Install the required Python packages:
    ```sh
    pip install pyspark matplotlib pandas
    ```

## Running the Project

1. Set up the environment variables in `main.py` with the correct paths to your data files and directories.

2. Run the main script:
    ```sh
    python main.py
    ```

## Files and Modules

### `data_processing/data_processor.py`

Processes the movie and ratings data, converting them from CSV format to Parquet format for efficient processing.

### `models/model_trainer.py`

Trains various recommendation models, including ALS (Alternating Least Squares), baseline model (mean rating per movie), linear regression model, and SVD (Singular Value Decomposition) model.

### `test/model_tester.py`

Tests the trained recommendation models on the ratings data and generates predictions.

### `evaluation/model_evaluator.py`

Evaluates the performance of the recommendation models using metrics such as RMSE (Root Mean Square Error).

### `visualization/results_visualization.py`

Visualizes the results of the model evaluation and generates plots for the top 10 movie recommendations for a specified user.

## Example Usage

To visualize the RMSE comparison for different models:
```python
visualizer = ResultsVisualizer(results)
visualizer.plot_rmse()
visualizer.save_plot()
