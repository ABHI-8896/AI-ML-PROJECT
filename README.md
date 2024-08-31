# Fault Detection Model

This project involves building and evaluating machine learning models for fault detection using a dataset from heat pumps. The analysis includes data preprocessing, dimensionality reduction, and classification using various algorithms. The project employs Python's data science stack, including scikit-learn, pandas, and seaborn, for model training and evaluation.

## Features

- **Data Loading and Exploration**: Loads the dataset, performs exploratory data analysis (EDA), and visualizes correlations.
- **Data Preprocessing**: Scales features and applies Principal Component Analysis (PCA) for dimensionality reduction.
- **Model Training and Evaluation**:
  - **Logistic Regression**: Trains a logistic regression model and evaluates it using accuracy, MSE, RMSE, and AUC.
  - **Random Forest**: Trains a random forest classifier and evaluates its performance.
  - **Decision Tree**: Trains a decision tree classifier and assesses its metrics.
- **Visualization**: Provides visualizations for model performance, including ROC curves, confusion matrices, and probability plots.
- **Alert Mechanism**: Implements a function for fault detection with rolling mean probabilities.

## Technologies Used

- Python
- scikit-learn
- pandas
- matplotlib
- seaborn

## Prerequisites

- Python 3.x
- Required Python packages (see `requirements.txt`)

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ABHI-8896/AI-ML-PROJECT.git
   cd AI-ML-PROJECT
    ```
2 . **Install dependencies:**
     ``` bash
     pip install -r requirements.txt
      ````
3.**Run the analysis:**
 ```python analysis.py
 ```
## File Structure

The project has the following structure:
fault-detection-model/ ├── data/ │ └── HeatPump-bc12-s1-088.csv # Dataset used for analysis ├── notebooks/ │ └── analysis.ipynb # Jupyter Notebook for interactive analysis ├── scripts/ │ └── analysis.py # Python script for running the analysis ├── results/ │ └── figures/ # Directory for saving figures and plots ├── requirements.txt # Python package dependencies ├── README.md # Project documentation └── LICENSE # License file

- **data/**: Contains the dataset used in the analysis.
- **notebooks/**: Includes Jupyter notebooks for interactive data exploration and analysis.
- **scripts/**: Python scripts for executing the analysis.
- **results/**: Directory where output figures and plots are saved.
- **requirements.txt**: Lists Python package dependencies.
- **README.md**: Project documentation file.
- **LICENSE**: Contains the license details for the project.
  ## How It Works

1. **Data Preparation**:
   - The dataset is loaded from `data/HeatPump-bc12-s1-088.csv`.
   - Target variable (`TARGET_VAR`) and hidden variables (`HIDDEN_VARS`) are defined.
   - Data is split into features (`X`) and target (`y`).

2. **Exploratory Data Analysis (EDA)**:
   - Data statistics and null values are checked.
   - A correlation heatmap is plotted to understand relationships between features.

3. **Data Scaling and PCA**:
   - Features are standardized using `StandardScaler`.
   - Principal Component Analysis (PCA) is performed to reduce dimensionality and extract the most important features.

4. **Model Training and Evaluation**:
   - **Logistic Regression**: The model is trained on the training set and evaluated using accuracy, mean squared error (MSE), root mean squared error (RMSE), and AUC-ROC.
   - **Random Forest Classifier**: Trained and evaluated with similar metrics as Logistic Regression.
   - **Decision Tree Classifier**: Another model is trained and evaluated using the same metrics.

5. **Performance Metrics**:
   - Confusion matrices and classification reports are generated for each model to assess performance.
   - ROC curves are plotted to visualize the trade-off between the true positive rate and false positive rate.
   - Matthews Correlation Coefficient (MCC) is calculated for a more balanced evaluation of classification performance.

6. **Fault Detection**:
   - A custom function `alert_trigger` is used to detect faults based on rolling mean probabilities and thresholds.
   - Plot functions are provided to visualize predicted probabilities and compare them with actual target values.

7. **Visualization**:
   - Various plots such as ROC curves, scatter plots of predicted probabilities, and confusion matrices are generated to present the results.
   - Fault detection results are visualized to assess the model's performance in a practical context.
## Future Enhancements

- **Advanced Data Preprocessing**:
  - Implement more sophisticated handling of missing values and outliers.
  - Explore feature engineering techniques to improve model performance.

- **Model Improvement**:
  - Experiment with additional classification algorithms such as Support Vector Machines (SVM) and Gradient Boosting Machines (GBM).
  - Optimize hyperparameters using techniques like Grid Search or Random Search.

- **Enhanced Evaluation Metrics**:
  - Incorporate additional performance metrics such as Precision-Recall curves and F1 scores.
  - Perform cross-validation to ensure robust model evaluation.

- **Real-Time Fault Detection**:
  - Develop real-time data streaming and fault detection capabilities.
  - Integrate with real-time data sources to enable live monitoring.

- **User Interface**:
  - Build a user-friendly web interface or dashboard for interactive model monitoring and results visualization.
  - Provide user-configurable options for model parameters and thresholds.

- **Deployment**:
  - Package the application for deployment in a production environment using containerization (e.g., Docker).
  - Develop API endpoints for integrating the model with other systems or services.

- **Documentation and Testing**:
  - Improve documentation to include more detailed explanations and examples.
  - Implement unit tests and integration tests to ensure code quality and functionality.

- **Scalability**:
  - Optimize the codebase for performance to handle larger datasets and more complex models efficiently.
  - Explore distributed computing options for scalability.



