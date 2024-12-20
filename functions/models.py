import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random 
import pickle 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, make_scorer
from sklearn.feature_selection import RFE
from sklearn.datasets import make_regression

import xgboost as xgb
from xgboost import XGBRegressor

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

def remove_outliers(df, column):
    """
    Removes outliers from a specified column in a DataFrame using the IQR method, 
    grouped by 'Age_group' and 'Gender'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to check for outliers.
    
    Returns:
    pd.DataFrame: A DataFrame with outliers removed for each group.
    """
    # Group by 'Age_group' and 'Gender'
    grouped_df = df.groupby(['Age_group', 'Gender'])
    
    def remove_outliers_from_group(group):
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each group
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        
        # Calculate the IQR
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the group to exclude outliers
        return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
    
    # Apply the outlier removal function to each group
    filtered_df = grouped_df.apply(remove_outliers_from_group)
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def plot_feature_importance(model, feature_names, title):
    """
    Function to plot feature importance of a trained model, sorted from most to least significant.
    
    Parameters:
    model: Trained model (e.g., RandomForestRegressor)
    feature_names: List of feature names
    title: Title for the plot
    
    Returns:
    feature_importances
    """
    # Extract feature importances
    feature_importances = model.feature_importances_
    
    # Sort features by importance in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names, sorted_feature_importances, color='skyblue')
    plt.xscale('log')  # Use log scale for better visualization
    plt.xlabel('Feature Importance')
    plt.title(f'{title} Feature Importances (Sorted)')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
    plt.tight_layout()
    plt.show()
    
    return sorted_feature_importances

def plot_predicted_vs_actual(y_test, y_pred, title):
    """
    Function to plot scatter plot of predicted vs actual values.
    
    Parameters:
    y_test: Actual target values
    y_pred: Predicted target values
    
    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line for perfect prediction
    
    plt.xlabel('Actual Run Time (minutes)')
    plt.ylabel('Predicted Run Time (minutes)')
    plt.title(f'{title} Scatter Plot: Predicted vs Actual Run Time')
    plt.show()

