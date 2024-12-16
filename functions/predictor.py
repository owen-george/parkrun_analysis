import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random 
import pickle
from datetime import datetime
import math 

from datetime import datetime

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

def process_parkrun_data_for_models(filepath='data/clean/cleaned_parkrun_no_outliers.csv'):
    """
    Preprocess the parkrun data from a CSV file.

    Parameters:
    - df: The cleaned dataframe

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Default age group mapping if none is provided
    df = pd.read_csv(filepath)
    age_group_map = {
        '18-19': 19,
        '20-24': 22,
        '25-29': 27,
        '30-34': 32,
        '35-39': 37,
        '40-44': 42,
        '45-49': 47,
        '50-54': 52,
        '55-59': 57,
        '60-64': 62,
        '65-69': 67,
        '70-74': 72
    }
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Map 'Age_group' to numeric values
    df['Age_group_numeric'] = df['Age_group'].map(age_group_map)
    
    # Calculate the first parkrun date for each runner
    df['first_parkrun_date'] = df.groupby('Runner_id')['Date'].transform('min')
    
    # Calculate days since the first parkrun
    df['Days_since_first_parkrun'] = (df['Date'] - df['first_parkrun_date']).dt.days
    
    # Map gender to binary values
    df['Male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # Drop rows with missing values
    df = df.dropna()
    df['time_change_index'] = df['Time_in_minutes'] / df['prev_run_time']
    return df

def user_input(df):
    """
    Function to collect user input for the parameters required for the stats dataframe.
    If the user leaves any blank, it will use the average from the dataframe for numeric fields.
    Returns a single-row dataframe.
    """
    # Calculate average values from the dataframe
    avg_values = {
        'temperature': df['temperature'].median(),
        'windspeed': df['windspeed'].median(),
        'precipitation': df[df['precipitation']>0]['precipitation'].median(),
        'Appearance_Instance': df['Appearance_Instance'].median(),
        'Days_since_last_parkrun': df['Days_since_last_parkrun'].median(),
        'prev_run_time': df['prev_run_time'].mean(),
        'prev_PB': df['prev_PB'].mean(),
        'avg_prev_run_times': df['avg_prev_run_times'].mean(),
        'Age_group_numeric': df['Age_group_numeric'].median(),
        'Days_since_first_parkrun': df['Days_since_first_parkrun'].median(),
        'Male': df['Male'].mean(),
    }

    stats = {}

    # Get temperature (or default to average)
    while True:
        try:
            temp = input(f"Enter expected temperature in (°C). (Leave blank for a default value of {avg_values['temperature']}): ")
            stats['temperature'] = float(temp) if temp else avg_values['temperature']
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a numeric value for temperature.")
    
    # Get windspeed (or default to average)
    while True:
        try:
            wind_speed = input(f"Enter expected windspeed in km/h (Leave blank for a default value of {avg_values['windspeed']}): ")
            stats['windspeed'] = float(wind_speed) if wind_speed else avg_values['windspeed']
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a numeric value for windspeed.")
    
    while True:
        try:
            precip_input = input("Is it likely to rain? (y/n): ").lower()
            if precip_input == "y":
                # If user says yes, prompt for expected rainfall
                precip_amt = input(f"Enter expected rainfall for the hour in mm (Leave blank for a default value of {avg_values['precipitation']}): ")
                
                # If input is not blank, use the provided value; otherwise, use default value
                stats['precipitation'] = float(precip_amt) if precip_amt else avg_values['precipitation']
            elif precip_input == "n":
                # If user says no, set precipitation to 0
                stats['precipitation'] = 0.0
            else:
                # Handle invalid input (not 'y' or 'n')
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
                continue  # Prompt again if input is invalid
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a numeric value for precipitation.")
    
    # Get appearance instance
    while True:
        try:
            instance = input(f"How many parkruns have you run previously? (Leave blank for a default value of {avg_values['Appearance_Instance'] - 1}): ")
            stats['Appearance_Instance'] = int(instance) + 1 if instance else avg_values['Appearance_Instance']
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a numeric value for the number of parkruns.")
    
    # Get Days_since_last_parkrun and Days_since_first_parkrun based on user input
    while True:
        try:
            planned_parkrun_date = input(f"Enter the planned parkrun date (leave blank for today, format YYYY-MM-DD): ")
            if not planned_parkrun_date:
                planned_parkrun_date = datetime.today().strftime('%Y-%m-%d')
            # Try to parse the date
            planned_parkrun_date = datetime.strptime(planned_parkrun_date, '%Y-%m-%d')
            break  # Exit loop if valid input is given
        except ValueError:
            print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")
    
    while True:
        try:
            previous_parkrun = input("Enter the date of your last parkrun (format YYYY-MM-DD): ")
            # Try to parse the previous parkrun date
            previous_parkrun_date = datetime.strptime(previous_parkrun, '%Y-%m-%d')
            break  # Exit loop if valid input is given
        except ValueError:
            print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")
    
    # Calculate the days since the last parkrun
    days_since_last_parkrun = (planned_parkrun_date - previous_parkrun_date).days
    
    # Update stats with average days since first parkrun
    avg_days_since_first_parkrun = avg_values['Days_since_first_parkrun']
    
    # Calculate estimated start date
    est_start_date = (planned_parkrun_date - pd.Timedelta(days=avg_days_since_first_parkrun)).strftime('%Y-%m-%d')
    
    while True:
        try:
            start_parkrun_date = input(f"Enter the rough date you started doing parkruns (leave blank for {est_start_date}, format YYYY-MM-DD): ")
            if not start_parkrun_date:
                start_parkrun_date = est_start_date
            # Try to parse the start parkrun date
            start_parkrun_date = datetime.strptime(start_parkrun_date, '%Y-%m-%d')
            break  # Exit loop if valid input is given
        except ValueError:
            print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")
    
    # Calculate days since the first parkrun
    days_since_first_parkrun = (planned_parkrun_date - start_parkrun_date).days
    
    # Update the stats dictionary
    stats['Days_since_first_parkrun'] = float(days_since_first_parkrun)
    stats['Days_since_last_parkrun'] = float(days_since_last_parkrun)

    # Get previous run time
    while True:
        try:
            # Prompt user for input
            prev_time = input("Enter your most recent parkrun time in the form 'mins, secs' (Required): ")
            
            # Split the input into mins and secs
            mins, secs = map(int, prev_time.split(','))  # Convert both parts to integers
            
            # Convert to float minutes and assign to stats
            stats['prev_run_time'] = float(mins + secs / 60)
            break  # Exit the loop if successful
        except ValueError:
            # Handle invalid input
            print("Invalid format. Please enter the time as 'mins, secs' (e.g., 25, 30).")
    
    # Get PB (Personal Best)
    while True:
        try:
            # Prompt user for input
            PB = input(f"Enter your previous PB in the form 'mins, secs'. "
                       f"Leave blank to use your previous time {stats['prev_run_time']:.1f} mins: ")
            
            if PB.strip():  # If the input is not blank
                # Split the input into mins and secs
                mins, secs = map(int, PB.split(','))  # Convert both parts to integers
                
                # Convert to float minutes and assign to stats
                stats['prev_PB'] = float(mins + secs / 60)
            else:
                # Use the default value (previous run time)
                stats['prev_PB'] = stats['prev_run_time']
            break  # Exit the loop if successful
        except ValueError:
            # Handle invalid input
            print("Invalid format. Please enter the PB as 'mins, secs' (e.g., 25, 30) or leave blank.")
    
    # Get average run time
    while True:
        try:
            # Prompt user for input
            ave_time = input(f"Enter your average parkrun time in the form 'mins, secs'. "
                             f"Leave blank to use your previous time {stats['prev_run_time']:.1f} mins: ")
            
            if ave_time.strip():  # If the input is not blank
                # Split the input into mins and secs
                mins, secs = map(int, ave_time.split(','))  # Convert both parts to integers
                
                # Convert to float minutes and assign to stats
                stats['avg_prev_run_times'] = float(mins + secs / 60)
            else:
                # Use the default value (previous run time)
                stats['avg_prev_run_times'] = stats['prev_run_time']
            break  # Exit the loop if successful
        except ValueError:
            # Handle invalid input
            print("Invalid format. Please enter the time as 'mins, secs' (e.g., 25, 30) or leave blank.") 
    
    
    # Get Age_group_numeric
    while True:
        try:
            age = input(f"Enter your age: ")
            stats['Age_group_numeric'] = float(age) if age else avg_values['Age_group_numeric']
            break  # Exit the loop if valid input is provided
        except ValueError:
            print("Invalid age. Please enter a valid number.")
    
    # Get gender and set Male to 1 if male, 0 if female, or use average from dataframe
    while True:
        gender = input("Enter your gender (m/f) or leave blank: ").lower()
        if gender == 'm':
            stats['Male'] = 1
            break
        elif gender == 'f':
            stats['Male'] = 0
            break
        elif gender == '':
            stats['Male'] = avg_values['Male']
            break
        else:
            print("Invalid input. Please enter 'm' for male or 'f' for female.")

    # Return as a single-row dataframe
    stats_df = pd.DataFrame([stats])

    user_stats = stats_df.astype({
        'temperature': 'float64',
        'windspeed': 'float64',
        'precipitation': 'float64',
        'Appearance_Instance': 'int64',
        'Days_since_last_parkrun': 'float64',
        'prev_run_time': 'float64',
        'prev_PB': 'float64',
        'avg_prev_run_times': 'float64',
        'Age_group_numeric': 'float64',
        'Days_since_first_parkrun': 'float64',
        'Male': 'int64',
    })

    user_stats = user_stats[['temperature', 'windspeed', 'precipitation',
           'Appearance_Instance', 'Days_since_last_parkrun',
           'prev_run_time', 'prev_PB', 'avg_prev_run_times',
           'Age_group_numeric', 'Days_since_first_parkrun',
           'Male']]    

    print("")
    print("Data added successfully")
    
    return user_stats

def target_time(user_stats,
                        df,
                        model_file_path='models/to_use/xgb_opt_model.pkl',
                        scaler_file_path='models/to_use/minmax_scaler.pkl'):
    """
    Predict the target time based on user statistics and a pre-trained model.

    Parameters:
    - user_stats (pd.DataFrame): DataFrame containing the user's stats (e.g., prev_run_time, etc.).
    - df (pd.DataFrame): The original dataset to match features (not used in prediction here, but may be helpful).
    - model_file_path (str): Path to the saved model file.
    - scaler_file_path (str): Path to the saved scaler file.

    Returns:
    - None: Prints the target time.
    """
    
    # Load the model from the pickle file
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)

    # Load the scaler from the pickle file
    with open(scaler_file_path, 'rb') as file:
        scaler = pickle.load(file)

    # Normalize the user stats
    norm_stats = scaler.transform(user_stats)

    # Convert the normalized data back to DataFrame (if necessary)
    norm_stats_df = pd.DataFrame(norm_stats, columns=user_stats.columns)

    # Make the prediction using the model
    prediction = model.predict(norm_stats_df)

    # Calculate the estimated time
    est_time = prediction[0] * user_stats['prev_run_time'][0]

    # Print the estimated time in minutes and seconds
    print(f"Target time: {math.floor(est_time)}m{(est_time % 1) * 60:.0f}s")
    return est_time