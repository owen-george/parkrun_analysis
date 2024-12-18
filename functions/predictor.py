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
import requests
from bs4 import BeautifulSoup
import sys 

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
            start_parkrun_date = input(f"Enter the rough date you started doing parkruns in the format YYYY-MM-DD (Leave blank for an estimated date of {est_start_date}): ")
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
            prev_time = input("Enter your most recent parkrun time in the form 'mm:ss' (Required): ")
            
            # Split the input into mins and secs
            mins, secs = map(int, prev_time.split(':'))  # Convert both parts to integers
            
            # Convert to float minutes and assign to stats
            stats['prev_run_time'] = float(mins + secs / 60)
            break  # Exit the loop if successful
        except ValueError:
            # Handle invalid input
            print("Invalid format. Please enter the time as 'mm:ss' (e.g. 25:30).")
    
    # Get PB (Personal Best)
    while True:
        try:
            # Prompt user for input
            PB = input(f"Enter your parkrun PB in the form 'mm:ss'. "
                       f"Leave blank to use your previous time {stats['prev_run_time']:.1f} mins: ")
            
            if PB.strip():  # If the input is not blank
                # Split the input into mins and secs
                mins, secs = map(int, PB.split(':'))  # Convert both parts to integers
                
                # Convert to float minutes and assign to stats
                stats['prev_PB'] = float(mins + secs / 60)
            else:
                # Use the default value (previous run time)
                stats['prev_PB'] = stats['prev_run_time']
            break  # Exit the loop if successful
        except ValueError:
            # Handle invalid input
            print("Invalid format. Please enter the PB as 'mm:ss' (e.g. 25:30) or leave blank.")
    
    # Get average run time
    while True:
        try:
            # Prompt user for input
            ave_time = input(f"Enter your average parkrun time in the form 'mm:ss'. "
                             f"Leave blank to use your previous time {stats['prev_run_time']:.1f} mins: ")
            
            if ave_time.strip():  # If the input is not blank
                # Split the input into mins and secs
                mins, secs = map(int, ave_time.split(':'))  # Convert both parts to integers
                
                # Convert to float minutes and assign to stats
                stats['avg_prev_run_times'] = float(mins + secs / 60)
            else:
                # Use the default value (previous run time)
                stats['avg_prev_run_times'] = stats['prev_run_time']
            break  # Exit the loop if successful
        except ValueError:
            # Handle invalid input
            print("Invalid format. Please enter the time as 'mm:ss' (e.g. 25:30) or leave blank.") 
    
    
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

def confirm_parkrunner(soup):
    """
    Confirms the parkrunner by showing the name and asking for input.
    
    Parameters:
    - soup: BeautifulSoup object of the page.
    """
    name_list = soup.find('h2').text.strip().split()
    name = " ".join(name_list)
    name_test = input(f"Found name: {name}. Press enter to continue, input a different id to try again, or 'n' to exit: ")

    if name_test == "":
        return  # Continue with the current flow
    elif name_test.isdigit():
        new_parkrun_id = int(name_test)
        fetch_runner_data(new_parkrun_id)  # Restart with the new ID
    elif name_test.lower() == 'n':
        raise SystemExit("Process cancelled")  # Exit the entire function
    else:
        print("Invalid input. Try again.")
        confirm_parkrunner(soup)  # Recurse to retry the input

def fetch_runner_data(parkrun_id: int, df: pd.DataFrame = None, next_date: str = None, weather: list = None) -> pd.DataFrame:
    """
    Fetches parkrun data for the given id and saves as a dataframe.
    
    Parameters:
    - parkrun_id: int, the parkrunner id as an integer.
    - df: pd.DataFrame, The dataframe of the parkrun location stats.
    - next_date: str, Date of the next parkrun in the format YYYY-MM-DD. If blank, uses today's date.
    - weather: list, a list of the estimated temperature (C), windspeed (km/h), and precipitation (mm). If blank, uses the average values for the location.

    Outputs:
    - A dataframe of the runner's stats
    """
  
    # The URL for the complete runner stats
    url = f'https://www.parkrun.org.uk/parkrunner/{parkrun_id}/all/'
    
    # Set up headers to avoid blocking by the website
    headers = {
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edge/110.0.1587.56', 
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
        'Cache-Control': 'max-age=0',
        'TE': 'Trailers',
        'Pragma': 'no-cache',
        'Referer': 'https://www.parkrun.org.uk/',
        'Origin': 'https://www.parkrun.org.uk',
        'X-Requested-With': 'XMLHttpRequest',
        'If-None-Match': 'W/"f0b3eb46c6c7e1f04161c38a1f041f4"'
    }

    try:
        # Request the page content
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for any HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")

        # Call the function to confirm the parkrunner details
        confirm_parkrunner(soup)
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        
    except SystemExit as e:
        print(e)  # Handle exit and display the message

    
    stats = {}

    #Gets parkrun date
    next_date = next_date or datetime.today().strftime('%Y-%m-%d')

    
    try:
        next_date = datetime.strptime(next_date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Using today's date.")
        next_date = datetime.today().strftime('%Y-%m-%d')
        
   # Weather handling with validation
    if weather is None:
        if df.empty:
            print("Dataframe is empty. Applied default weather values.")
            temp, wind, precip = 10, 20, 0
        else:
            temp = df['temperature'].median()
            wind = df['windspeed'].median()
            precip = 0
    else:
        if len(weather) == 3:
            temp = float(weather[0])
            wind = float(weather[1])
            precip = float(weather[2])
        else:
            print("Weather list must contain exactly 3 values (temp, wind, precip). Using default values.")
            if df.empty:
                temp, wind, precip = 10, 20, 0
            else:
                temp = df['temperature'].median()
                wind = df['windspeed'].median()
                precip = 0

    stats['temperature'] = temp
    stats['windspeed'] = wind
    stats['precipitation'] = precip
    
    # Gets age category
    age_cat = soup.find('p').text.strip().split()[-1]

    # Converts to age (approx) and gender
    age = int(age_cat[-2:])-2
    gender = age_cat[1]

    stats['Age_group_numeric'] = age
    
    if gender.lower() == 'm':
        stats['Male'] = 1
    elif gender.lower() == 'f':
        stats['Male'] = 0
    else: stats['Male'] = 0.5
        
    table = soup.find_all('table')

    dates = []
    times = []
    
    for row in table[2].find_all('tr'):
        data_point = row.find_all('td')
        if len(data_point) > 4:
            date = data_point[1].find('span', class_="format-date")
            time = data_point[4]
    
            if date:
                dates.append(date.text.strip())
            if time:
                times.append(time.text.strip())
                
    # Saves as a dataframe
    date_time_df = pd.DataFrame({'Date': dates, 'Time': times})

    # Converts columns into appropriate format
    date_time_df['Date'] = pd.to_datetime(date_time_df['Date'])
    date_time_df['Time'] = date_time_df['Time'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)
    
    #Gets stats from runner dataframe
    last_date = date_time_df['Date'].max()
    first_date = date_time_df['Date'].min()
    PB = date_time_df['Time'].min()
    last_time = date_time_df['Time'].iloc[0]
    ave_time = date_time_df['Time'].mean()
    instance = date_time_df['Time'].count()+1

    # Adds stats to dictionary
    stats['Appearance_Instance'] = instance
    stats['prev_PB'] = PB
    stats['avg_prev_run_times'] = ave_time
    stats['prev_run_time'] = last_time
    
    # Calculates missing stats
    stats['Days_since_last_parkrun'] = (next_date - last_date).days
    stats['Days_since_first_parkrun'] = (next_date - first_date).days

    # Return as a single-row dataframe
    stats_df = pd.DataFrame([stats])
    
    # Ensures everything is in correct format
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

    prev_time = user_stats['prev_run_time'][0]
    PB = user_stats['prev_PB'][0]
    ave_time = user_stats['avg_prev_run_times'][0]
    
    # Print the estimated time in minutes and seconds
    print(f"Personal PB: {math.floor(PB)}:{(PB % 1) * 60:02.0f}")  
    print(f"Ave. time: {math.floor(ave_time)}:{(ave_time % 1) * 60:02.0f}") 
    print(f"Previous time: {math.floor(prev_time)}:{(prev_time % 1) * 60:02.0f}")
    print(f"Target time: {math.floor(est_time)}:{(est_time % 1) * 60:02.0f}")
    return est_time