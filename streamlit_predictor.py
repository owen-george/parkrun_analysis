import streamlit as st
import pandas as pd
import pickle
import math
from datetime import date
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# File paths
model_to_use = 'models/to_use/xgb_opt_model.pkl'
scaler_to_use = 'models/to_use/minmax_scaler.pkl'
data_for_model = 'data/clean/cleaned_parkrun_no_outliers.csv'

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
        '15-17': 16,
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
        '70-74': 72,
        '75-79': 77,
        '80-84': 82,
        '85-89': 87
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


def fetch_runner_data(parkrun_id: int, df: pd.DataFrame = None, next_date = None, weather: list = [10, 20, 0]) -> pd.DataFrame:
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
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        
    except SystemExit as e:
        print(e)  # Handle exit and display the message

    
    stats = {}

    # Gets parkrun date
    next_date = next_date or datetime.today().date()
      
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
    last_date = last_date.date() if isinstance(last_date, datetime) else last_date
    first_date = first_date.date() if isinstance(first_date, datetime) else first_date

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
    #print(f"Personal PB: {math.floor(PB)}:{(PB % 1) * 60:02.0f}")  
    #print(f"Ave. time: {math.floor(ave_time)}:{(ave_time % 1) * 60:02.0f}") 
    #print(f"Previous time: {math.floor(prev_time)}:{(prev_time % 1) * 60:02.0f}")
    #print(f"Target time: {math.floor(est_time)}:{(est_time % 1) * 60:02.0f}")
    return est_time, prev_time, PB, ave_time

def confirm_parkrunner(parkrun_id):
    """
    Fetches and returns the parkrunner's name based on their ID.
    
    Parameters:
    - parkrun_id: The parkrunner's ID on parkrun.org.uk.
    
    Returns:
    - str: The name of the parkrunner if found, otherwise None.
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
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None  # Exit the function if there's an error
    
    # Extract and return the parkrunner's name
    name_element = soup.find('h2')
    if name_element:
        return name_element.text.strip()
    else:
        return None  # Return None if the name cannot be found
    
# Load data
st.title("Parkrun Predictor")
st.write("Predict your next parkrun target time based on your historical data, weather conditions, and personal details.")
df = process_parkrun_data_for_models(data_for_model)

# User Inputs
st.header("Enter Your Details")

# Input parkrun ID
parkrun_id = st.number_input("Enter your Parkrun ID:", min_value=1, value=5125087)

# Generate Prediction
if st.button("Check ID"):
    name = confirm_parkrunner(parkrun_id)
    st.success(f"Runner name: {name}")

# Optional: Next parkrun date
next_date = st.date_input("Next Parkrun Date:", value=date.today())
#next_date = datetime.strptime(next_parkrun_date, '%Y-%m-%d')

# Weather Parameters
st.subheader("Expected Weather Conditions")
temp = st.slider("Temperature (°C):", min_value=-10, max_value=40, value=10)
wind = st.slider("Wind Speed (km/h):", min_value=0, max_value=50, value=25)
precipitation = st.slider("Precipitation (mm):", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
weather_list = [temp, wind, precipitation]

# Generate Prediction
if st.button("Calculate Target Time"):
    runner_df = fetch_runner_data(parkrun_id, df, next_date, weather_list)
    prediction, prev_time, PB, ave_time = target_time(runner_df, model_to_use, scaler_to_use)

    st.write(f"Current PB: {math.floor(PB)}:{(PB % 1) * 60:02.0f} minutes")
    st.write(f"Average Time: {math.floor(ave_time)}:{(ave_time % 1) * 60:02.0f}")
    st.write(f"Previous Time: {math.floor(prev_time)}:{(prev_time % 1) * 60:02.0f}")
    st.success(f"Target Time: {math.floor(prediction)}:{(prediction % 1) * 60:02.0f}")

