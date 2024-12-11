import requests
import pandas as pd
import time
import numpy as np

def fetch_weather(date, latitude=50.8425, longitude=-0.1718):
    """
    Fetches weather data (temperature, windspeed, precipitation) at 9 AM for a given date 
    from the Open Meteo API.

    Parameters:
    - date (str): The date for which to fetch the weather data, in 'YYYY-MM-DD' format.
    - latitude (float, optional): The latitude of the location. Default is 50.8425 (Brighton, UK).
    - longitude (float, optional): The longitude of the location. Default is -0.1718 (Brighton, UK).

    Returns:
    - dict: A dictionary containing the weather data for the specified date, including:
        - 'date' (str): The date for which the weather was fetched.
        - 'temperature' (float or None): The temperature in degrees Celsius at 9 AM, or None if unavailable.
        - 'windspeed' (float or None): The windspeed in meters per second at 9 AM, or None if unavailable.
        - 'precipitation' (float or None): The precipitation in millimeters at 9 AM, or None if unavailable.

    If the API call fails or the data for 9 AM is not available, the function will return None for the weather values.
    """
   
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,windspeed_10m,precipitation",
        "timezone": "Europe/London"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            # Find the index of 9 AM data
            index = data["hourly"]["time"].index(f"{date}T09:00")
            return {
                "date": date,
                "temperature": data["hourly"]["temperature_2m"][index],
                "windspeed": data["hourly"]["windspeed_10m"][index],
                "precipitation": data["hourly"]["precipitation"][index]
            }
        except ValueError:
            return {"date": date, "temperature": None, "windspeed": None, "precipitation": None}
    else:
        print(f"Failed to fetch data for {date}: {response.status_code}")
        return {"date": date, "temperature": None, "windspeed": None, "precipitation": None}