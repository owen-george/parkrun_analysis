import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_parkrun_data(x, y, location='brighton', existing_df=None):
    """
    Fetches parkrun data from the given page range and appends it to an existing dataframe.
    
    Parameters:
    - x: int, the starting page number (inclusive).
    - y: int, the ending page number (inclusive).
    - location: The parkrun location as it appears in: https://www.parkrun.org.uk/{location}/results/
            - If nothing is provided it defaults to "brighton"
    - existing_df: pandas DataFrame (optional), existing dataframe to append data to. If None, a new DataFrame is created.
    
    Returns:
    - pandas DataFrame containing the fetched data with duplicates removed.
    """
    y += 1
    
    # Column names for the DataFrame
    columns = ['Date', 'Position', 'Name', 'Runner_id', 'Parkrun_count', 'Gender', 'Age_group', 'Time']
    data = []
    
    # Loop over the range of pages from x to y
    for i in range(x, y):

        url = f'https://www.parkrun.org.uk/{location}/results/{i}/'
        print(f"Processing page {i - x + 1}/{y - x}...            ", end="\r")  # Updates on status
        # Set up headers to avoid blocking by the website
        headers = {
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edge/110.0.1587.56',  # Updated User-Agent for newer browsers
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

            # Extract the parkrun date from the page
            prun_date = soup.select('span.format-date')[0].get_text(strip=True)

            # Loop through each row of results
            for row in soup.select('tr.Results-table-row'):
                if row.get('data-name') != 'Unknown':  # Filter out "Unknown" names
                    position = row.get('data-position')
                    prun_count = row.get('data-runs')
                    name = row.get('data-name')
                    gender = row.get('data-gender')
                    age_group = row.get('data-agegroup')
                    time = row.find('td', class_='Results-table-td--time').find('div', class_='compact').get_text(strip=True).split("\n")[0]
                    runner_id_tag = row.find('a', href=True)
                    runner_id = runner_id_tag['href'].split('/')[3] if runner_id_tag else None
                    
                    # Append the extracted data as a list
                    data.append([prun_date, position, name, runner_id, prun_count, gender, age_group, time])

        except requests.exceptions.RequestException as e:
            # If there's an error loading the page, print the error and continue with the next page
            print(f"Error fetching {url}: {e}")
            continue

    # Convert the collected data into a DataFrame
    new_df = pd.DataFrame(data, columns=columns)

    # If an existing DataFrame is provided, concatenate it with the new data and remove duplicates
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        print("Dataframe updated")
        return combined_df
    else:
        # If no existing DataFrame is provided, return the new DataFrame
        print("Dataframe saved")
        return new_df