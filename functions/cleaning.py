import pandas as pd

def preprocess_parkrun_data(df):
    """
    Preprocesses a parkrun DataFrame to include datetime conversions, 
    time calculations, and appearance-based metrics.

    Parameters:
    df (pd.DataFrame): Original DataFrame with parkrun data.

    Returns:
    pd.DataFrame: Processed DataFrame with new calculated columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df2 = df.copy()

    # Convert the Date column to datetime
    df2['Date'] = pd.to_datetime(df2['Date'])

    # Convert the Time column to timedelta
    df2['Time'] = df2['Time'].apply(lambda x: '00:' + x if len(x.split(':')) == 2 else x)
    df2['Time'] = pd.to_timedelta(df2['Time'])

    # Calculate time in seconds and minutes
    df2['Time_in_seconds'] = df2['Time'].apply(lambda x: pd.to_timedelta(x).total_seconds())
    df2['Time_in_minutes'] = df2['Time_in_seconds'] / 60
    df2['Time_in_minutes'] = df2['Time_in_minutes'].round(2)

    # Calculate total appearances and appearance instance for each runner
    df2['Total_Appearances'] = df2.groupby('Runner_id')['Runner_id'].transform('count')
    df2['Appearance_Instance'] = df2.groupby('Runner_id').cumcount() + 1

    # Sort by 'Runner_id' and 'Date'
    df2 = df2.sort_values(by=['Runner_id', 'Date'])

    # Calculate the difference in days from the previous row's 'Date' (for each runner)
    df2['Days_since_last_parkrun'] = (
        df2.groupby('Runner_id')['Date'].shift(0) - df2.groupby('Runner_id')['Date'].shift(1)
    ).dt.days

    # Handle missing values for first appearances
    df2.loc[df2['Appearance_Instance'] == 1, 'Days_since_last_parkrun'] = None

    # Return the DataFrame sorted by its original index
    return df2.sort_index()

def add_total_event_runners(df):
    """
    Adds the total number of runners per event (by date) to the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Updated dataframe with a 'Total_event_runners' column.
    """
    pos_df = df.groupby('Date')['Position'].max().reset_index()
    pos_df = pos_df.rename(columns={'Position': 'Total_event_runners'})
    return pd.merge(df, pos_df, on='Date', how='left')

def filter_and_clean_age_groups(df):
    """
    Filters and cleans the Age_group column to retain relevant age groups.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Filtered and cleaned dataframe.
    """
   
    age_list = ['SM20-24', 'VM40-44', 'VM50-54', 'VW35-39', 'VW40-44', 'VM35-39',
       'VM45-49', 'SW25-29', 'SW30-34', 'SM30-34', 'VM55-59', 'VW45-49',
       'SM25-29', 'SW20-24', 'SM18-19',
       'VM60-64', 'VM70-74', 'VW50-54', 'VW55-59', 'VW70-74',
       'VM65-69', 'VW60-64', 'SW18-19',
       'VW65-69']
    
    filt_df = df[df['Age_group'].isin(age_list)].copy()
    filt_df['Age_group'] = filt_df['Age_group'].str[-5:]
    return filt_df

def filter_by_appearance_ratio(df, fraction_threshold=0.5):
    """
    Filters runners with over a threshold of runs in the specified location.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    fraction_threshold : The minimum acceptable threshold of share of all parkruns in the location (default: 0.5)

    Returns:
    pd.DataFrame: Filtered dataframe.
    """
    return df[df['Total_Appearances'] >= (df['Parkrun_count'] * fraction_threshold)]

def calculate_personal_bests(df):
    """
    Adds personal best (PB) times to the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Updated dataframe with PB times.
    """
    pb_df = df.groupby('Runner_id')['Time_in_minutes'].min().reset_index()
    pb_df = pb_df.rename(columns={'Time_in_minutes': 'PB_mins'})
    return pd.merge(df, pb_df, on='Runner_id', how='left')

def calculate_average_times(df):
    """
    Adds average times for each runner to the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Updated dataframe with average times.
    """
    ave_df = df.groupby('Runner_id')['Time_in_minutes'].mean().reset_index()
    ave_df = ave_df.rename(columns={'Time_in_minutes': 'ave_mins'})
    return pd.merge(df, ave_df, on='Runner_id', how='left')

def calculate_previous_metrics(df):
    """
    Adds cumulative and average metrics of previous runs to the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Updated dataframe with cumulative and average metrics.
    """
    # Sort the dataframe for correct chronological calculations
    df = df.sort_values(by=['Runner_id', 'Date'])

    # Shift Time_in_minutes to exclude the current run for PB calculation
    df['prev_run_time'] = df.groupby('Runner_id')['Time_in_minutes'].shift(1)
    df['prev_PB'] = df.groupby('Runner_id')['prev_run_time'].cummin()

    # Add cumulative sum and count for previous runs
    df['prev_run_time_cumsum'] = df.groupby('Runner_id')['prev_run_time'].cumsum()
    df['prev_run_count'] = df.groupby('Runner_id')['prev_run_time'].cumcount()

    # Calculate average of previous runs
    df['avg_prev_run_time'] = df['prev_run_time_cumsum'] / df['prev_run_count'].replace(0, 1)

    # Drop helper columns
    df = df.drop(columns=['prev_run_time', 'prev_run_time_cumsum', 'prev_run_count'])

    # Return to the original order
    return df.sort_index()

def calculated_columns(df):
    '''
    Calculates and adds a new column 'Position_score' to the DataFrame.
    
    The 'Position_score' is calculated as the normalized position of a runner
    in the event, where the first position is 0, and the last position is 1.  

    Parameters:
    df (pd.DataFrame): The input DataFrame, which must contain the columns 
                       'Position' and 'Total_event_runners'.
    
    Returns:
    pd.DataFrame: The DataFrame with the newly added 'Position_score' column.    
    '''
    
    df['Position_score'] = (df['Position']-1)/(df['Total_event_runners']-1)
    
    return df

def filter_runners_with_minimum_runs(df, min_runs=2):
    """
    Filters the dataframe to only include runners with at least a minimum number of runs.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    min_runs (int): Minimum number of total appearances required. (Default: 2)

    Returns:
    pd.DataFrame: Filtered dataframe.
    """
    return df[df['Appearance_Instance'] >= min_runs]

def filter_columns(df):
    """
    Filters the dataframe to remove some of the unnecessary columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Filtered dataframe.
    """

    df = df.drop(columns=['Time', 'Time_in_seconds'])
    
    return df

def reorder_columns(df):
    """
    Reorders the columns of the DataFrame into a specified order.
    
    Ignores columns not present in the DataFrame, reordering is performed only on existing columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to reorder.

    Returns:
    pd.DataFrame: The DataFrame with columns reordered.
    """    
    # Define the desired column order
    column_order = [
        'Date', 'Position', 'Position_score', 'Name', 'Runner_id', 
        'Parkrun_count', 'Gender', 'Age_group', 'Time_in_minutes', 
        'temperature', 'windspeed', 'precipitation', 'Total_Appearances', 
        'Appearance_Instance', 'Days_since_last_parkrun', 
        'Total_event_runners', 'PB_mins', 'ave_mins', 'prev_PB', 
        'avg_prev_run_time'
    ]
    
    # Reorder columns, keeping only those present in the DataFrame
    reordered_df = df[[col for col in column_order if col in df.columns]]
    
    return reordered_df

def process_parkrun_data(df):
    """
    End-to-end processing of parkrun data with all transformations applied.

    Parameters:
    df (pd.DataFrame): Input dataframe with raw parkrun data.

    Returns:
    pd.DataFrame: Fully processed dataframe.
    """
    df = preprocess_parkrun_data(df)
    df = add_total_event_runners(df)
    df = filter_and_clean_age_groups(df)
    df = filter_by_appearance_ratio(df)
    df = calculate_personal_bests(df)
    df = calculate_average_times(df)
    df = calculate_previous_metrics(df)
    df = calculated_columns(df)
    df = filter_runners_with_minimum_runs(df)
    df = filter_columns(df)
    df = reorder_columns(df)
    df = df.dropna()
    return df