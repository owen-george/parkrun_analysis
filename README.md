# Parkrun Analysis and Prediction

## Overview
This project analyzes parkrun data to provide insights into individual and event-level performance. Using historical and real-time data, it builds machine learning models to predict target times for upcoming runs. The project aims to assist runners in setting achievable goals by incorporating factors such as previous performance, weather, and event characteristics.

## Project Features
- **Data Collection:** Scrapes historical and real-time data from parkrun websites.
- **Weather Integration:** Fetches weather data (temperature, wind speed, precipitation) using the Open Meteo API.
- **Data Cleaning and Analysis:** Processes raw data to remove outliers, calculate metrics, and generate supplementary data for runners and events.
- **Machine Learning:** Trains models to predict future run times based on features like PB time, average run time, age, gender, and weather.
- **Visualization:** Generates charts and graphs to track trends and highlight key insights.

## Technologies Used
- **Programming Languages:** Python
- **Libraries:** pandas, numpy, matplotlib, scikit-learn, pickle
- **Tools:** Jupyter Notebooks, Open Meteo API

## Getting Started

### Prerequisites
Before you start, ensure you have the following installed:
- Python 3.x
- Anaconda (recommended) or a virtual environment
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pickle`, `requests`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/owen-george/parkrun_analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd parkrun_analysis
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Run Notebooks in Order:**
   - Notebooks are numbered sequentially. Start with `1_scrape_and_load.ipynb` to collect raw parkrun data.
   - Each notebook builds on the previous one, but you can skip directly to `5_Predictor.ipynb` to use the pre-trained model for predictions.

2. **Example Command:**
   - Use the default settings in `1_scrape_and_load.ipynb` to scrape Brighton parkrun data (events 1–826).
   - Add weather data using `2_add_weather.ipynb`.
   - Train a model in `4_ML_time_index.ipynb` or use the pre-trained model directly in `5_Predictor.ipynb`.

3. **API Keys:**
   - The weather data notebook requires an API key for Open Meteo. Ensure you set this up before running.

### File Structure
- **`data/`**: Contains raw and cleaned parkrun data.
  - `raw/`: Unprocessed CSV files.
  - `clean/`: Data ready for analysis.
- **`EDA/`**: Exploratory Data Analysis (EDA) results, including supplementary figures and tables.
- **`functions/`**: Python modules for data scraping, cleaning, and feature engineering.
- **`models/`**: Contains pickled machine learning models and scalers.
- **`figures/`**: Visualizations generated from analysis and modeling.

### Notebooks Overview
1. **`1_scrape_and_load.ipynb`:** Scrapes parkrun data and exports it as a CSV file.
2. **`2_add_weather.ipynb`:** Integrates weather data with the parkrun data.
3. **`3_clean_and_filter.ipynb`:** Cleans, filters, and formats the data.
4. **`4_ML_time_index.ipynb`:** Trains machine learning models to predict target run times.
5. **`5_Predictor.ipynb`:** Predicts the target time for the next parkrun using the pre-trained model.

### Outputs
- Predicted target time for the next parkrun based on key features.
- Cleaned datasets for runners and events.
- Visualizations (e.g., performance trends, weather impacts).

## Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

## Acknowledgments
- parkrun.org.uk for the event data.
- Open Meteo for the weather data.

