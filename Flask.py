from flask import Flask, send_file
from flask_cors import CORS
import io
import sqlite3
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

@app.route('/plot')
def serve_plot():

    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    cursor.execute("SELECT serialized_df1 FROM data_plot5 ORDER BY id DESC LIMIT 1")
    serialized_df_data = cursor.fetchone()

    conn.close()

    if serialized_df_data:

        serialized_df = serialized_df_data[0]
        monthly_df = pd.read_pickle(io.BytesIO(serialized_df))

        # plt.figure(figsize=(8, 5))
        monthly_df['Close'].plot(title='Monthly Stock Close Prices 1791-2013')
        plt.ylabel('Close', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.legend()

        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        plt.close()

        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/stationary_plot')
def stationary_plot():

    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    cursor.execute("SELECT serialized_df2 FROM data_plot5 ORDER BY id DESC LIMIT 1")
    serialized_df_data = cursor.fetchone()

    conn.close()

    if serialized_df_data:

        serialized_df = serialized_df_data[0]
        diff_rol_mean = pd.read_pickle(io.BytesIO(serialized_df))

        diff_rol_mean['Close'].plot(title='Stationary Monthly Stock Close Prices 1791-2013', figsize=(8, 5))
        plt.ylabel('Close', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.legend()

        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        plt.close()

        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/arima_plot')
def arima_plot():

    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    cursor.execute("SELECT train_data, test_data, arima_residuals, forecast_values, rmse FROM arima_ploted ORDER BY id DESC LIMIT 1")
    plot_data = cursor.fetchone()

    conn.close()

    if plot_data:

        train_data = pickle.loads(plot_data[0])
        test_data = pickle.loads(plot_data[1])
        arima_residuals = pickle.loads(plot_data[2])
        forecast_values = pd.read_pickle(io.BytesIO(plot_data[3]))
        rmse = plot_data[4]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(train_data.index, train_data.values, label='Training Data', color='blue')
        ax1.plot(test_data.index, test_data.values, label='Testing Data', color='green')
        ax1.plot(test_data.index, forecast_values.values, label='Forecasted Values', color='orange')
        ax1.set_title('Training, Testing, and Forecasted Values')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Values')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(test_data.index, arima_residuals, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Residuals and RMSE')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Values')
        ax2.legend()
        ax2.grid(True)

        ax2.text(0.5, 0.95, f'RMSE: {rmse:.2f}', horizontalalignment='center', verticalalignment='top', transform=ax2.transAxes, fontsize=20, color='blue')

        plt.tight_layout()

        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        plt.close()

        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/arima_forecast')
def arima_forecast():

    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT diff_rol_mean, forecast_index, forecast_values FROM arima_forecastData ORDER BY id DESC LIMIT 1")
    forecast_data = cursor.fetchone()

    conn.close()

    if forecast_data:

        diff_rol_mean = pd.read_pickle(io.BytesIO(forecast_data[0]))
        forecast_index_str = forecast_data[1].split(',')
        forecast_values = pd.read_pickle(io.BytesIO(forecast_data[2]))

        forecast_index = pd.to_datetime(forecast_index_str)

        plt.figure(figsize=(10, 5))

        plt.plot(diff_rol_mean.index, diff_rol_mean['Close'].values, label='Original Data', color='blue')

        plt.plot(forecast_index, forecast_values, label='Forecasted Values', color='orange')

        plt.title('ARIMA Model: Original Data and Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        plt.close()

        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No forecast data found in the database."

@app.route('/ann_plot')
def ann_plot():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Step 2: Execute a query to retrieve the necessary data from the table
    cursor.execute("SELECT y_test, y_pred, residuals, rmse FROM ann_ploted ORDER BY id DESC LIMIT 1")
    plot_data = cursor.fetchone()

    # Step 3: Close the database connection
    conn.close()

    if plot_data:
        # Step 4: Deserialize the data
        y_test = pickle.loads(plot_data[0])
        y_pred = pickle.loads(plot_data[1])
        residuals = pickle.loads(plot_data[2])
        rmse = plot_data[3]

        # Step 5: Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting actual vs predicted closing prices
        ax1.plot(y_test, label='Actual')
        ax1.plot(y_pred, label='Predicted', linestyle='--')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Closing Price')
        ax1.set_title('Actual vs Predicted Closing Prices')
        ax1.legend()

        # Plotting residuals
        ax2.plot(range(len(residuals)), residuals, color='blue')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('Residuals Plot')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Residuals')
        ax2.set_ylim(-0.6, 0.6)

        # Add RMSE value to the plot
        ax2.text(0.5, 0.95, f'RMSE: {rmse:.2f}', horizontalalignment='center', verticalalignment='top', transform=ax2.transAxes, fontsize=20, color='blue')

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/ann_forecast')
def ann_forecast():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Step 2: Execute a query to retrieve the necessary data from the table
    cursor.execute("SELECT future_dates, diff_rol_mean, future_predictions FROM ann_forecasted ORDER BY id DESC LIMIT 1")
    forecast_data = cursor.fetchone()

    # Step 3: Close the database connection
    conn.close()

    if forecast_data:
        # Step 4: Deserialize the data
        future_dates_str = forecast_data[0].split(',')
        future_dates = pd.to_datetime(future_dates_str)
        diff_rol_mean = pickle.loads(forecast_data[1])
        future_predictions = pickle.loads(forecast_data[2])

        # Step 5: Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(diff_rol_mean.index, diff_rol_mean['Close'], label='Historical Data')
        plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Historical and Future Predicted Closing Prices')
        plt.legend()

        # Save the plot to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/sarima_plot')
def sarima_plot():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Step 2: Execute a query to retrieve the necessary data from the table
    cursor.execute("SELECT train_data, test_data, forecast, sarima_residuals, rmse FROM sarima_plots ORDER BY id DESC LIMIT 1")
    plot_data = cursor.fetchone()

    # Step 3: Close the database connection
    conn.close()

    if plot_data:
        # Step 4: Deserialize the data
        train_data = pickle.loads(plot_data[0])
        test_data = pickle.loads(plot_data[1])
        forecast = pickle.loads(plot_data[2])
        sarima_residuals = pickle.loads(plot_data[3])
        rmse = plot_data[4]

        # Step 5: Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plotting training, testing, and forecasted values
        ax1.plot(train_data.index, train_data.values, label='Training Data', color='blue')
        ax1.plot(test_data.index, test_data.values, label='Testing Data', color='green')
        ax1.plot(test_data.index, forecast.values, label='Forecasted Values', color='orange')
        ax1.set_title('Training, Testing, and Forecasted Values')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Values')
        ax1.legend()
        ax1.grid(True)

        # Plot residuals and RMSE
        ax2.plot(test_data.index, sarima_residuals, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Residuals and RMSE')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Values')
        ax2.legend()
        ax2.grid(True)

        # Add RMSE value to the plot
        ax2.text(0.5, 0.95, f'RMSE: {rmse:.2f}', horizontalalignment='center', verticalalignment='top', transform=ax2.transAxes, fontsize=20, color='blue')

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/sarima_forecast')
def sarima_forecast():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Step 2: Execute a query to retrieve the necessary data from the table
    cursor.execute("SELECT diff_rol_mean, forecast_index, forecast FROM sarima_forecastData ORDER BY id DESC LIMIT 1")
    plot_data = cursor.fetchone()

    # Step 3: Close the database connection
    conn.close()

    if plot_data:
        # Step 4: Deserialize the data
        diff_rol_mean = pickle.loads(plot_data[0])
        forecast_index = pickle.loads(plot_data[1])
        forecast = pickle.loads(plot_data[2])

        # Step 5: Create the plot
        plt.figure(figsize=(10, 5))

        # Plotting the entire dataset
        plt.plot(diff_rol_mean.index, diff_rol_mean['Close'].values, label='Original Data', color='blue')

        # Plotting forecasted values
        plt.plot(forecast_index, forecast, label='Forecasted Values', color='orange')

        # Adding labels and legend
        plt.title('SARIMA Model: Original Data and Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        # Save the plot to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/ets_plot')
def ets_plot():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Step 2: Execute a query to retrieve the necessary data from the table
    cursor.execute("SELECT train_data, test_data, ets_forecast, ets_residuals, rmse FROM ets_plots ORDER BY id DESC LIMIT 1")
    plot_data = cursor.fetchone()

    # Step 3: Close the database connection
    conn.close()

    if plot_data:
        # Step 4: Deserialize the data
        train_data = pickle.loads(plot_data[0])
        test_data = pickle.loads(plot_data[1])
        ets_forecast = pickle.loads(plot_data[2])
        ets_residuals = pickle.loads(plot_data[3])
        rmse = plot_data[4]

        # Step 5: Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plotting training, testing, and forecasted values
        ax1.plot(train_data.index, train_data.values, label='Training Data', color='blue')
        ax1.plot(test_data.index, test_data.values, label='Testing Data', color='green')
        ax1.plot(test_data.index, ets_forecast.values, label='Forecasted Values', color='orange')
        ax1.set_title('Training, Testing, and Forecasted Values')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Values')
        ax1.legend()
        ax1.grid(True)

        # Plot residuals and RMSE
        ax2.plot(test_data.index, ets_residuals, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Residuals and RMSE')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Values')
        ax2.grid(True)

        # Add RMSE value to the plot
        ax2.text(0.5, 0.95, f'RMSE: {rmse:.2f}', horizontalalignment='center', verticalalignment='top', transform=ax2.transAxes, fontsize=20, color='blue')

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/ets_forecast')
def ets_forecast():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Step 2: Execute a query to retrieve the necessary data from the table
    cursor.execute("SELECT diff_rol_mean, forecast_index, ETS_forecast FROM ets_forecastData ORDER BY id DESC LIMIT 1")
    plot_data = cursor.fetchone()

    # Step 3: Close the database connection
    conn.close()

    if plot_data:
        # Step 4: Deserialize the data
        forecast_index_str = plot_data[1].split(',')
        forecast_index = pd.to_datetime(forecast_index_str)
        diff_rol_mean = pickle.loads(plot_data[0])
        # forecast_index = pd.to_datetime(pickle.loads(plot_data[1]))
        ETS_forecast = pickle.loads(plot_data[2])

        # Step 5: Create the plot
        plt.figure(figsize=(10, 5))

        # Plotting the entire dataset
        plt.plot(diff_rol_mean.index, diff_rol_mean['Close'], label='Original Data', color='blue')

        # Plotting forecasted values
        plt.plot(forecast_index, ETS_forecast, label='Forecasted Values', color='orange')

        # Adding labels and legend
        plt.title('ETS Model: Original Data and Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        # Save the plot to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No plot data found in the database."

@app.route('/svr_plot')
def svr_plot():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Execute a query to retrieve the stored data
    cursor.execute("SELECT y_test, y_pred, residuals, mse FROM SVR_ploted ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    # Close the database connection
    conn.close()

    if row:
        # Deserialize the data
        y_test = pickle.loads(row[0])
        y_pred = pickle.loads(row[1])
        residuals = pickle.loads(row[2])
        mse = row[3]

        # Plotting actual vs predicted values with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting actual vs predicted values
        ax1.plot(y_test, label='Actual')
        ax1.plot(y_pred, label='Predicted', linestyle='--')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Closing Price')
        ax1.set_title('Actual vs Predicted Closing Prices')
        ax1.legend()

        # Plotting residuals
        ax2.plot(range(len(residuals)), residuals, color='blue')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('Residuals Plot')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Residuals')
        ax2.set_ylim(-0.6, 0.6)

        # Add RMSE value to the plot
        ax2.text(0.5, 0.95, f'MSE: {mse:.3f}', horizontalalignment='center', verticalalignment='top',
                 transform=ax2.transAxes, fontsize=12, color='blue')

        # Adjust layout
        plt.tight_layout()

        # Create BytesIO object to store the plot
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')

    else:
        return "No data found in the database."

@app.route('/svr_forecast')
def svr_forecast():
    # Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Execute a query to retrieve the stored data
    cursor.execute("SELECT y, diff_rol_mean_index, future_dates, future_predictions FROM SVR_forecast ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    # Close the database connection
    conn.close()

    if row:
        # Deserialize the data
        y = pickle.loads(row[0])
        diff_rol_mean_index = pickle.loads(row[1])
        future_dates = pickle.loads(row[2])
        future_predictions = pickle.loads(row[3])

        # Plotting future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(diff_rol_mean_index, y, label='Historical Data')
        plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Historical and Future Predicted Closing Prices')
        plt.legend()

        # Create BytesIO object to store the plot
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')

    else:
        return "No data found in the database."

@app.route('/prophet_plot')
def prophet_plot():

    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM Prophet_data ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    if row:
        # Deserialize the data
        train_df = pickle.loads(row[1])
        test_df = pickle.loads(row[2])
        residuals = pickle.loads(row[3])
        forecast = pickle.loads(row[4])
        rmse = row[5]

        # Plotting the historical data and forecast with residuals and RMSE
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting historical data and forecast
        ax1.plot(train_df['ds'], train_df['y'], label='Training Data')
        ax1.plot(test_df['ds'], test_df['y'], label='Testing Data')
        ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='orange')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price')
        ax1.set_title('Prophet Model - Forecast with Monthly Seasonality')
        ax1.legend()
        ax1.grid(True)

        # Plotting residuals
        ax2.plot(test_df['ds'], residuals, label='Residuals', color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'Residuals and RMSE: {rmse:.2f}')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Create BytesIO object to store the plot
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No data found in the database."

@app.route('/prophet_forecast')
def prophet_forecast():
    # Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Retrieve the data from the table
    cursor.execute("SELECT * FROM Prophet_forecasted ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    if row:
        # Deserialize the data
        forecast_dates = pickle.loads(row[1])
        forecast_values = pickle.loads(row[2])

        # Plot the forecast
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--', color='orange')
        plt.title('Prophet Model - Forecast with Monthly Seasonality')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.grid(True)
        plt.ylim(-0.2, 0.2)  # Adjust the y-axis limits as needed
        plt.legend()
        plt.tight_layout()

        # Create BytesIO object to store the plot
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')
    else:
        return "No data found in the database."

@app.route('/lstm_plot')
def lstm_plot():
    # Step 1: Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Execute a query to retrieve the stored data
    cursor.execute("SELECT y_test, y_pred, residuals, rmse FROM LSTM_plots ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    # Close the database connection
    conn.close()

    if row:
        # Deserialize the data
        y_test = pickle.loads(row[0])
        y_pred = pickle.loads(row[1])
        residuals = pickle.loads(row[2])
        rmse = pickle.loads(row[3])

        # Plotting actual vs predicted values with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting actual vs predicted values
        ax1.plot(y_test, label='Actual')
        ax1.plot(y_pred, label='Predicted', linestyle='--')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Closing Price')
        ax1.set_title('LSTM Model - Actual vs Predicted Closing Prices')
        ax1.legend()

        # Plotting residuals
        ax2.plot(range(len(residuals)), residuals, color='blue')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('Residuals Plot')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Residuals')

        # Add RMSE value to the plot
        ax2.text(0.5, 0.95, f'RMSE: {rmse:.3f}', horizontalalignment='center', verticalalignment='top',
                 transform=ax2.transAxes, fontsize=12, color='blue')

        # Adjust layout
        plt.tight_layout()

        # Create BytesIO object to store the plot
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')

    else:
        return "No data found in the database."

@app.route('/lstm_forecast')
def lstm_forecast():
    # Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Execute a query to retrieve the stored data
    cursor.execute(
        "SELECT y, diff_rol_mean_index, future_dates, future_predictions FROM SVR_forecast ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    # Close the database connection
    conn.close()

    if row:
        # Deserialize the data
        y = pickle.loads(row[0])
        diff_rol_mean_index = pickle.loads(row[1])
        future_dates = pickle.loads(row[2])
        future_predictions = pickle.loads(row[3])

        # Plotting future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(diff_rol_mean_index, y, label='Historical Data')
        plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Historical and Future Predicted Closing Prices')
        plt.legend()

        # Create BytesIO object to store the plot
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the plot
        plt.close()

        # Return the plot using send_file
        return send_file(plot_bytes, mimetype='image/png')

    else:
        return "No data found in the database."

@app.route('/hybrid_plot')
def hybrid_plot():
    # Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Retrieve the data from the table
    cursor.execute("SELECT * FROM hybrid_plots ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    if row:
        # Deserialize the data
        train_data = pickle.loads(row[1])
        test_data = pickle.loads(row[2])
        arima_forecast = pickle.loads(row[3])
        hybrid_forecast = pickle.loads(row[4])
        residuals = pickle.loads(row[5])
        rmse = row[6]

        # Plot the data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plotting training data, testing data, ARIMA forecast, and hybrid forecast
        ax1.plot(train_data.index, train_data.values, label='Training Data', color='blue')
        ax1.plot(test_data.index, test_data.values, label='Testing Data', color='green')
        ax1.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='orange')
        ax1.plot(test_data.index, hybrid_forecast, label='Hybrid Forecast', color='purple')
        ax1.set_title('Training, Testing, ARIMA Forecast, and Hybrid Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Values')
        ax1.legend()
        ax1.grid(True)

        # Plot residuals
        ax2.plot(test_data.index, residuals, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title(f'Residuals and RMSE: {rmse:.2f}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residuals')
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as a file
        plot_file = 'hybrid_forecast_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # Return the plot file
        return send_file(plot_file, mimetype='image/png')
    else:
        return "No data found in the database."

@app.route('/hybrid_forecast')
def hybrid_forecast():
    # Connect to SQLite database
    conn = sqlite3.connect(r"C:\Users\moiz\DM_Project_Saved.db")
    cursor = conn.cursor()

    # Retrieve the data from the table
    cursor.execute("SELECT * FROM Hybrid_forecasted ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    if row:
        # Deserialize the data
        diff_rol_mean_index = pickle.loads(row[1])
        future_timestamps = pickle.loads(row[2])
        future_hybrid_forecast = pickle.loads(row[3])
        diff_rol_mean = pickle.loads(row[4])

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(diff_rol_mean_index, diff_rol_mean['Close'], label='Historical Data')
        plt.plot(future_timestamps, future_hybrid_forecast, label='Future Predictions', linestyle='--', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Historical Data and Future Predictions')
        plt.legend()
        plt.grid(True)

        # Save the plot as a file
        plot_file = 'hybrid_future_plot.png'
        plt.savefig(plot_file)
        plt.close()

        # Return the plot file
        return send_file(plot_file, mimetype='image/png')
    else:
        return "No data found in the database."

if __name__ == '__main__':
    app.run(debug=True, port=4000)

