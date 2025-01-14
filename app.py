from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from wordcloud import WordCloud
import os

# Initialize Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to save plots
def save_plot(df, forecast, forecast_ci):
    # Distribusi Kategori
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Product Category', order=df['Product Category'].value_counts().index)
    plt.title('Distribusi Kategori')
    plt.xlabel('Kategori')
    plt.ylabel('Jumlah Pesanan')
    category_plot_path = os.path.join(UPLOAD_FOLDER, 'category_distribution.png')
    plt.savefig(category_plot_path)
    plt.close()

    # Word Cloud untuk Kategori
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Product Category']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud Kategori Transaksi')
    plt.axis('off')
    wordcloud_plot_path = os.path.join(UPLOAD_FOLDER, 'wordcloud_categories.png')
    plt.savefig(wordcloud_plot_path)
    plt.close()

    # Forecast Plot
    plt.figure(figsize=(10, 6))
    ax = df['Total Revenue'].resample('M').sum().plot(label='Jumlah Pesanan')
    forecast.predicted_mean.plot(ax=ax, label='Ramalan', color='r', alpha=.7)
    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='r', alpha=.2)
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Jumlah')
    forecast_plot_path = os.path.join(UPLOAD_FOLDER, 'forecast_plot.png')
    plt.savefig(forecast_plot_path)
    plt.close()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to upload CSV file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            # Load CSV file into dataframe
            df = pd.read_csv(file_path)

            # Ensure 'Date' column exists and convert to datetime
            if 'Date' not in df.columns:
                return jsonify({'error': "'Date' column is missing in the CSV file"}), 400

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isnull().any():
                return jsonify({'error': "'Date' column contains invalid date values"}), 400

            df.set_index('Date', inplace=True)

            # Ensure required columns exist
            if 'Product Category' not in df.columns or 'Total Revenue' not in df.columns:
                return jsonify({'error': "Required columns are missing in the CSV file"}), 400

            # Aggregasi data bulanan
            monthly_data = df['Total Revenue'].resample('M').sum()

            # Check if there are enough months of data for seasonal decomposition
            if len(monthly_data) < 12:
                # Use a non-seasonal ARIMA model if less than 12 months of data
                model = SARIMAX(monthly_data, order=(1, 1, 1))
            else:
                # Perform seasonal decomposition and SARIMAX with seasonal components
                result = seasonal_decompose(monthly_data, model='additive', period=12)
                model = SARIMAX(monthly_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

            # Fit and forecast using the model
            model_fit = model.fit(disp=False)
            forecast = model_fit.get_forecast(steps=12)
            forecast_ci = forecast.conf_int()

            # Save all plots
            save_plot(df, forecast, forecast_ci)

            # Return forecast data
            forecast_df = pd.DataFrame({
                'Date': forecast_ci.index.strftime('%Y-%m-%d'),
                'Predicted Revenue': forecast.predicted_mean,
                'Lower CI': forecast_ci.iloc[:, 0],
                'Upper CI': forecast_ci.iloc[:, 1]
            })

            return jsonify({
                'message': 'File processed successfully!',
                'plots': {
                    'category_distribution': '/static/uploads/category_distribution.png',
                    'wordcloud_categories': '/static/uploads/wordcloud_categories.png',
                    'forecast_plot': '/static/uploads/forecast_plot.png'
                },
                'forecast': forecast_df.to_dict(orient='records')
            }), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
