# 📊 AI-Based MultiModel Time Series Forecasting System

A forecasting system that implements and compares different time series models (ARIMA, ANN, Hybrid ARIMA–ANN, and more) across various sectors, with both backend and frontend support for **data visualization** and **forecasting results**.

The project integrates:

* **Python (Flask)** as the backend service to handle requests, run models, and serve results.
* **React.js (App.js)** as the frontend for users to interact with forecasts and visualizations.
* **SQLite3** for storing model results.

---

## 👥 Team Contributions

### 🟢 Wasif Mehboob

* Implemented the **Flask backend** and **React.js frontend**.
* Built API endpoints in **Flask** to serve forecasting results and plots.
* Designed **frontend components (App.js)** for displaying forecasts interactively.
* Implemented a **future forecasting workaround** for models with only `predict()` (no `forecast()`). This method used the last observed data to initialize a future data array, which was then passed into the model’s prediction function.

### 🟠 Ahsan Waseem

* Focused on **data preprocessing and normalization**.
* Tested different data transformation strategies:

  * Difference between Original & Rolling Mean
  * Difference between Original & Exponentially Weighted Mean
  * Differenced data (shift=1)
* Selected the **Rolling Mean differencing** due to its lower p-value and better stationarity.
* Configured, trained, and validated models with built-in forecasting (`forecast()`) and evaluated them using cross-validation.
* Ensured proper model tuning for **accurate forecasts**.

---

## 🗂 Project Structure

```bash
├── App.js              # React frontend – displays data & forecast results
├── Flask.py            # Flask backend – loads DB results & serves APIs
├── Main_Code.py        # Core script – preprocessing, training models, forecasting
├── Documentation.pdf   # Project report & documentation
└── README.md           # Project description
```

---

## 🔑 Key Files

* **Main_Code.py**

  * Handles data preprocessing and transformations.
  * Trains models using `statsmodels` (ARIMA, SARIMA etc.).
  * Includes other architectures **ANN, Hybrid ARIMA–ANN, LSTM, Prophet, SVR**.
  * Generates predictive forecasts and stores results in **SQLite3 DB**.

* **Flask.py**

  * Loads forecasting results from the database.
  * Hosts API endpoints to send results and plots to the frontend.

* **App.js**

  * React.js frontend file.
  * Displays forecasts, plots, and allows user interaction with the system.

* **Documentation.pdf**

  * Detailed explanation of methodology, models, and results.

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Data-Mining-Project.git
cd Data-Mining-Project
```

### 2️⃣ Set Up the Backend (Flask)

Install dependencies:

```bash
pip install -r requirements.txt   # if available
```

Or manually install:

```bash
pip install flask statsmodels tensorflow scikit-learn prophet sqlite3
```

Run the backend:

```bash
python Flask.py
```

The backend should now be running on:
👉 `http://127.0.0.1:5000/`

---

### 3️⃣ Set Up the Frontend (React)

Navigate to frontend directory (if structured separately) or same folder if `App.js` is standalone.
Install dependencies:

```bash
npm install
```

Run the frontend:

```bash
npm start
```

The frontend should now be running on:
👉 `http://localhost:3000/`

---

### 4️⃣ Usage Flow

* Preprocess & train models using `Main_Code.py` → results stored in SQLite3.
* Start Flask backend (`Flask.py`) → serves API endpoints.
* Launch React frontend (`App.js`) → fetches and displays results.

---

## 📌 Features

✔️ Multiple model support: ARIMA, SARIMA, ANN, Hybrid ARIMA–ANN, LSTM, Prophet, SVR.

✔️ Custom workaround for `predict()`-only models.

✔️ SQLite3 integration for storing results.

✔️ Flask API for backend handling.

✔️ React.js frontend for visualizations and user interaction.

---
