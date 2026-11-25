# AI-Based Traffic Flow Prediction and Congestion Monitoring System

An intelligent traffic forecasting and monitoring platform that leverages machine learning to predict vehicle flow and identify congestion patterns across multiple road junctions.

The system analyzes historical traffic observations to generate short-term vehicle count forecasts, estimate prediction intervals, and classify traffic conditions into interpretable congestion levels. An interactive dashboard provides traffic visualization, forecast exploration, historical replay, and model performance analysis.

---

## Overview

The **AI-Based Traffic Flow Prediction and Congestion Monitoring System** is a machine learning-powered web application designed to analyze historical traffic patterns and forecast vehicle volumes across six road junctions.

Using gradient-boosted decision trees, the system generates predictions at multiple future time horizons, enabling users to examine anticipated traffic conditions and monitor potential congestion.

### Key Features

- **Multi-Junction Forecasting:** Predicts traffic flow across six road junctions.
- **Multi-Horizon Prediction:** Generates forecasts for 5, 15, 30, and 60 minutes ahead.
- **Prediction Intervals:** Provides likely ranges alongside point predictions to communicate forecast uncertainty.
- **Congestion Classification:** Categorizes traffic conditions into Low, Moderate, and Heavy levels.
- **Interactive Dashboard:** Displays current traffic observations, historical trends, and future predictions.
- **Historical Traffic Replay:** Reproduces held-out traffic observations in synchronization with the system clock.
- **Custom Time Exploration:** Allows users to select specific dates and times to examine traffic conditions.
- **Forecast Export:** Supports downloading daily traffic predictions in CSV format.
- **Model Evaluation:** Provides a dedicated model performance page comparing predictions against baseline forecasting methods.
- **Secure Authentication:** Implements server-side account management and password hashing.
- **Responsive Interface:** Supports desktop and mobile screen sizes with light and dark themes.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11+ |
| Backend Framework | Flask |
| Machine Learning | Scikit-learn |
| Forecasting Algorithm | Gradient-Boosted Decision Trees |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| Data Visualization | Interactive Charts |
| Authentication | Server-side authentication, password hashing |
| Security | CSRF Protection, Login Rate Limiting |
| Testing | Pytest |
| Deployment Server | Waitress |

---

## Machine Learning Architecture

The forecasting pipeline uses historical vehicle count observations to learn temporal traffic patterns and generate predictions at multiple future time horizons.

### 1. Dataset

The system uses the [Road Traffic Prediction Dataset](https://www.kaggle.com/datasets/annsanababy/road-traffic-prediction-dataset) available on Kaggle.

**Dataset characteristics:**

- Approximately 56 days of historical traffic observations.
- Traffic measurements from six different road junctions.
- Observations recorded at five-minute intervals.
- Vehicle counts used as the primary prediction target.

### 2. Data Preprocessing

Historical traffic observations are processed before model training to improve data consistency and forecasting reliability.

The preprocessing pipeline includes:

- Loading and organizing historical traffic observations.
- Handling missing and invalid sensor measurements.
- Identifying sensor outages, including intervals where all junctions simultaneously report zero vehicles.
- Treating identified sensor outages as missing observations rather than interpreting them as genuine traffic conditions.
- Preparing temporally ordered training and evaluation datasets.

### 3. Feature Engineering

The forecasting models utilize temporal and historical traffic information to capture short-term and recurring traffic patterns.

The feature set incorporates:

- Recent vehicle count observations from previous time intervals.
- Historical traffic measurements from the same time on the previous day.
- Historical traffic measurements from the corresponding time in the previous week.

These features allow the model to learn both recent traffic dynamics and recurring temporal patterns.

### 4. Model Development

The system uses **Gradient-Boosted Decision Trees**, implemented using Scikit-learn.

Separate forecasting models are trained for different prediction horizons:

| Forecast Horizon | Prediction Interval |
|---|---|
| 5 Minutes | Short-term |
| 15 Minutes | Short-term |
| 30 Minutes | Medium-term |
| 60 Minutes | Long-term |

This multi-model approach allows each forecasting model to learn patterns specific to its respective prediction horizon.

### 5. Training and Evaluation Strategy

The dataset is divided chronologically to preserve the temporal ordering of traffic observations and prevent future information from entering the training process.

| Dataset Split | Duration | Purpose |
|---|---:|---|
| Training Set | First 48 days | Model training |
| Testing Set | Final 8 days | Unseen-data evaluation |

The final eight days are excluded from model training and reserved for evaluating forecasting performance.

This chronological evaluation strategy provides a more realistic assessment of how the models perform on future traffic observations.

### 6. Prediction and Congestion Monitoring

For each road junction, the system generates:

- Predicted vehicle counts at four future time horizons.
- Likely prediction ranges.
- Congestion classifications based on traffic levels.

The predictions are presented through an interactive dashboard, allowing users to monitor traffic trends and examine anticipated congestion conditions.

---

## Model Performance

The forecasting system is evaluated using the held-out test dataset and compared against a persistence baseline.

The persistence baseline assumes that the future vehicle count will remain equal to the current observed vehicle count.

**Key evaluation result:**

The 60-minute forecasting model achieves an average prediction error of approximately **13 vehicles**, representing roughly a 50% reduction in error compared with the persistence baseline on the held-out test data.

This demonstrates the model's ability to capture historical traffic patterns and provide more informative forecasts than simply assuming traffic remains unchanged.

The application includes a dedicated model evaluation page at `/model`, where forecasting performance and baseline comparisons can be explored.

---

## Application Architecture

The application follows a modular architecture separating the web interface, backend services, forecasting pipeline, and persistent data storage.

**High-level workflow:**

1. Historical traffic data is loaded and preprocessed.
2. Temporal features are extracted from historical vehicle observations.
3. Separate forecasting models are trained for each prediction horizon.
4. Trained models and evaluation results are stored for application use.
5. The Flask backend manages authentication, dashboard requests, and forecasting operations.
6. Predictions are presented through interactive dashboard visualizations.
7. Users can explore historical traffic replay and export forecast data.

---

## Application Functionality

### Traffic Monitoring Dashboard

The main dashboard provides an overview of traffic conditions across all six monitored junctions.

It includes:

- Junction-level vehicle count summaries.
- Historical traffic trends covering the previous six hours.
- Forecast visualizations covering the next hour.
- Multi-horizon prediction tables.
- Congestion level indicators.
- Interactive historical replay controls.
- Date and time selection.
- CSV forecast downloads.
- Light and dark display themes.

### Historical Traffic Replay

The application includes a historical replay mechanism that reproduces the final eight days of the dataset.

Users can:

- Explore traffic conditions at specific dates and times.
- Start and control historical playback.
- Observe changes in vehicle counts and forecast values.
- Examine traffic patterns across different junctions.

The replay functionality provides an interactive simulation of traffic monitoring using previously recorded observations rather than a live traffic sensor feed.

### Model Evaluation Dashboard

The dedicated model evaluation page provides insights into the forecasting system's performance.

It presents model accuracy information and comparisons against baseline forecasting approaches using observations from the held-out testing period.

### User Account Management

The application includes server-side user authentication and account management.

Security-related functionality includes:

- Secure password hashing.
- Server-side session management.
- CSRF protection for form submissions.
- Protection against repeated unsuccessful login attempts.
- Password change functionality.
- Account deletion functionality.

---

## Project Structure

```text
AI-Based-Traffic-Flow-Prediction-and-Congestion-Monitoring-System/
│
├── traffic_predictor/
│   ├── Application modules
│   ├── Authentication
│   ├── Dashboard
│   ├── Forecasting logic
│   ├── API endpoints
│   └── Templates and static assets
│
├── scripts/
│   └── train.py
│
├── tests/
│   └── Automated test suite
│
├── data/
│   └── Historical traffic dataset
│
├── models/
│   └── Trained forecasting models
│
├── notebooks/
│   └── Dataset exploration notebook
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

*Note: Directory descriptions are provided at a high level; refer to the repository for the complete implementation.*

---

## Installation and Setup

### Prerequisites

Ensure the following are installed:

- Python 3.11 or newer
- pip
- Git
- Kaggle account and API credentials for dataset access

### 1. Clone the Repository

```bash
git clone https://github.com/sritejaswini15/AI-Based-Traffic-Flow-Prediction-and-Congestion-Monitoring-System.git

cd AI-Based-Traffic-Flow-Prediction-and-Congestion-Monitoring-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Download the Dataset

Install the Kaggle CLI:

```bash
pip install kaggle
```

Configure your Kaggle API credentials according to the [Kaggle API documentation](https://www.kaggle.com/docs/api).

Download and extract the dataset:

```bash
kaggle datasets download -d annsanababy/road-traffic-prediction-dataset -p data --unzip
```

Ensure the extracted dataset is placed in the expected location used by the training pipeline.

### 5. Train the Forecasting Models

Run the training pipeline:

```bash
python -m scripts.train
```

The training process generates the forecasting models and associated evaluation artifacts required by the application.

### 6. Launch the Application

Start the Flask development server:

```bash
flask --app traffic_predictor run
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

Create an account to access the traffic monitoring dashboard.

---

## Testing

The project includes an automated test suite covering core application functionality.

Run the tests using:

```bash
python -m pytest
```

The test suite contains 49 automated tests covering:

- Machine learning and forecasting functionality.
- Backend application routes.
- API behavior.
- Authentication and account management.
- Security-related functionality.

---

## Production Deployment

For production execution, install the production dependencies:

```bash
pip install -r requirements.txt
```

Configure a strong application secret key using an environment variable.

**Windows PowerShell:**

```powershell
$env:TP_SECRET_KEY = "your-long-random-secret-key"
```

**macOS / Linux:**

```bash
export TP_SECRET_KEY="your-long-random-secret-key"
```

Run the application using Waitress:

```bash
python -m waitress --listen=0.0.0.0:8000 --call traffic_predictor:create_app
```

The application will be available on the configured host and port.

For production environments, use a strong secret key, configure HTTPS, and follow appropriate database backup and deployment security practices.

---

## Future Enhancements

Potential areas for further development include:

- Integration with real-time traffic sensor feeds.
- Live traffic monitoring and continuous prediction updates.
- Incorporation of weather and road-condition information.
- Expansion to additional road junctions and geographical regions.
- Advanced spatiotemporal forecasting architectures.
- Automated model retraining and performance monitoring.
- Geospatial traffic visualization and congestion mapping.

---
