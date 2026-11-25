# Traffic Predictor

A web app that predicts how busy six road junctions will be over the next hour.

Sign in, open the dashboard, and for every junction you see:

- the current number of vehicles
- forecasts for 5, 15, 30 and 60 minutes ahead, each with a likely range
- a congestion level: **low**, **moderate** or **heavy**

## How it works

1. **Data:** 56 days of vehicle counts from six junctions, one reading every 5 minutes
   ([Kaggle dataset](https://www.kaggle.com/datasets/annsanababy/road-traffic-prediction-dataset)).
2. **Model:** a machine-learning model (gradient-boosted trees, scikit-learn) learns from the last few hours,
   the same time yesterday and the same time last week. There is one model per forecast horizon.
3. **Training:** the model learns from the first 48 days. The last 8 days are kept aside for testing.
4. **Dashboard:** plays back those 8 test days in step with your clock, so it feels like live traffic.
   You can also pick any day and time, press **Play** to watch the traffic change, or download a day's forecasts as CSV.
5. **Model page:** `/model` shows how accurate the forecasts are.

On the test days, the 60-minute forecast is off by about 13 vehicles on average. That is half the error of
simply assuming traffic stays the same.

## Getting started

You need Python 3.11 or newer.

```bash
# 1. Set up
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Download the dataset into data/
#    (needs a Kaggle account and API token: https://www.kaggle.com/docs/api)
pip install kaggle
kaggle datasets download -d annsanababy/road-traffic-prediction-dataset -p data --unzip

# 3. Train the model (about 40 seconds)
python -m scripts.train

# 4. Run the app
flask --app traffic_predictor run
```

Open http://127.0.0.1:5000, create an account and go to the dashboard.

To run the tests:

```bash
python -m pytest
```

## What was done

This project started as static HTML pages. It was rebuilt into a working application:

- **Real backend:** a Flask web app with a SQLite database.
- **Secure accounts:** passwords used to be saved as plain text in the browser. Now they are stored safely hashed
  on the server, with protection against fake form posts (CSRF) and repeated wrong sign-ins.
  You can change your password or delete your account.
- **Real predictions:** the old charts showed random numbers. Now a trained model makes the forecasts,
  and each one has a likely range.
- **New dashboard:** a summary of all junctions, a card for each junction, a chart of the last
  6 hours and the next hour, a forecast table, replay controls, CSV download, and light and dark themes.
  It also works on phones.
- **Accuracy page:** compares the model with simple guesses, using days it never saw during training.
- **Clean data:** sensor outages (hours where every junction reads 0) are treated as missing, not as empty roads.
- **Tests:** 49 automated tests cover the model, the pages, the API and the security features.

## Project structure

```
traffic_predictor/     the web app (pages, accounts, API, forecasting code)
scripts/train.py       trains the model and saves it to models/
tests/                 automated tests
data/                  put the downloaded dataset here
models/                trained model and its accuracy results
notebooks/             first look at the dataset (Google Colab)
```

## Running in production

```bash
pip install -r requirements.txt
set TP_SECRET_KEY=<a long random string>      # macOS/Linux: export TP_SECRET_KEY=...
python -m waitress --listen=0.0.0.0:8000 --call traffic_predictor:create_app
```

## License

MIT
