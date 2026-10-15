# Food Delivery Time Prediction

Predict how many **minutes** a food delivery will take, given the order details —
who is delivering it, how far it has to travel, the weather, traffic, time of day,
and more. Framed as a **supervised regression** problem and solved end-to-end, from
raw messy data to a saved model ready for inference.

---

## Results

Three models were trained and compared. Random Forest came out on top.

| Model | Test R² | Test RMSE (min) | Train R² | Notes |
|---|---|---|---|---|
| Linear Regression | 0.57 | 6.21 | 0.58 | Baseline — relationships aren't linear |
| Decision Tree | 0.66 | 5.49 | 1.00 | Overfits (memorises the training set) |
| **Random Forest** | **0.82** | **4.03** | 0.97 | Big gain — averaging many trees tames overfitting |

*RMSE is in minutes, so the final model is off by roughly 4 minutes on average.*

The Random Forest still fits the training data more closely than the test data
(train R² 0.97 vs test R² 0.82), so there is room to improve generalisation —
for example by limiting tree depth or leaf size.

---

## Approach

1. **Load & inspect** the raw data
2. **Clean** messy text and missing values (e.g. `"conditions Sunny"` → `Sunny`, `"(min) 24"` → `24`)
3. **EDA** — find which features actually move delivery time
4. **Feature engineering**
   - `distance_km` from restaurant/customer coordinates via the **Haversine formula**
   - `order_hour` extracted from the order timestamp
5. **Encode** categoricals (ordinal for traffic density, one-hot for the rest) and **scale** features
6. **Train & compare** Linear Regression, Decision Tree, Random Forest
7. **Evaluate** (RMSE, R², Adjusted R²) and **save** the best model for inference

A small experiment also tested an extra `order_period` feature (Lunch / Dinner / Normal).
It left Adjusted R² essentially unchanged, so it was not kept.

The maths and reasoning behind each step are explained inline in the notebook.

---

## Dataset

The dataset contains historical food-delivery orders with features such as the
delivery person's age and rating, restaurant and delivery coordinates, weather,
road-traffic density, vehicle type, order type, festival flag, and city type.
The target is `Time_taken(min)`.

---

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib

---

## How to run

Requires **Python 3.11+**.

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/food-delivery-time-prediction.git
cd food-delivery-time-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook (run it from the repo root so it finds food_delivery.csv)
jupyter notebook Food_Delivery_Time_Prediction.ipynb
```

Running the notebook top to bottom retrains the models and regenerates
`rf_model.pkl` and `standard_scaler.pkl`.

---

## Using the saved model

The model expects the same 21 features, in the same order, that it was trained on.
`scaler.feature_names_in_` stores that order, so you can build one row, fill any
missing one-hot columns with `0`, and predict:

```python
import joblib
import pandas as pd

model  = joblib.load("rf_model.pkl")
scaler = joblib.load("standard_scaler.pkl")

order = pd.DataFrame([{
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.7,
    "distance_km": 5.2,
    "order_hour": 19,
    "Road_traffic_density": 3,        # Low=0, Medium=1, High=2, Jam=3
    "Vehicle_condition": 1,
    "multiple_deliveries": 1,
    "Type_of_order_Meal": 1,          # one-hot: the dropped baseline is Buffet
    "Type_of_vehicle_scooter": 1,     # baseline: bicycle
    "City_Urban": 1,                  # baseline: Metropolitian
    "Weatherconditions_Stormy": 1,    # baseline: Cloudy
}])

# Add every other column the scaler saw during training (missing = 0), in the same order
order = order.reindex(columns=scaler.feature_names_in_, fill_value=0)

scaled = pd.DataFrame(scaler.transform(order), columns=order.columns)
print(f"Predicted delivery time: {model.predict(scaled)[0]:.1f} minutes")
```

> `rf_model.pkl` is a pickle, so load it with the same scikit-learn version pinned in
> `requirements.txt`. Only load pickle files from sources you trust.

---

## Repository structure

```
food-delivery-time-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Food_Delivery_Time_Prediction.ipynb   # the full, documented notebook (with outputs)
├── food_delivery.csv                      # dataset
├── rf_model.pkl                           # saved best model (Random Forest, ~44 MB)
└── standard_scaler.pkl                    # saved scaler
```
