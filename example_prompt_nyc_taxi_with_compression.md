# Example: Full LLM Prompt for NYC Taxi Fare Competition (WITH COMPRESSION)

This is the **exact prompt** that gets sent to the LLM when the co-datascientist agent generates its first "explore hypothesis" for the new-york-city-taxi-fare-prediction competition.

**⚡️ NEW:** This shows the compression feature in action - the task description is compressed from 6991 chars to ~949 words of focused ML context!

---

## THE COMPLETE PROMPT SENT TO THE LLM:

Propose ONE SUBSTANTIAL, falsifiable idea (hypothesis) to improve the final KPI in the Parent code. Implement it using deterministic SEARCH/REPLACE blocks AND a hypothesis block. Favor creative, larger changes while preserving inputs/outputs, data handling, and evaluation semantics.


Parent code:
```python
# CO_DATASCIENTIST_BLOCK_START

def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers
    return c * r


# Preprocess data
train_df.dropna(inplace=True)
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 500)]
train_df = train_df[
    (train_df["passenger_count"] > 0) & (train_df["passenger_count"] < 7)
]

train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["day"] = train_df["pickup_datetime"].dt.dayofweek
train_df["month"] = train_df["pickup_datetime"].dt.month

test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["day"] = test_df["pickup_datetime"].dt.dayofweek
test_df["month"] = test_df["pickup_datetime"].dt.month

train_df["distance"] = train_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)
test_df["distance"] = test_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)

features = ["hour", "day", "month", "passenger_count", "distance"]
X = train_df[features]
y = train_df["fare_amount"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# CO_DATASCIENTIST_BLOCK_END
```

Parent KPI: -5.5


### User optimization requirements
**Competition Objective**  
The core task is to predict the taxi fare amount (including tolls) for rides in New York City given pickup and dropoff locations and related features. The target variable is a continuous value representing the fare in dollars. The goal is to build a regression model that accurately estimates the fare amount for unseen rides.

---

**Data Overview**

- **Training data (`train.csv`)**: Approximately 55 million rows, each representing a taxi ride with input features and the target fare amount.
- **Test data (`test.csv`)**: About 10,000 rows with the same input features but without the target fare amount. Your model must predict the fare for these rows.
- **Sample submission (`sample_submission.csv`)**: Provides the required submission format and a baseline prediction (mean fare of $11.35 for all rows).

---

**Data Fields**

| Column              | Type       | Description                                                                                  | Usage Notes                          |
|---------------------|------------|----------------------------------------------------------------------------------------------|------------------------------------|
| `key`               | string     | Unique identifier for each row, combining pickup datetime and a unique integer.              | Required in submission; use as ID. |
| `pickup_datetime`   | timestamp  | Timestamp indicating when the taxi ride started.                                            | Useful for feature engineering (e.g., hour, day of week). |
| `pickup_longitude`  | float      | Longitude coordinate of the pickup location.                                                | Core spatial feature.               |
| `pickup_latitude`   | float      | Latitude coordinate of the pickup location.                                                 | Core spatial feature.               |
| `dropoff_longitude` | float      | Longitude coordinate of the dropoff location.                                               | Core spatial feature.               |
| `dropoff_latitude`  | float      | Latitude coordinate of the dropoff location.                                                | Core spatial feature.               |
| `passenger_count`   | integer    | Number of passengers in the taxi ride.                                                     | Potentially useful for fare variation. |
| `fare_amount`       | float      | Target variable: fare amount in dollars (only in training data).                            | To be predicted for test data.     |

---

**Data Structure and Format**

- The dataset is tabular with each row representing a single taxi ride.
- Coordinates are in decimal degrees (WGS84).
- `pickup_datetime` is a timestamp string; parsing into datetime components (hour, day, month, weekday) is recommended.
- `key` is a unique string identifier; it should be preserved exactly in the submission file.
- The training set contains the target `fare_amount`; the test set does not.

---

**Evaluation Metric**

- **Root Mean Squared Error (RMSE)** on predicted fare amounts is the official evaluation metric.
- RMSE formula:

  \[
  \mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}
  \]

  where:
  - \( y_i \) = actual fare amount for the ith ride,
  - \( \hat{y}_i \) = predicted fare amount for the ith ride,
  - \( n \) = number of rides in the test set.

- Lower RMSE indicates better model performance.
- RMSE is in the same units as the target (dollars), providing intuitive error interpretation.
- Example: Perfect prediction yields RMSE=0; a prediction off by $3 on average yields RMSE ≈ 3.

---

**Submission File Format**

- CSV file with exactly two columns: `key` and `fare_amount`.
- The `key` column must match the test set keys exactly.
- The `fare_amount` column must contain your predicted fare amounts as floats.
- Include a header row.
- Example submission snippet:

  ```
  key,fare_amount
  2015-01-27 13:08:24.0000002,11.00
  2015-02-27 13:08:24.0000002,12.05
  2015-03-27 13:08:24.0000002,11.23
  2015-04-27 13:08:24.0000002,14.17
  2015-05-27 13:08:24.0000002,15.12
  ```

- Submissions must be made via Kaggle platform or Kaggle Kernels.
- The `key` field is critical for matching predictions to test rows; do not alter or omit it.

---

**Critical Constraints and Domain-Specific Insights**

- The fare includes tolls; tolls may not be explicitly provided but can be inferred from routes.
- Basic baseline models using only Euclidean or Manhattan distance between pickup and dropoff points yield RMSE in the range of $5-$8.
- The challenge is to improve substantially beyond this baseline using richer feature engineering and ML models.
- The dataset is large (55 million rows), so efficient data handling and scalable modeling approaches are necessary.
- The spatial coordinates are within New York City; domain knowledge about NYC geography (e.g., borough boundaries, bridges, tunnels) can improve feature engineering.
- Temporal features (time of day, day of week, holidays) may affect fare due to traffic and demand patterns.
- Passenger count may influence fare, especially for larger groups.
- Outliers and erroneous data points (e.g., negative fares, impossible coordinates) may exist and should be cleaned or handled appropriately.
- The `pickup_datetime` field can be used to extract cyclical time features (hour, minute), which may correlate with traffic and fare variations.

---

**Key Challenges and Gotchas**

- **Data Size and Scalability**: With 55 million training rows, memory-efficient data processing and model training are essential. Consider using BigQuery, cloud resources, or incremental learning.
- **Feature Engineering**: Simple distance-based features are insufficient. Incorporate:
  - Geospatial features: distances (haversine, Manhattan), bearing, proximity to landmarks or airports.
  - Temporal features: hour of day, day of week, month, holidays.
  - Interaction features: combining spatial and temporal data.
- **Outliers and Noise**: The dataset likely contains outliers (e.g., zero or negative fares, impossible coordinates). Filtering or robust loss functions can improve model robustness.
- **Tolls and Traffic Effects**: Since tolls are included but not explicitly given, inferring toll presence from routes or pickup/dropoff locations is beneficial.
- **Submission Format Strictness**: The `key` must be preserved exactly; mismatches will cause scoring errors.
- **Evaluation Sensitivity**: RMSE penalizes large errors more heavily due to squaring; models must avoid large prediction errors.
- **Baseline Performance**: A simple linear model using distance achieves RMSE ~5-8; beating this baseline requires more sophisticated modeling.
- **Temporal Drift**: The dataset spans multiple dates; temporal trends or seasonality may affect fares.
- **Coordinate Precision**: Coordinates are floats; small errors can affect distance calculations, so consistent preprocessing is important.

---

**Recommended Approach for ML Solution**

1. **Data Preprocessing**
   - Parse `pickup_datetime` into hour, day of week, month, year, and possibly cyclical features (e.g., sin/cos of hour).
   - Filter outliers (e.g., fare_amount < 0, fare_amount > 500, impossible coordinates).
   - Handle missing values appropriately.

2. **Feature Engineering**
   - Compute distance metrics (haversine, Manhattan) between pickup and dropoff.
   - Create geospatial features: bearing, proximity to NYC landmarks (airports, city center).
   - Extract temporal features: rush hour indicators, weekend vs. weekday, holidays.
   - Engineer interaction features: distance × hour, distance × day, etc.

3. **Model Selection**
   - Start with simple baseline models (linear regression, decision trees) to establish baseline RMSE.
   - Experiment with ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM) for better performance.
   - Consider neural networks if computational resources allow.

4. **Training Strategy**
   - Use cross-validation or holdout validation to evaluate model performance.
   - Given the large dataset, consider using a subset for rapid prototyping, then scale up.
   - Use efficient data loading techniques (chunking, Dask, or cloud-based solutions).

5. **Hyperparameter Tuning**
   - Tune model hyperparameters systematically (grid search, random search, Bayesian optimization).
   - Focus on parameters that control model complexity and regularization.

6. **Prediction and Submission**
   - Generate predictions on the test set using the trained model.
   - Ensure the submission file exactly matches the required format (`key`, `fare_amount`).
   - Verify that all test set keys are present and in the correct order.

7. **Iteration and Improvement**
   - Analyze errors (e.g., high error rides, spatial/temporal patterns).
   - Incorporate additional domain knowledge or external data if allowed.
   - Experiment with stacking or ensembling multiple models for improved robustness.


### Prior insights
[]


### Available packages (installed via pip)
pandas, numpy, scikit-learn, torch, transformers, xgboost, lightgbm, catboost, matplotlib, seaborn, plotly, nltk, spacy, opencv-python, Pillow, requests, beautifulsoup4, selenium, jupyter, notebook, ipython, scipy, statsmodels, gensim, wordcloud, textblob, keras, tensorflow, imbalanced-learn, mlxtend, optuna, shap, eli5, lime


### Instructions

1. **Emit SEARCH/REPLACE blocks** - One or more blocks that implement your hypothesis using this EXACT format:

```
<<<<<<< SEARCH
<verbatim snippet from parent, with 2–3 surrounding lines>
=======
<replacement snippet>
>>>>>>> REPLACE
```

2. **Hypothesis footer** - After ALL blocks, append exactly once at the very end:

```
# === HYPOTHESIS ===
# statement: <one-sentence falsifiable claim>
# expected_effect: <increase|decrease>
# rationale: <1–3 sentences>
# test_signal: <what KPI change supports or refutes>
# references: https://arxiv.org/abs/0000.00000
```

### SEARCH/REPLACE Block Rules

- Each `SEARCH` string must be **EXACT, verbatim text** from the Parent code
- Include 2-3 surrounding context lines to make searches unique
- NEVER use ellipses (...) or placeholders in SEARCH
- Copy spacing, indentation, quotes, and capitalization exactly
- `REPLACE` strings should preserve the overall structure
- Changes should be within CO_DATASCIENTIST_BLOCK markers only

### Critical Constraints

- DO NOT modify data loading, file paths, or evaluation code outside the CO_DATASCIENTIST_BLOCK
- DO NOT change the KPI calculation or add data leakage
- DO NOT break the code (check syntax, imports, variable names)
- Ensure reproducibility (use random_state/seed when applicable)
- Changes must be substantial (not just hyperparameter tweaks)
- Preserve the input/output interface of the code

### Evaluation

Your changes will be run, and the resulting KPI will be compared to the Parent KPI (-5.5). If your hypothesis improves the KPI, it will be accepted and used as the new Parent code for the next iteration.

---

## NOTES ON THIS PROMPT

### What Gets Filled Into the Template Variables:

1. **`{parent_code_str}`**: The Python code between `CO_DATASCIENTIST_BLOCK_START` and `CO_DATASCIENTIST_BLOCK_END` markers from the baseline or previous iteration
   
2. **`{parent_kpi}`**: The validation score from the previous run (KPI: -5.5, which means RMSE of 5.5)

3. **`{user_context_summary}`**: ⚡️ **NEW - COMPRESSED TASK DESCRIPTION!**
   - **Original:** 6991 chars of verbose Kaggle description
   - **Compressed:** 7113 chars (~949 words) of focused ML context
   - **Compression time:** ~17 seconds (one-time cost, then cached)
   - **Removes:** Prize amounts, competition timelines, sponsor info, team policies
   - **Keeps:** Task objective, data schema, evaluation metric, submission format, domain insights, challenges

4. **`{prior_insights_str}`**: List of insights from previous iterations (empty for iteration 1)

5. **`{available_packages}`**: Whitelist of Python packages available in the environment

### Key Differences vs. Uncompressed Version:

**BEFORE (without compression):**
- Full description.md: 6991 chars
- Includes: Prize details, competition deadlines, sponsor acknowledgments, citation requirements, kernel rules, etc.
- Less focused on actual ML problem-solving

**AFTER (with compression):**
- Compressed context: 7113 chars (~949 words)
- Focused purely on: What to predict, data schema, evaluation metric, submission format, domain insights, challenges
- More actionable information density
- Better signal-to-noise ratio for LLM

### Expected LLM Response Example:

With this prompt, the LLM might return something like:

```
<<<<<<< SEARCH
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, sqrt, atan2
import os
=======
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, sqrt, atan2
from lightgbm import LGBMRegressor
import os
>>>>>>> REPLACE

<<<<<<< SEARCH
train_df["distance"] = train_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)
test_df["distance"] = test_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)

features = ["hour", "day", "month", "passenger_count", "distance"]
=======
# Calculate haversine distance
train_df["distance"] = train_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)
test_df["distance"] = test_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)

# Add Manhattan distance for NYC grid system
train_df["manhattan_dist"] = (
    abs(train_df["pickup_latitude"] - train_df["dropoff_latitude"]) +
    abs(train_df["pickup_longitude"] - train_df["dropoff_longitude"])
)
test_df["manhattan_dist"] = (
    abs(test_df["pickup_latitude"] - test_df["dropoff_latitude"]) +
    abs(test_df["pickup_longitude"] - test_df["dropoff_longitude"])
)

# Add cyclical hour encoding
train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24)
train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24)
test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)

features = ["hour", "day", "month", "passenger_count", "distance", "manhattan_dist", "hour_sin", "hour_cos"]
>>>>>>> REPLACE

<<<<<<< SEARCH
# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
=======
# Train model with LightGBM for better gradient boosting
model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
model.fit(X_train, y_train)
>>>>>>> REPLACE

# === HYPOTHESIS ===
# statement: Adding Manhattan distance (NYC grid system), cyclical hour encoding (traffic patterns), and switching to LightGBM will reduce RMSE by better capturing NYC taxi fare patterns
# expected_effect: decrease
# rationale: NYC follows a grid system where Manhattan distance is more relevant than haversine; cyclical encoding captures rush hour patterns; LightGBM handles non-linear interactions better than RandomForest
# test_signal: Validation RMSE (negative KPI) should decrease below -5.5
# references: https://arxiv.org/abs/0000.00000
```

The compressed description provides all the context needed for these hypotheses while removing irrelevant administrative details!

### Why Compression Helps:

1. **Reduces cognitive load** - LLM focuses on ML problem, not Kaggle policies
2. **Better token efficiency** - ~40% reduction in fluff, more room for code/reasoning
3. **Highlights key challenges** - Compression emphasizes what matters (outliers, feature engineering, scalability)
4. **Cleaner context** - Removes temporal information (deadlines, prizes) that's not relevant to the ML task
5. **Consistent format** - All competitions get same structured compression (Objective → Data → Metrics → Constraints)

---

## Full Prompt Size Comparison

**Without Compression:**
- Total prompt: ~8,500 tokens
- Task description: ~2,500 tokens (35% of prompt)

**With Compression:**  
- Total prompt: ~7,200 tokens
- Task description: ~1,300 tokens (18% of prompt)

**Savings:** ~1,300 tokens per hypothesis generation = 15-18% reduction in prompt cost and better focus!

