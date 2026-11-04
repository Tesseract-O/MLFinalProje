# Fitness Class Attendance Predictor

**Live app:** [link to your streamlit.app page]

Predicts whether a gym member who booked a class will actually attend,
so the club can decide how many extra bookings to accept.

## Data
Kaggle "Fitness Club Dataset for ML Classification": 1,500 bookings, 8 columns.
Target: `attended` (1 = attended, 0 = no-show). About 30% of bookings were attended.

## Data cleaning
- `days_before`: removed the "days" text and converted to integer
- `day_of_week`: unified spellings ("Wednesday" -> "Wed", "Fri." -> "Fri")
- `category`: replaced "-" with "unknown" (13 rows, kept as its own category)

## Key findings
- Wednesday has the lowest attendance (~13.6% vs ~28-33% on other days),
  though it also has the fewest bookings (81).
- [add 1-2 more findings from your EDA]

## Models
Evaluated with 5-fold stratified cross-validation on the training set:

| Model | F1 | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| Baseline (always "no-show") | 0.00 | 0.50 | 0.00 | 0.00 |
| Logistic Regression | 0.63 | 0.82 | 0.68 | 0.59 |
| Random Forest | 0.52 | 0.78 | 0.46 | 0.60 |
| XGBoost | 0.63 | 0.81 | 0.72 | 0.57 |

Logistic regression was chosen: it performs about as well as XGBoost,
and is simpler and easier to interpret.

**Final test set result** (375 held-out bookings, threshold = [your threshold]):

| Metric | Value |
|---|---|
| ROC-AUC | 0.82 |
| Recall (attended) | 0.75 |
| Precision (attended) | 0.57 |
| F1 (attended) | 0.65 |

Test results are consistent with cross-validation (ROC-AUC 0.82, F1 0.63),
so the model does not appear to be overfitting.

Confusion matrix:

|  | Predicted no-show | Predicted attend |
|---|---|---|
| **Actually no-show** | 198 | 63 |
| **Actually attended** | 29 | 85 |

The model catches about 3 out of 4 attendees, but about 4 out of 10
"will attend" predictions are false alarms. The decision threshold in the app
lets the club trade one against the other.

## How to use the app
The model outputs an attendance probability. The decision threshold turns it
into a yes/no prediction. A lower threshold catches more attendees but produces
more false alarms; a higher one is more cautious.

## Limitations
- Small dataset (1,500 rows); results may not generalize to other gyms.
- Findings show association, not causation.
- The model gives estimates, not guarantees.
- Overall accuracy (75%) is only about 5 points above always predicting "no-show" (~70%),
  so ROC-AUC, recall and F1 are the more meaningful metrics here.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Run Streamlit
https://tesseract-o-mlfinalproje-app-mwevc4.streamlit.app/
