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

**Final test set result:** ROC-AUC [0.82], F1 [0.75], recall [0.75], precision [0.75]

## How to use the app
The model outputs an attendance probability. The decision threshold turns it
into a yes/no prediction. A lower threshold catches more attendees but produces
more false alarms; a higher one is more cautious.

## Limitations
- Small dataset (1,500 rows); results may not generalize to other gyms.
- Findings show association, not causation.
- The model gives estimates, not guarantees.

## Run locally
pip install -r requirements.txt
streamlit run app.py
