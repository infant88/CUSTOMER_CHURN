# Customer Churn Prediction

Predicts whether a telecom customer will churn using XGBoost, with SHAP-based explainability and a live Streamlit interface.

## Live Demo
> Deploy to Streamlit Cloud (free) — instructions below

## Results
| Model | AUC-ROC | F1 (churn) |
|---|---|---|
| Logistic Regression | ~0.84 | ~0.58 |
| Random Forest | ~0.85 | ~0.60 |
| **XGBoost** | **~0.86** | **~0.62** |

## Project Structure
```
churn_project/
├── train.py           # EDA, feature engineering, model training
├── app.py             # Streamlit app
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
- Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Place it in the same folder as `train.py`

### 3. Train the model
```bash
python train.py
```
This generates: `model.pkl`, `scaler.pkl`, `feature_names.pkl`, and 3 plot images.

### 4. Run the app locally
```bash
streamlit run app.py
```

### 5. Deploy to Streamlit Cloud (free public URL)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Done — you get a live URL to put on your resume!

## Tech Stack
- **Python** · pandas · numpy · scikit-learn
- **XGBoost** — gradient boosted trees
- **SHAP** — model explainability
- **Streamlit** — web interface
- **Matplotlib / Seaborn** — EDA visualisations

## Key Features
- Compares 3 ML models with AUC and F1 metrics
- SHAP waterfall plot explains every individual prediction
- Feature engineering: `AvgMonthlyCharge`, `HasMultipleServices`
- Clean Streamlit UI with sidebar inputs and risk level output

## Resume Bullet (copy this)
[> Built a customer churn predictor using XGBoost on the Telco dataset (AUC 0.86), with SHAP-based explainability and a Streamlit deployment — [your-app-url]](http://localhost:8501/).
