import os
from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils.multiclass import type_of_target

# Initialize app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Helper to detect distribution
def identify_distribution(series):
    skew = series.skew()
    return 'normal' if abs(skew) < 0.5 else 'skewed'

# Missing value imputation
def handle_missing_values(df, summary):
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            summary['imputations'][col] = f"mode ({mode_val})"
        else:
            dist = identify_distribution(df[col].dropna())
            if dist == 'normal':
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                summary['imputations'][col] = f"mean ({mean_val:.2f})"
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                summary['imputations'][col] = f"median ({median_val})"
    return df

# IQR-based outlier treatment
def treat_outliers(df, summary, method='cap'):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]

        if not outliers.empty:
            if method == 'cap':
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])
                summary['outliers'][col] = f"capped to [{lower:.2f}, {upper:.2f}]"
            elif method == 'remove':
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                summary['outliers'][col] = "removed"
    return df

# EDA analysis
def perform_eda(df):
    summary = {
        'numerical': [],
        'categorical': [],
        'distributions': {},
        'imputations': {},
        'outliers': {}
    }

    for col in df.columns:
        if df[col].dtype == 'object':
            summary['categorical'].append(col)
        else:
            summary['numerical'].append(col)
            summary['distributions'][col] = identify_distribution(df[col].dropna())

    return summary

# Feature scaling suggestion
def suggest_scaling_methods(df, summary):
    suggestions = {}
    for col in summary['numerical']:
        dist = summary['distributions'].get(col, 'skewed')
        if dist == 'normal':
            suggestions[col] = 'StandardScaler (Z-score)'
        else:
            suggestions[col] = 'MinMaxScaler / RobustScaler / Log Transformation'
    return suggestions

# Encoding suggestion
def suggest_encoding(df, summary):
    suggestions = {}
    for col in summary['categorical']:
        cardinality = df[col].nunique()
        if cardinality <= 10:
            suggestions[col] = 'One-Hot Encoding'
        elif cardinality <= 50:
            suggestions[col] = 'Ordinal Encoding'
        else:
            suggestions[col] = 'Target Encoding / Embedding'
    return suggestions

# ML recommendation
def recommend_algorithms(df):
    recommendation = {}
    target_col = df.columns[-1]
    target = df[target_col]

    if target.nunique() <= 1:
        return {'error': 'Target column is constant or invalid.'}

    target_type = type_of_target(target)

    if target_type in ['binary', 'multiclass']:
        recommendation['problem_type'] = 'classification'
        recommendation['algorithms'] = [
            'Logistic Regression',
            'Random Forest Classifier',
            'SVM',
            'Gradient Boosting',
            'XGBoost'
        ]
    elif target_type in ['continuous']:
        recommendation['problem_type'] = 'regression'
        recommendation['algorithms'] = [
            'Linear Regression',
            'Random Forest Regressor',
            'XGBoost Regressor',
            'SVR'
        ]
    else:
        recommendation['problem_type'] = 'unknown'

    recommendation['target_column'] = target_col
    recommendation['target_type'] = target_type
    recommendation['class_balance'] = dict(df[target_col].value_counts().to_dict()) \
        if target_type == 'binary' or target_type == 'multiclass' else None

    return recommendation

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 500

    summary = perform_eda(df)
    df = handle_missing_values(df, summary)
    df = treat_outliers(df, summary)

    scaling = suggest_scaling_methods(df, summary)
    encoding = suggest_encoding(df, summary)
    ml_recommendation = recommend_algorithms(df)

    cleaned_path = os.path.join(STATIC_FOLDER, 'cleaned_dataset.csv')
    df.to_csv(cleaned_path, index=False)

    return jsonify({
        'summary': summary,
        'scaling_suggestions': scaling,
        'encoding_suggestions': encoding,
        'ml_recommendation': ml_recommendation,
        'download_link': '/download'
    })

# Download cleaned file
@app.route('/download', methods=['GET'])
def download_file():
    return send_file(os.path.join(STATIC_FOLDER, 'cleaned_dataset.csv'), as_attachment=True)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
