import pandas as pd
import numpy as np
import joblib
import os

# Path ke model dan fitur
MODEL_PATH = os.path.join('model', 'model_logreg.pkl')
FEATURES_PATH = os.path.join('model', 'features_list.npy')

# Load model dan fitur
model = joblib.load(MODEL_PATH)
features = np.load(FEATURES_PATH, allow_pickle=True)

def predict_attrition(input_dict):
    """
    input_dict: dict, key=nama kolom, value=nilai (sudah sesuai hasil one-hot encoding)
    return: prediksi (0/1), probabilitas keluar
    """
    df_input = pd.DataFrame([input_dict])
    # Pastikan urutan dan kolom sama dengan training
    df_input = df_input.reindex(columns=features, fill_value=0)
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    return int(pred), float(prob)

if __name__ == '__main__':
    # Contoh input manual (isi sesuai fitur hasil one-hot encoding)
    input_data = {
        'Age': 35,
        'DailyRate': 800,
        'DistanceFromHome': 5,
        'Education': 3,
        'EnvironmentSatisfaction': 2,
        'HourlyRate': 60,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'JobSatisfaction': 3,
        'MonthlyIncome': 5000,
        # ...tambahkan semua fitur yang digunakan pada training, termasuk hasil one-hot encoding
        # Contoh:
        'JobRole_Research Scientist': 1,
        'JobRole_Sales Executive': 0,
        'MaritalStatus_Married': 1,
        'OverTime_Yes': 0
        # dst.
    }
    pred, prob = predict_attrition(input_data)
    print(f"Prediksi Attrition: {pred} (Probabilitas keluar: {prob:.2f})")
