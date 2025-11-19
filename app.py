import streamlit as st
import pickle
import pandas as pd

# Load model
with open('Student_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data if needed
@st.cache_data
def load_data():
    df = pd.read_csv('Employee_clean_D.csv')
    return df

df = load_data()

st.title("Student Model Prediction App")

st.write("Enter the feature values below:")

# Auto-generate input fields from model expected features
# Assuming the model was trained on dataframe columns excluding target
feature_columns = [col for col in df.columns if col.lower() not in ["target", "label", "output"]]

user_input = {}

for col in feature_columns:
    if df[col].dtype in ["int64", "float64"]:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    else:
        options = df[col].unique().tolist()
        val = st.selectbox(f"{col}", options)
    user_input[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
