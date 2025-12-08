import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Insurance Charge Predictor Dashboard", layout="wide")


#Load Model + Artifacts
# Get the directory where the script is located
# Use the current working directory as fallback for Streamlit apps
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "rf_model.pkl")
COLUMNS_PATH = os.path.join(SCRIPT_DIR, "models", "model_columns.pkl")


if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    st.error(f"Model or artifacts not found. Looking in: {SCRIPT_DIR}/models/")
    st.error(f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
    st.error(f"COLUMNS_PATH exists: {os.path.exists(COLUMNS_PATH)}")
    st.info("Ensure rf_model.pkl and model_columns.pkl exist inside /models/ directory.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

st.title("💰 Insurance Charge Prediction Dashboard")
st.write("Predict healthcare insurance charges for individuals or upload multiple records.")


#Tabs
tabs = st.tabs(["Single Prediction", "Batch Prediction & Visualization", "What-If Scenario"])


#Tab 1: Single Prediction
with tabs[0]:
    st.subheader("📌 Single Prediction")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=5.0, max_value=70.0, value=25.0, format="%.1f")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict", key="single"):
        # Manual encoding
        input_encoded = pd.DataFrame(columns=model_columns)
        input_encoded.loc[0] = 0
        input_encoded.at[0, "age"] = age
        input_encoded.at[0, "bmi"] = bmi
        input_encoded.at[0, "children"] = children
        if sex == "male":
            input_encoded.at[0, "sex_male"] = 1
        if smoker == "yes":
            input_encoded.at[0, "smoker_yes"] = 1
        if region == "northwest":
            input_encoded.at[0, "region_northwest"] = 1
        elif region == "southeast":
            input_encoded.at[0, "region_southeast"] = 1
        elif region == "southwest":
            input_encoded.at[0, "region_southwest"] = 1

        pred_log = model.predict(input_encoded)[0]
        pred_charge = np.expm1(pred_log)
        st.success(f"Predicted Insurance Charge: **${pred_charge:,.2f}**")

        with st.expander("View Encoded Input Features"):
            st.dataframe(input_encoded.T)


#Tab 2: Batch Prediction & Visualization
with tabs[1]:
    st.subheader("📂 Batch Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        # Encode batch
        encoded_rows = pd.DataFrame(columns=model_columns)
        for i, row in batch_df.iterrows():
            encoded_row = pd.Series(0, index=model_columns)
            encoded_row["age"] = row["age"]
            encoded_row["bmi"] = row["bmi"]
            encoded_row["children"] = row["children"]
            if str(row["sex"]).lower() == "male":
                encoded_row["sex_male"] = 1
            if str(row["smoker"]).lower() == "yes":
                encoded_row["smoker_yes"] = 1
            if str(row["region"]).lower() == "northwest":
                encoded_row["region_northwest"] = 1
            elif str(row["region"]).lower() == "southeast":
                encoded_row["region_southeast"] = 1
            elif str(row["region"]).lower() == "southwest":
                encoded_row["region_southwest"] = 1
            encoded_rows = pd.concat([encoded_rows, encoded_row.to_frame().T], ignore_index=True)

        preds_log = model.predict(encoded_rows)
        batch_df["Predicted_Charges"] = np.expm1(preds_log)
        st.success("Predictions completed!")
        st.dataframe(batch_df.head())

        # Download button
        csv = batch_df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions CSV", data=csv, file_name="predicted_charges.csv", mime="text/csv")

        # Visualizations
        st.subheader("📊 Visualizations")
        plt.figure(figsize=(8,4))
        sns.histplot(batch_df["Predicted_Charges"], bins=30, kde=True, color="skyblue")
        plt.xlabel("Predicted Charges")
        plt.title("Distribution of Predicted Charges")
        st.pyplot(plt)

        # By smoker
        smoker_summary = batch_df.groupby("smoker")["Predicted_Charges"].mean().reset_index()
        st.markdown("**Average Predicted Charges by Smoker Status**")
        st.dataframe(smoker_summary)

        # By region
        region_summary = batch_df.groupby("region")["Predicted_Charges"].mean().reset_index()
        st.markdown("**Average Predicted Charges by Region**")
        st.dataframe(region_summary)

        # By age group
        bins = [0, 18, 30, 45, 60, 120]
        labels = ["0-18","19-30","31-45","46-60","61+"]
        batch_df["age_group"] = pd.cut(batch_df["age"], bins=bins, labels=labels)
        age_summary = batch_df.groupby("age_group")["Predicted_Charges"].mean().reset_index()
        st.markdown("**Average Predicted Charges by Age Group**")
        st.dataframe(age_summary)

        # BMI vs Predicted Charges
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=batch_df, x="bmi", y="Predicted_Charges", hue="smoker", palette="Set1")
        plt.title("BMI vs Predicted Charges by Smoker Status")
        st.pyplot(plt)


# Tab 3: What-If Scenario

with tabs[2]:
    st.subheader("🔮 What-If Scenario Analysis")
    st.write("Adjust variables to see how predicted insurance charges change dynamically.")

    age_s = st.slider("Age", min_value=0, max_value=120, value=30)
    bmi_s = st.slider("BMI", min_value=5.0, max_value=70.0, value=25.0, step=0.5)
    children_s = st.slider("Number of Children", min_value=0, max_value=10, value=0)
    sex_s = st.selectbox("Sex", ["female", "male"], key="whatif_sex")
    smoker_s = st.selectbox("Smoker", ["no", "yes"], key="whatif_smoker")
    region_s = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], key="whatif_region")

    scenario_encoded = pd.DataFrame(columns=model_columns)
    scenario_encoded.loc[0] = 0
    scenario_encoded.at[0, "age"] = age_s
    scenario_encoded.at[0, "bmi"] = bmi_s
    scenario_encoded.at[0, "children"] = children_s
    if sex_s == "male":
        scenario_encoded.at[0, "sex_male"] = 1
    if smoker_s == "yes":
        scenario_encoded.at[0, "smoker_yes"] = 1
    if region_s == "northwest":
        scenario_encoded.at[0, "region_northwest"] = 1
    elif region_s == "southeast":
        scenario_encoded.at[0, "region_southeast"] = 1
    elif region_s == "southwest":
        scenario_encoded.at[0, "region_southwest"] = 1

    pred_log_s = model.predict(scenario_encoded)[0]
    pred_charge_s = np.expm1(pred_log_s)
    st.success(f"Predicted Insurance Charge: **${pred_charge_s:,.2f}**")

    # Optional line chart: varying BMI
    bmi_range = np.arange(15, 50, 1)
    preds_range = []
    scenario_encoded_temp = scenario_encoded.copy()
    for b in bmi_range:
        scenario_encoded_temp.at[0, "bmi"] = b
        preds_range.append(np.expm1(model.predict(scenario_encoded_temp)[0]))
    st.line_chart(pd.DataFrame({"BMI": bmi_range, "Predicted Charges": preds_range}).set_index("BMI"))
