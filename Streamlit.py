import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer

# Load your pre-trained model and dataset
model = load_model("D:/Documents/UTeM2024/myann_model.h5")  # Replace with the actual path to your saved model

# Load the dataset (use your actual dataset paths here)
df = pd.read_csv("D:/Documents/UTeM2024/Trainingdata2.csv")
df2 = pd.read_csv("D:/Documents/UTeM2024/Testingdata40row.csv")

# Preprocess the data
df = df.drop(['BizDate', 'Description'], axis=1)
df2 = df2.drop(['BizDate', 'Description'], axis=1)

# Target Encoder and Standard Scaler for preprocessing
categorical_cols = ['Loc_group', 'Day', 'Division', 'Dept', 'SubDept', 'Category', 'Weekend_Status', 'FestivePeriod']
target_encoder = ce.TargetEncoder(cols=categorical_cols)
X_train = df.drop(['DiscountPercent'], axis=1)
y_train = df['DiscountPercent']
X_train_encoded = target_encoder.fit_transform(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Prepare the test data
X_test = df2.drop(['DiscountPercent'], axis=1)  # Features for Testing
y_test = df2['DiscountPercent']                # Target for Testing
X_test_encoded = target_encoder.transform(X_test)
X_test_scaled = scaler.transform(X_test_encoded)

# Create LimeTabularExplainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,                 # ใช้ข้อมูลที่สเกลแล้วสำหรับการเทรน
    feature_names=X_train.columns.tolist(),      # ชื่อของฟีเจอร์
    mode='regression'                            # ใช้ regression mode
)

# Prediction and LIME explanation loop
y_pred = model.predict(X_test_scaled)  # Get predictions for the test data

# Streamlit UI
st.title("Dynamic Pricing Prediction")
# 1. Date Input to allow the user to select a date
selected_date = st.date_input("Select Date", pd.to_datetime('today'))

# You can use the selected date to filter or adjust data here if needed
# For now, it's displayed on the UI
st.write(f"Selected Date: {selected_date}")

# 2. Dropdown for Division
division = st.selectbox("Select Division", df['Division'].unique())

# 3. Filter data based on Division
filtered_depts = df[df['Division'] == division]['Dept'].unique()
dept = st.selectbox("Select Dept", filtered_depts)

# 4. Filter data based on Dept
filtered_subdepts = df[df['Dept'] == dept]['SubDept'].unique()
subdept = st.selectbox("Select SubDept", filtered_subdepts)

# 5. Filter data based on SubDept
filtered_categories = df[df['SubDept'] == subdept]['Category'].unique()
category = st.selectbox("Select Category", filtered_categories)

# 7. Automatically display PackSize and WeightPrice based on selections
selected_data = df[(df['Division'] == division) & (df['Dept'] == dept) & (df['SubDept'] == subdept) & (df['Category'] == category)]
if not selected_data.empty:
    pack_size = selected_data['PackSize'].values[0]
    st.write(f"PackSize: {pack_size}")

    # Prediction Button
    predict_button = st.button("Predict Price and Discount")

    if predict_button:
        # Predict Discount using the pre-trained model
        selected_features = selected_data.drop(['DiscountPercent'], axis=1)
        selected_features_encoded = target_encoder.transform(selected_features)
        selected_features_scaled = scaler.transform(selected_features_encoded)
        predicted_discount = model.predict(selected_features_scaled)
        predicted_price = selected_data['UnitPrice'].values[0] * (1 - predicted_discount[0][0] / 100)

        st.write(f"Predicted Discount: {predicted_discount[0][0]:.2f}%")
        st.write(f"Predicted Price: {predicted_price:.2f}")

        # LIME explanation for the prediction
        lime_exp = lime_explainer.explain_instance(
            data_row=selected_features_scaled[0],  # Pass the scaled data
            predict_fn=model.predict,  # Model's predict function
            num_features=10  # Number of features to show in explanation
        )
        
        # Show LIME explanation as HTML
        lime_html = lime_exp.as_html()
        st.subheader("LIME Explanation for Prediction")
        st.components.v1.html(lime_html, height=800)
