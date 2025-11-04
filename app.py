import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("🔢 Linear Regression Model Predictor")

st.write("""
This app uses a **Linear Regression Model** to make predictions based on your input values.  
Adjust the input sliders or boxes below to see the predicted result.
""")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Example input fields (you can customize based on your dataset features)
# Replace with your actual feature names and ranges
def user_input_features():
    feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=10.0)
    feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=20.0)
    feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, value=30.0)
    
    features = np.array([[feature1, feature2, feature3]])
    return features

# Get user input
input_data = user_input_features()

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted Output: {prediction[0]:.2f}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn.")
