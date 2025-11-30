#streamlit presentation
import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("Random Forest_model.pkl", "rb") as f:   # replace with your actual model file name
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

#Load Scaler
with open("Scaler.pkl", "rb") as f:
    Scaler = pickle.load(f) 

#load randonforest classifier
#Load Scaler
with open("rf_top15.pkl", "rb") as f:
    model = pickle.load(f) 

st.title("🌲 Forest Type Prediction")

st.write("Enter the values for the following features to predict the forest type:")

# Collecting user inputs
Elevation = st.slider("Elevation", min_value=2360.00, max_value=3900.00, value=2360.00)
st.write("Enter values max = 3900.00")
Aspect = st.slider("Aspect", min_value=0, max_value=360, value=0)
Slope = st.slider("Slope", min_value=0, max_value=27, value=0)
st.write("Enter values max = 27")
Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance To Hydrology", min_value=0, max_value=763, value=0)
st.write("Enter values max = 763")
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance To Hydrology", min_value=0, max_value=762, value=0)
Scaled_hydolody = Scaler.transform([[Vertical_Distance_To_Hydrology]])
st.write("Enter values max = 762")
Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance To Roadways", min_value=0, max_value=54, value=0)
st.write("Enter values max = 54")
Hillshade_9am = st.slider("Hillshade 9am", min_value=169, max_value=255, value=169)
st.write("Enter values max = 255")
Hillshade_Noon = st.slider("Hillshade Noon", min_value=186, max_value=254, value=186)
st.write("Enter values max = 254")
Hillshade_3pm = st.slider("Hillshade 3pm", min_value=64, max_value=216, value=64)
st.write("Enter values max = 216")
Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance To Fire Points", min_value=0, max_value=7173, value=0)
st.write("Enter values max = 7173")
Wilderness_Area_1 = st.selectbox("Wilderness Area1 ", [0, 1])
Wilderness_Area_2 = st.selectbox("Wilderness Area 2", [0, 1])
Wilderness_Area_3 = st.selectbox("Wilderness Area 3", [0, 1])
Wilderness_Area_4 = st.selectbox("Wilderness Area 4", [0, 1])
Soil_Type_1 = st.selectbox("Soil Type 1", [0, 1])


# Arrange inputs into feature vector
features = np.array([[
    Elevation,
    Aspect,
    Slope,
    Horizontal_Distance_To_Hydrology,
    Vertical_Distance_To_Hydrology,
    Horizontal_Distance_To_Roadways,
    Hillshade_9am,
    Hillshade_Noon,
    Hillshade_3pm,
    Horizontal_Distance_To_Fire_Points,
    Wilderness_Area_1,
    Wilderness_Area_2,
    Wilderness_Area_3,
    Wilderness_Area_4,
    Soil_Type_1

]])

# Predict
if st.button("Predict"):
    prediction_encoded = model.predict(features)   # This gives encoded number
    st.write(prediction_encoded)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    st.write(prediction_label)
    st.balloons()
    st.success(f"🌳 Predicted Forest Type: **{prediction_label[0]}**")