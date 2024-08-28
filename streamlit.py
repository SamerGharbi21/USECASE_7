import streamlit as st
import requests

# Set the URL of your FastAPI application
API_URL = "https://usecase-7.onrender.com/predict"  # Update with your FastAPI URL

# Set the title for the Streamlit app
st.title("DBSCAN Model Prediction Interface")

# Input fields for user to enter data using sliders
age = st.slider("Select Age", min_value=0, max_value=100, value=25)
current_value = st.slider("Select Current Value", min_value=0, max_value=1_000_000_000_0, value=5_000)

# Button to trigger the prediction
if st.button("Predict Cluster"):
    # Prepare data to send to the FastAPI backend
    input_data = {
        "age": age,
        "current_value": current_value
    }
    
    try:
        # Make a POST request to the FastAPI endpoint
        response = requests.post(API_URL, json=input_data)
        
        # Print the response details for debugging
        #st.write(f"Response Status Code: {response.status_code}")
        #st.write(f"Response Content: {response.text}")
        
        # Check the response status
        if response.status_code == 200:
            # Extract and display the predicted cluster
            result = response.json()
            cluster = result['cluster']
            st.success(f"The predicted cluster is: {cluster}")
        else:
            st.error("Error in prediction. Please check your input and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
