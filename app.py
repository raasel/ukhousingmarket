import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (You should replace this with your actual data)
data = pd.DataFrame({
    'area': ['Area1', 'Area2'],
    'price': [100000, 120000],
    'sale_price': [95000, 110000],
    'property_type': ['Type1', 'Type2']
})

def load_data():
    # Replace with your data loading logic
    return data

def search_data(area):
    # Replace with your data searching logic
    return data[data['area'] == area]

def predict_price(data):
    # Replace with a prediction model
    model = LinearRegression()
    X = data[['price']]
    y = data['sale_price']
    model.fit(X, y)
    prediction = model.predict([[data['price'].iloc[-1]]])[0]
    return prediction

data = load_data()
# Navigation
st.title("UK HOUSING MARKET")

# Default page
navigation = st.session_state.get("navigation", "Home")

# Side by side button navigation
col1, col2 = st.columns(2)
if col1.button("Home"):
    st.session_state.navigation = "Home"
    navigation = "Home"
if col2.button("Overview"):
    st.session_state.navigation = "Overview of Housing Market"
    navigation = "Overview of Housing Market"


if navigation == "Home":
    st.header("Search by Region Name")
    area = st.text_input("Enter the Region Name:")
    if st.button("Search"):
        result = search_data(area)
        if len(result) > 0:
            # Display line graph
            fig, ax = plt.subplots()
            ax.plot(result['area'], result['price'], label="Property Price")
            ax.plot(result['area'], result['sale_price'], label="Sale Price")
            ax.legend()
            st.pyplot(fig)

            # Displaying other stats
            st.write(f"Average Property Price: £{result['price'].mean()}")
            st.write(f"Average Sale Price: £{result['sale_price'].mean()}")
            st.write(f"Predicted Price: £{predict_price(result)}")
        else:
            st.write("No data found for this Region.")

elif navigation == "Overview of Housing Market":
    st.header("Overview")
    # Display other line graphs
    fig, ax = plt.subplots()
    ax.plot(data['area'], data['price'], label="Property Price")
    ax.legend()
    st.pyplot(fig)
