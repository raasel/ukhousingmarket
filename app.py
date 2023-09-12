import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model


#pip3 install tensorflow-cpu --no-cache-dir
################## Menu Hide #################
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

########## END #############

#--------Data import
df=pd.read_csv('./data/average_price_2023.csv')
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)
df['Average_Price'] = pd.to_numeric(df['Average_Price'], errors='coerce')
df.dropna(subset=['Average_Price'], inplace=True)


dfav=pd.read_csv('./data/average_price_2023.csv')
# Convert the 'Date' column to datetime format
dfav['Date'] = pd.to_datetime(dfav['Date'], format='%d/%m/%Y')


dftype=pd.read_csv('./data/price_by_type_2023.csv')
# Convert the 'Date' column to datetime format
dftype['Date'] = pd.to_datetime(dftype['Date'], format='%d/%m/%Y')


dfsale=pd.read_csv('./data/sales_2023.csv')
# Convert the 'Date' column to datetime format
dfsale['Date'] = pd.to_datetime(dfsale['Date'], format='%d/%m/%Y')
# print(df['Date'].max())

dfinterest=pd.read_csv('./data/Bank_Interest_Rate.csv')

#--------Data Import done

# Sample dataset (You should replace this with your actual data)
data = pd.DataFrame({
    'area': ['Area1', 'Area2'],
    'price': [100000, 120000],
    'sale_price': [95000, 110000],
    'property_type': ['Type1', 'Type2']
})



def search_data_property_type(area):
    # Find the maximum date
    max_date = dftype['Date'].max()

    # Filter the rows with the maximum date and "Wales" as the Region_Name
    filtered_dftype = dftype[(dftype['Date'] == max_date) & (dftype['Region_Name'].str.lower() == area.lower())]
    
    return filtered_dftype


def search_data_average(area):
    # Find the maximum date
    max_date = dfav['Date'].max()

    # Filter the rows with the maximum date and "Wales" as the Region_Name
    filtered_dftype = dfav[(dfav['Date'] == max_date) & (dfav['Region_Name'].str.lower() == area.lower())]
    
    return filtered_dftype


def search_data_sales(area):
    # Find the maximum date
    max_date = dfsale['Date'].max()

    # Filter the rows with the maximum date and "Wales" as the Region_Name
    filtered_dftype = dfsale[(dfsale['Date'] == max_date) & (dfsale['Region_Name'].str.lower() == area.lower())]
    
    return filtered_dftype

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)




def predict_price(area):
    with st.spinner('Wait for Prediction...'):
        # Use 'Average_Price' for LSTM model
        df_england=dfav[dfav['Region_Name'].str.lower() == area.lower()]
        # Use only the 'Average_Price' column and normalize the values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_england[['Average_Price']])

        # Prepare the dataset for LSTM
        X, y = [], []
        for i in range(1, len(scaled_data)):
            X.append(scaled_data[i-1:i])
            y.append(scaled_data[i])

        X, y = np.array(X), np.array(y)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=100, batch_size=5)

        # Generate future dates (10 years ahead, annual data)
        future_years = 10

        # Initialize the future prediction array
        future_predictions = []

        # Use the last data point to make the first prediction, then use the prediction for subsequent ones
        current_input = np.array([scaled_data[-1]])

        for i in range(future_years):
            prediction = model.predict(current_input.reshape(1, 1, 1))
            future_predictions.append(prediction)
            current_input = prediction

        # Inverse transform the scaled data back to original scale
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Generate future dates
        future_dates = pd.date_range(start='2023-07-01', end='2033-06-01', freq='A')

        # Create a DataFrame for future predictions
        df_future = pd.DataFrame({
            'Date': future_dates,
            'Region_Name': ['England'] * future_years,
            'Average_Price': future_predictions.flatten()
        })

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df_england['Date'], df_england['Average_Price'], label='Existing Data', marker='o')
        plt.plot(df_future['Date'], df_future['Average_Price'], label='LSTM Predictions', linestyle='--', marker='x')
        plt.xlabel('Year')
        plt.ylabel('Average Price')
        plt.title(f'Average House Price in {area.upper()} (Existing and LSTM 10 Year Predictions)')
        plt.legend()
        st.pyplot(plt)


       

def predict_price_overview(area):
    with st.spinner('Wait for Prediction...'):
        # Use 'Average_Price' for LSTM model
        df_england=dfav[dfav['Region_Name'].str.lower() == area.lower()]
        # Use only the 'Average_Price' column and normalize the values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_england[['Average_Price']])

        # Prepare the dataset for LSTM
        X, y = [], []
        for i in range(1, len(scaled_data)):
            X.append(scaled_data[i-1:i])
            y.append(scaled_data[i])

        X, y = np.array(X), np.array(y)

        # # Build the LSTM model
        # model = Sequential()
        # model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        # model.add(LSTM(units=50))
        # model.add(Dense(units=1))

        # model.compile(optimizer='adam', loss='mean_squared_error')

        # # Train the model
        # model.fit(X, y, epochs=100, batch_size=5)

        # # Saving the model
        # model.save(area+"_model", save_format='tf')


        # # import the model
        model = load_model(area+"_model")

        # Generate future dates (10 years ahead, annual data)
        future_years = 10

        # Initialize the future prediction array
        future_predictions = []

        # Use the last data point to make the first prediction, then use the prediction for subsequent ones
        current_input = np.array([scaled_data[-1]])

        for i in range(future_years):
            prediction = model.predict(current_input.reshape(1, 1, 1))
            future_predictions.append(prediction)
            current_input = prediction

        # Inverse transform the scaled data back to original scale
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Generate future dates
        future_dates = pd.date_range(start='2023-07-01', end='2033-06-01', freq='A')

        # Create a DataFrame for future predictions
        df_future = pd.DataFrame({
            'Date': future_dates,
            'Region_Name': ['England'] * future_years,
            'Average_Price': future_predictions.flatten()
        })

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df_england['Date'], df_england['Average_Price'], label='Existing Data', marker='o')
        plt.plot(df_future['Date'], df_future['Average_Price'], label='LSTM Predictions', linestyle='--', marker='x')
        plt.xlabel('Year')
        plt.ylabel('Average Price')
        plt.title(f'Average House Price in {area} (Existing and LSTM 10 Year Predictions)')
        plt.legend()
        st.pyplot(plt)
 
       

        





    # return prediction



st.image("logo/logousw.jpg", width=100)

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
    if st.button("Search") or area:
        result = search_data_property_type(area)
        result_average = search_data_average(area)

        result_sales = search_data_sales(area)

        if len(result) > 0:
            # # Display line graph
            # fig, ax = plt.subplots()
            # ax.plot(result['area'], result['price'], label="Property Price")
            # ax.plot(result['area'], result['sale_price'], label="Sale Price")
            # ax.legend()
            # st.pyplot(fig)

            # Inline CSS
            # Inline CSS
            card_style = """
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); 
            transition: 0.3s; 
            width: 40%; 
            margin: 10px;
            background-color: white;
            """

            container_style = """
            padding: 2px 16px;
            text-align: center;
            """

            # Flex container for row alignment
            flex_container_style = """
            display: flex; 
            justify-content: space-around;
            """

            # HTML1 with Inline CSS
            card_html = f"""
            <div style='{flex_container_style}'>
                <!-- Card 1 -->
                <div style='{card_style}' onmouseover="this.style.boxShadow='0 8px 16px 0 rgba(0,0,0,0.2)';" onmouseout="this.style.boxShadow='0 4px 8px 0 rgba(0,0,0,0.2)';">
                    <div style='{container_style}'>
                        <h4><b>Detached House Average Price</b></h4>
                        <p>£{result['Detached_Average_Price'].mean()}</p>
                    </div>
                </div>

                <!-- Card 2 -->
                <div style='{card_style}' onmouseover="this.style.boxShadow='0 8px 16px 0 rgba(0,0,0,0.2)';" onmouseout="this.style.boxShadow='0 4px 8px 0 rgba(0,0,0,0.2)';">
                    <div style='{container_style}'>
                        <h4><b>Semi Detached House Average Price</b></h4>
                        <p>£{result['Semi_Detached_Average_Price'].mean()}</p>
                    </div>
                </div>
            </div>
            """

            # Display in Streamlit using components
            components.html(card_html, height=140)

            # HTML2 with Inline CSS
            card_html2 = f"""
            <div style='{flex_container_style}'>
                <!-- Card 1 -->
                <div style='{card_style}' onmouseover="this.style.boxShadow='0 8px 16px 0 rgba(0,0,0,0.2)';" onmouseout="this.style.boxShadow='0 4px 8px 0 rgba(0,0,0,0.2)';">
                    <div style='{container_style}'>
                        <h4><b>Terraced House Average Price</b></h4>
                        <p>£{result['Terraced_Average_Price'].mean()}</p>
                    </div>
                </div>

                <!-- Card 2 -->
                <div style='{card_style}' onmouseover="this.style.boxShadow='0 8px 16px 0 rgba(0,0,0,0.2)';" onmouseout="this.style.boxShadow='0 4px 8px 0 rgba(0,0,0,0.2)';">
                    <div style='{container_style}'>
                        <h4><b>Flat House Average Price</b></h4>
                        <p>£{result['Flat_Average_Price'].mean()}</p>
                    </div>
                </div>
            </div>
            """

            # Display in Streamlit using components
            components.html(card_html2, height=140)

            # HTML2 with Inline CSS
            card_html3 = f"""
            <div style='{flex_container_style}'>
                <!-- Card 1 -->
                <div style='{card_style}' onmouseover="this.style.boxShadow='0 8px 16px 0 rgba(0,0,0,0.2)';" onmouseout="this.style.boxShadow='0 4px 8px 0 rgba(0,0,0,0.2)';">
                    <div style='{container_style}'>
                        <h4><b>Average House Price</b></h4>
                        <p>£{result_average['Average_Price'].mean()}</p>
                    </div>
                </div>

                <!-- Card 2 -->
                <div style='{card_style}' onmouseover="this.style.boxShadow='0 8px 16px 0 rgba(0,0,0,0.2)';" onmouseout="this.style.boxShadow='0 4px 8px 0 rgba(0,0,0,0.2)';">
                    <div style='{container_style}'>
                        <h4><b>Sales Amount</b></h4>
                        <p>{result_sales['Sales_Volume'].mean()}</p>
                    </div>
                </div>
            </div>
            """

            # Display in Streamlit using components
            components.html(card_html3, height=140)


            st.success(f"Price Data Revision Date: {result.iloc[0]['Date']}")
            st.success(f"Sales Data Revision Date: {result_sales.iloc[0]['Date']}")
            st.subheader(f"Region Name: {result.iloc[0]['Region_Name']}")


            #Making Prediction
            st.header(f"Prediction Making The Average Price in {result.iloc[0]['Region_Name']}")
            predict_price(area)





            # date=str(result['Date'])
            # date= date[4:18]


            st.warning(f"Note: This Website is Only Suitable for Light Themes.")
          
            # st.write(f"Predicted Price: £{predict_price(result)}")




        else:
            st.write("No data found for this Region.")




elif navigation == "Overview of Housing Market":
    st.header("Overview")
    # Display other line graphs

    ##-----Bank Interest Line Graph
    # Filter the data for the last 20 years
    # Convert the 'Date' column to datetime
    st.header("Interest Rates Throughout the Year")
    dfinterest['Date'] = pd.to_datetime(dfinterest['Date'])
    df_last_20_years = dfinterest[dfinterest['Date'].dt.year >= 1983]

    # Plot the data
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df_last_20_years['Date'], df_last_20_years['New_rate'], marker='o', linestyle='-')
    ax.set_title('Interest Rate Vs Year')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rate')

    # Setting x-axis ticks for each year
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    #--------End
    #-------------Country Wise
    st.header("Average Price per 5 Years for Each Country")
    dfline=df
    # List of countries you're interested in
    countries = ['England', 'Wales', 'Scotland','Northern Ireland']

    plt.figure(figsize=(10,6))

    for country in countries:
        # Filter data for each country
        df_country = dfline[dfline['Region_Name'] == country]

        # Resample your data to get average price per year
        df_yearly = df_country.resample('Y').mean(numeric_only=True)

        # Filter out data points that are not 5 years apart
        df_5year = df_yearly[df_yearly.index.year % 5 == 0]

        # Plot your data
        plt.plot(df_5year.index, df_5year['Average_Price'], marker='o', label=country)

    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.title('Average Price per 5 Years for Each Country')
    plt.grid(True)
    plt.legend()
    # plt.show()
    st.pyplot(plt)

    #-------End

    #------------Capital of Each Country
    #Capital of Country Wise
    st.header("Average Price per 5 Years for Each Capital")
    dfline=df
    # List of countries you're interested in
    countries = ['London', 'Cardiff', 'City of Edinburgh','Belfast']

    plt.figure(figsize=(10,6))

    for country in countries:
        # Filter data for each country
        df_country = dfline[dfline['Region_Name'] == country]

        # Resample your data to get average price per year
        df_yearly = df_country.resample('Y').mean(numeric_only=True)

        # Filter out data points that are not 5 years apart
        df_5year = df_yearly[df_yearly.index.year % 5 == 0]

        # Plot your data
        plt.plot(df_5year.index, df_5year['Average_Price'], marker='o', label=country)

    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.title('Average Price per 5 Years for Each Capital')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)



    #------------------End
    ##-----------Prediction of English House Price
    st.header(f"Prediction of The Average Price in England")
    predict_price_overview("England")
    ###-----------End
    ##-----------Prediction of Wales House Price
    st.header(f"Prediction of The Average Price in Wales")
    predict_price_overview("Wales")
    ###-----------End
    ##-----------Prediction of Scotland House Price
    st.header(f"Prediction of The Average Price in Scotland")
    predict_price_overview("Scotland")
    ###-----------End




