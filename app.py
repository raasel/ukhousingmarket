import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#--------Data import
df=pd.read_csv('./data/average_price_2023.csv')
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)
df['Average_Price'] = pd.to_numeric(df['Average_Price'], errors='coerce')
df.dropna(subset=['Average_Price'], inplace=True)


dftype=pd.read_csv('./data/price_by_type_2023.csv')
# Convert the 'Date' column to datetime format
dftype['Date'] = pd.to_datetime(dftype['Date'], format='%d/%m/%Y')


dfsale=pd.read_csv('./data/sales_2023.csv')
# Convert the 'Date' column to datetime format
dfsale['Date'] = pd.to_datetime(dfsale['Date'], format='%d/%m/%Y')
# print(df['Date'].max())
#--------Data Import done

# Sample dataset (You should replace this with your actual data)
data = pd.DataFrame({
    'area': ['Area1', 'Area2'],
    'price': [100000, 120000],
    'sale_price': [95000, 110000],
    'property_type': ['Type1', 'Type2']
})



def search_data(area):
    # Find the maximum date
    max_date = dftype['Date'].max()

    # Filter the rows with the maximum date and "Wales" as the Region_Name
    filtered_dftype = dftype[(dftype['Date'] == max_date) & (dftype['Region_Name'].str.lower() == area.lower())]
    
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
        dfRegion=df[df['Region_Name'] == area]
        dataset = dfRegion['Average_Price'].values
        dataset = dataset.astype('float32')

        # Reshape to be [samples, time steps, features]
        dataset = np.reshape(dataset, (-1, 1))

        # Normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # Let's use 80% data for training and 30% for testing
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]



        # Use look_back to decide the number of previous time steps to use as input variables to predict the next time period
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back))) # 4 is the number of neurons, you can change it as you want
        model.add(Dense(1)) # output layer
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) # adjust epochs and batch_size as needed

        # Make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # Invert predictions back to normal values
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        # Shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        dates = dfRegion.index.to_numpy() # assuming the 'Date' is your index

        # plt.figure(figsize=(15, 8))

        plt.plot(dates, scaler.inverse_transform(dataset), label='Original Data')
        plt.plot(dates, trainPredictPlot, label='Train Prediction')
        plt.plot(dates, testPredictPlot, label='Test Prediction')

        # Adjusting for better date format
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)

        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", fontsize=10)

        plt.legend()
        plt.title("Housing Market Predictions from 1969 to 2023")
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        st.pyplot(plt)


        # Calculate root mean squared error
        trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        st.write('Train Score: %.2f RMSE' % (trainScore))
        testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        st.write('Test Score: %.2f RMSE' % (testScore))

def predict_price_overview(area):
    with st.spinner('Wait for Prediction...'):
        # Use 'Average_Price' for LSTM model
        dfRegion=df[df['Region_Name'] == area]
        dataset = dfRegion['Average_Price'].values
        dataset = dataset.astype('float32')

        # Reshape to be [samples, time steps, features]
        dataset = np.reshape(dataset, (-1, 1))

        # Normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # Let's use 80% data for training and 30% for testing
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]



        # Use look_back to decide the number of previous time steps to use as input variables to predict the next time period
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back))) # 4 is the number of neurons, you can change it as you want
        model.add(Dense(1)) # output layer
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) # adjust epochs and batch_size as needed

        # Make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # Invert predictions back to normal values
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        # Shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        dates = dfRegion.index.to_numpy() # assuming the 'Date' is your index

        plt.figure(figsize=(15, 8))

        plt.plot(dates, scaler.inverse_transform(dataset), label='Original Data')
        plt.plot(dates, trainPredictPlot, label='Train Prediction')
        plt.plot(dates, testPredictPlot, label='Test Prediction')

        # Adjusting for better date format
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)

        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", fontsize=10)

        plt.legend()
        plt.title("Housing Market Predictions from 1969 to 2023")
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        st.pyplot(plt)


        # Calculate root mean squared error
        trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        st.write('Train Score: %.2f RMSE' % (trainScore))
        testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        st.write('Test Score: %.2f RMSE' % (testScore))




    # return prediction


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
        result = search_data(area)
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
            components.html(card_html, height=200)

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
            components.html(card_html2, height=200)

            #Making Prediction
            st.header(f"Prediction Making The Average Price in {result.iloc[0]['Region_Name']}")
            predict_price(area)





            st.write(f"Region Name: {result.iloc[0]['Region_Name']}")
            # date=str(result['Date'])
            # date= date[4:18]
            st.success(f"Data Revision Date: {result.iloc[0]['Date']}")

            st.warning(f"Note: This Website is Only Suitable for Light Themes.")
          
            # st.write(f"Predicted Price: £{predict_price(result)}")




        else:
            st.write("No data found for this Region.")




elif navigation == "Overview of Housing Market":
    st.header("Overview")
    # Display other line graphs
    #-------------Country Wise
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




