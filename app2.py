### New App For Car Pricing ###
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st




#### Page Config ###
st.set_page_config(
    page_title="Car Costs Calculator",
    page_icon="https://img.freepik.com/premium-vector/car-icon-logo-element-illustration-car-symbol-design-from-2-colored-collection-simple-car-concept-can-be-used-web-mobile_159242-5040.jpg?w=2000",
    menu_items={
        "Get help": "mailto:hikmetemreguler@gmail.com",
        "About": "For More Information\n" + "https://github.com/HikmetEmre/Project_3"
    }
)

### Title of Project ###
st.title(" A Model For Used Car Price Prediction ")

### Markdown ###
st.markdown("Used Car Marketplace is **:red[DODGY]** and **:blue[QUITE POUPLAR]**  nowadays so via using this Machine Learning Model we can predict car prices with high accuracy! .")

### Adding Image ###
st.image("https://etimg.etb2bimg.com/photo/74218074.cms")

st.markdown("The advantage of having **a machine learning model with %92 ACCURACY** to predict used car prices is its ability to analyze vast amounts of data and identify complex patterns, leading to more accurate and reliable price estimations.")
st.markdown("In addition, having a machine learning model to predict used car prices based on new car information provides the advantage of accurately estimating the price, enabling us to offer tailored products and services that align with the predicted value.")
st.markdown("*Alright, Let's Dive In!*")

st.image("https://cdn.pixabay.com/photo/2014/09/22/07/08/bargain-455988_960_720.png")

#### Header and definition of columns ###
st.header("META DATA")

st.markdown("- **Km**: The total number of kilometers traveled.")
st.markdown("- **EngSize**: Engine size is the volume of fuel and air that can be pushed through a car's cylinders.")
st.markdown("- **Hp**:The power an engine produces.")
st.markdown("- **Fuel Consumption**: Average consumption = (fuel used / number of kilometres) x 100")
st.markdown("- **Price**: Money Value of The Car. ")
st.markdown("- **Age**: Number of the years in between and including both the calendar year and model year.")
st.markdown("- **Fiat**: A Car Brand.")
st.markdown("- **Ford**: A Car Brand.")
st.markdown("- **Renault**: A Car Brand.")
st.markdown("- **Toyota**: A Car Brand.")
st.markdown("- **Volkswagen**: A Car Brand.")
st.markdown("- **Manuel**: Gear Type of The Car.")
st.markdown("- **Automatic** Gear Type of The Car.")
st.markdown("- **Semi-Automatic**: Gear Type of The Car.")
st.markdown("- **Benzin**: Fuel Type of The Car.")
st.markdown("- **Diesel**: Fuel Type of The Car.")
st.markdown("- **Hybrid**: Fuel Type of The Car.")
st.markdown("- **Benzin&LPG**: Fuel Type of The Car.")


### Example DF ON STREAMLIT PAGE ###
df=pd.read_csv("app_version_cars.csv")
df.drop(["Unnamed: 0"],axis=1,inplace=True)


### Example TABLE ###
st.table(df.sample(5, random_state=18))

#---------------------------------------------------------------------------------------------------------------------

car_make_options = ['Fiat', 'Ford', 'Renault', 'Toyota', 'Volkswagen']
gear_options = ['Manuel Gear', 'Automatic Gear', 'Semi-Auto Gear']
fuel_options = ['Benzin', 'Diesel', 'Hybrid', 'Benzin&LPG']

Km = st.sidebar.number_input("**Numeric value of total trip.**", min_value=0)
EngSize = st.sidebar.number_input("**Volume of the car's engine.**", min_value=0)
Hp = st.sidebar.number_input("**Power of Engine.**", min_value=0)
Fuel_Consumption = st.sidebar.number_input("**Numeric value of consumption**.", min_value=0)
Age = st.sidebar.number_input("**Age of the car**.", min_value=0)

Brands = st.sidebar.selectbox("Car Make", car_make_options)
Fiat = 1 if Brands == "Fiat" else 0
Ford = 1 if Brands == "Ford" else 0
Renault = 1 if Brands == "Renault" else 0
Toyota = 1 if Brands == "Toyota" else 0
Volkswagen = 1 if Brands == "Volkswagen" else 0

gear_type = st.sidebar.selectbox("Gear Type", gear_options)
Manuel = 1 if gear_type == "Manuel Gear" else 0
Automatic = 1 if gear_type == "Automatic Gear" else 0
Semi_Automatic = 1 if gear_type == "Semi-Auto Gear" else 0

fuel_type = st.sidebar.selectbox("Fuel Type", fuel_options)
Benzin = 1 if fuel_type == "Benzin" else 0
Diesel = 1 if fuel_type == "Diesel" else 0
Hybrid = 1 if fuel_type == "Hybrid" else 0
Benzin_LPG = 1 if fuel_type == "Benzin&LPG" else 0


#---------------------------------------------------------------------------------------------------------------------

### Recall Model ###
from joblib import load

regression_model = load('lr_model_lastest.pkl')

input_df = [[Km,EngSize,Hp,Fuel_Consumption,Age,Fiat,Ford,Renault,Toyota,Volkswagen,Manuel,Automatic,Semi_Automatic,Benzin,Diesel,Hybrid,Benzin_LPG]]
    
### For fit StandartScaler ###
df=pd.read_csv("app_version_cars.csv")

# Define X and y
X = df[['Km', 'EngSize', 'Hp', 'Fuel Consumption', 'Age', 'Fiat',
       'Ford', 'Renault', 'Toyota', 'Volkswagen', 'Manuel', 'Automatic',
       'Semi-Automatic', 'Benzin', 'Diesel', 'Hybrid', 'Benzin&LPG']]

y = np.log(df["Price"]) 


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



### Scale the new input data###

input_df_scaled = scaler.transform(input_df)

pred = regression_model.predict(input_df_scaled)

result = np.log(pred)*np.log(pred)
lira_symbol = '\u20BA'
result_str = lira_symbol + str(result)




#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

### Result Screen ###
if st.sidebar.button("Submit"):

    ### Info message ###
    st.info("You can find the result below.")

    ### Inquiry Time Info ###
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    ### For showing results create a df ###
    results_df = pd.DataFrame({
    'Date': [today],
    'Time': [time],
    'Price of Car': [result_str]
    })

   


    st.table(results_df)
if result_str is not None:
    st.image("https://static.vecteezy.com/system/resources/previews/005/576/332/original/car-icon-car-icon-car-icon-simple-sign-free-vector.jpg")
else:
     st.markdown("**Please click the *Submit Button!**")
