import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Customer Churn Analysis Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('telco-customer-churn.csv')
    # Convert TotalCharges to numeric, replacing empty strings with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill NaN values with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def create_eda_plots(df):
    # Create tabs for different categories of plots
    tab1, tab2, tab3 = st.tabs(["Customer Demographics", "Service Usage", "Financial Analysis"])
    
    with tab1:
        st.subheader("Customer Demographics Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution by churn
            fig_gender = px.pie(df, names='gender', title='Gender Distribution',
                              hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_gender)

        with col2:
            # Senior Citizen distribution
            fig_senior = px.pie(df, names='SeniorCitizen', title='Senior Citizen Distribution',
                              hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_senior)

        # Tenure distribution by churn
        fig_tenure = px.histogram(df, x='tenure', color='Churn', 
                                title='Customer Tenure Distribution by Churn Status',
                                marginal="box")
        st.plotly_chart(fig_tenure, use_container_width=True)

    with tab2:
        st.subheader("Service Usage Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Internet Service distribution
            internet_service_counts = df['InternetService'].value_counts()
            fig_internet = px.bar(x=internet_service_counts.index, 
                                y=internet_service_counts.values,
                                title='Internet Service Distribution',
                                labels={'x': 'Service Type', 'y': 'Count'})
            st.plotly_chart(fig_internet)

        with col2:
            # Contract type distribution
            contract_counts = df['Contract'].value_counts()
            fig_contract = px.bar(x=contract_counts.index, 
                                y=contract_counts.values,
                                title='Contract Type Distribution',
                                labels={'x': 'Contract Type', 'y': 'Count'})
            st.plotly_chart(fig_contract)

        # Services usage heatmap
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        service_corr = df[service_columns].apply(lambda x: pd.factorize(x)[0]).corr()
        
        fig_heatmap = px.imshow(service_corr, 
                               title='Service Usage Correlation Heatmap',
                               color_continuous_scale='RdBu')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab3:
        st.subheader("Financial Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly Charges distribution
            fig_monthly = px.box(df, x='Contract', y='MonthlyCharges', 
                               color='Churn', title='Monthly Charges by Contract Type and Churn')
            st.plotly_chart(fig_monthly)

        with col2:
            # Total Charges vs Tenure
            fig_total = px.scatter(df, x='tenure', y='TotalCharges', 
                                 color='Churn', title='Total Charges vs Tenure',
                                 trendline="ols")
            st.plotly_chart(fig_total)

        # Payment method distribution
        payment_counts = df['PaymentMethod'].value_counts()
        fig_payment = px.pie(values=payment_counts.values, 
                           names=payment_counts.index,
                           title='Payment Method Distribution')
        st.plotly_chart(fig_payment, use_container_width=True)

def preprocess_data(df):
    # Drop customerID as it's not relevant for prediction
    df = df.drop('customerID', axis=1)
    
    # Create label encoders for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    return df, label_encoders

def train_model(df):
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def create_prediction_plots(input_df, prediction_proba, model):
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart for churn probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[0][1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightpink"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))
        st.plotly_chart(fig_gauge)

    with col2:
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                              title='Feature Importance',
                              orientation='h')
        st.plotly_chart(fig_importance)

def main():
    st.title("Customer Churn Analysis Dashboard")
    
    # Load and preprocess data
    df = load_data()
    
    # Add tabs for different sections
    tab1, tab2 = st.tabs(["Data Analysis", "Churn Prediction"])
    
    with tab1:
        st.header("Exploratory Data Analysis")
        create_eda_plots(df)
    
    with tab2:
        st.header("Churn Prediction")
        # Process data and train model
        df_processed, label_encoders = preprocess_data(df)
        model, scaler = train_model(df_processed)
        
        # Create sidebar for user input
        st.sidebar.header("Customer Information")
        
        # Create input fields for each feature
        input_data = {}
        
        input_data['gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        input_data['SeniorCitizen'] = st.sidebar.selectbox('Senior Citizen', [0, 1])
        input_data['Partner'] = st.sidebar.selectbox('Partner', ['Yes', 'No'])
        input_data['Dependents'] = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
        input_data['tenure'] = st.sidebar.slider('Tenure (months)', 0, 72, 1)
        input_data['PhoneService'] = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
        input_data['MultipleLines'] = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
        input_data['InternetService'] = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        input_data['OnlineSecurity'] = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
        input_data['OnlineBackup'] = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
        input_data['DeviceProtection'] = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
        input_data['TechSupport'] = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
        input_data['StreamingTV'] = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
        input_data['StreamingMovies'] = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
        input_data['Contract'] = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        input_data['PaperlessBilling'] = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
        input_data['PaymentMethod'] = st.sidebar.selectbox('Payment Method', 
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        input_data['MonthlyCharges'] = st.sidebar.slider('Monthly Charges', 0, 150, 50)
        input_data['TotalCharges'] = st.sidebar.slider('Total Charges', 0, 8000, 1000)

        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Transform categorical variables
        for column in input_df.select_dtypes(include=['object']).columns:
            input_df[column] = label_encoders[column].transform(input_df[column])
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        if st.sidebar.button('Predict Churn'):
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            st.subheader('Prediction Results')
            
            # Display prediction
            if prediction[0] == 1:
                st.error('⚠️ Customer is likely to churn')
            else:
                st.success('✅ Customer is likely to stay')
            
            # Create prediction visualization plots
            create_prediction_plots(input_df, prediction_proba, model)

if __name__ == '__main__':
    main()