# Customer Churn Analysis Dashboard

This project is a web-based dashboard for analyzing customer churn in a telecommunications company. It allows users to explore customer demographics, service usage, financial metrics, and predict the likelihood of customer churn based on input data.

## Features

- **Exploratory Data Analysis (EDA):**
  - Customer demographics visualization (gender, senior citizen status).
  - Service usage analysis (Internet service types, contract types).
  - Financial analysis (monthly charges, total charges, payment methods).

- **Churn Prediction:**
  - User-friendly sidebar for inputting customer information.
  - Random Forest model trained on historical churn data.
  - Displays churn probability and feature importance.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Seaborn
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-analysis.git
   cd customer-churn-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains information about customer behavior and demographics. Make sure to place the `telco-customer-churn.csv` file in the project directory.

## Usage

- Navigate to the **Data Analysis** tab to explore visualizations.
- Use the **Churn Prediction** tab to input customer information and predict churn likelihood.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the dataset.
- Streamlit documentation for providing an easy way to create web applications with Python.
