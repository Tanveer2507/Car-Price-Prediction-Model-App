██████╗  █████╗ ██████╗      ██████╗ ██████╗ ██╗███████╗
██╔══██╗██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██║██╔════╝
██████╔╝███████║██████╔╝    ██║     ██║   ██║██║█████╗  
██╔══██╗██╔══██║██╔══██╗    ██║     ██║   ██║██║██╔══╝  
██████╔╝██║  ██║██║  ██║    ╚██████╗╚██████╔╝██║███████╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═╝╚══════╝

          🚗 Car Price Prediction Web App 🚗
     Machine Learning | Streamlit | Python | Scikit-Learn

# 🚗 Car Price Prediction Model | Streamlit Web App

A machine learning–powered Car Price Prediction Web Application built using Python, Scikit-Learn, Pandas, NumPy, and Streamlit.
This project predicts the price of a used car based on parameters like year, fuel type, transmission, km driven, seller type, and ownership.

# 📌 Project Features

✔ Machine Learning model trained on car price dataset

✔ Streamlit-based interactive web application

✔ User-friendly sidebar for input

✔ Auto-detection of model input features

✔ JSON fallback input when feature detection fails

✔ Real-time car price prediction using .pkl model

✔ Displays prediction output instantly

# 📂 Project Structure
Car_Price_Prediction/
│── app.py                     # Streamlit front-end code
│── car_train_model.pkl        # Trained ML model
│── README.md                  # Project documentation
│── dataset.csv / notebook.ipynb (optional)

# 🧠 Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Streamlit

Pickle

# ⚙️ How to Run This Project Locally
# 1️⃣ Install required libraries
pip install streamlit pandas numpy scikit-learn

# 2️⃣ Run the Streamlit App
streamlit run app.py

# 3️⃣ Open in Browser
Streamlit will automatically open:
👉 [http://localhost:8501](https://carprice-predictionmodel-app.streamlit.app/)

# 🧩 How the Model Works

Loads the saved ML model from car_train_model.pkl

Automatically detects required feature names

Takes user inputs (year, fuel, transmission, etc.)

Creates a DataFrame

Uses the ML model to predict the price

Displays output on the UI
