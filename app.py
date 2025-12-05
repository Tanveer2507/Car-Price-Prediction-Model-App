# Streamlit app: Car Price Prediction (auto-generated)
# Path of model pickle used: car_train_model.pkl
# This app attempts to load the provided model and dynamically build input widgets
# based on detected feature names. If auto-detection fails, it falls back to a manual JSON input.

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
import sys
import sklearn
# --- How to Run This App ---
st.sidebar.header("How to Run This App")
st.sidebar.markdown(
    """
    1.  **Save** the code above as a Python file (e.g., `app.py`).
    2.  **Open** your terminal or command prompt.
    3.  **Activate it:** `venv` folder(e.g., `.\.venv\Scripts\activate`)
    4.  **Run** the command: `streamlit run app.py`
    """
)

# --- Sidebar Inputs ---
st.sidebar.header("Car Details Input")

# Year selection
year = st.sidebar.selectbox(
    "Select Year",
    options=[1992, 1997, 1995, 2001, 1999, 2000, 2020, 
             2002, 2003, 1998, 2004, 2008, 2010, 2005, 
             1996, 2006, 2019, 2009, 2011, 2018, 2013, 
             2015, 2016, 2014, 2017, 2012, 2007]
)

# Km Driven (fixed value for now)
km_driven = st.sidebar.number_input("Km Driven", min_value=0, value=1)

# Fuel selection (numeric values)
fuel_options = {'Petrol':1, 'Diesel':2, 'CNG':3, 'LPG':4, 'Electric':5}
fuel = st.sidebar.selectbox("Fuel Type", options=list(fuel_options.keys()))
fuel_value = fuel_options[fuel]

# Seller Type (numeric values)
seller_options = {'Individual':1, 'Dealer':2, 'Trustmark Dealer':3}
seller_type = st.sidebar.selectbox("Seller Type", options=list(seller_options.keys()))
seller_type_value = seller_options[seller_type]

# Transmission (numeric values)
transmission_options = {'Manual':0, 'Automatic':1}
transmission = st.sidebar.selectbox("Transmission", options=list(transmission_options.keys()))
transmission_value = transmission_options[transmission]

# Owner (numeric values)
owner_options = {'First Owner':1, 'Second Owner':2, 'Fourth & Above Owner':3,
                 'Third Owner':4, 'Test Drive Car':5}
owner = st.sidebar.selectbox("Owner Type", options=list(owner_options.keys()))
owner_value = owner_options[owner]


MODEL_PATH = "./directory files/car_train_model.pkl"

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("Car Price Prediction Model")

def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_artifacts():
    model = load_model(MODEL_PATH)
    # try to detect feature names
    feature_names = None
    if isinstance(model, dict):
        for k in ["columns","feature_names","features","input_columns","cols","columns_"]:
            if k in model:
                feature_names = model[k]
                break
        # check nested entries for a model object
        for v in model.values():
            try:
                if hasattr(v, "feature_names_in_"):
                    feature_names = list(getattr(v, "feature_names_in_"))
                    break
            except Exception:
                pass
    else:
        # sklearn estimator or pipeline
        if hasattr(model, "feature_names_in_"):
            feature_names = list(getattr(model, "feature_names_in_"))
        elif hasattr(model, "named_steps"):
            # search for a transformer with feature names
            for step in getattr(model, "named_steps").values():
                if hasattr(step, "feature_names_in_"):
                    feature_names = list(getattr(step, "feature_names_in_"))
                    break
    return model, feature_names

model, feature_names = load_artifacts()

st.subheader("Model information")
st.write("Loaded model type:", type(model).__name__)
if feature_names:
    st.write("Detected feature names (order matters):")
    st.write(feature_names)
else:
    st.info("Could not reliably detect expected input feature names. Use the manual JSON input below or adjust the model to include feature names.")

st.markdown("---")

# Display numeric values (for confirmation)
st.write("### Enter Numeric Values")
st.write("Year:", year)
st.write("Km Driven:", km_driven)
st.write("Fuel Type:", fuel_value)
st.write("Seller Type:", seller_type_value)
st.write("Transmission:", transmission_value)
st.write("Owner Type:", owner_value)

st.subheader("Enter Car Details for prediction")

input_data = None
if feature_names:
    st.markdown("Enter values for each feature detected by the model:")
    values = {}
    with st.form("input_form"):
        for feat in feature_names:
            # Heuristic: if feature name suggests numeric, use number_input; otherwise text_input
            lname = feat.lower()
            if any(x in lname for x in ["year","age","km","kmdriven","mileage","engine","power","seats","price","owner"] ) or "float" in lname or "int" in lname:
                # use text_input but validate numeric on submit to avoid widget errors for unknown ranges
                values[feat] = st.text_input(label=feat, value="0")
            else:
                # categorical / textual
                values[feat] = st.text_input(label=feat, value=0)
        submitted = st.form_submit_button("Predict price")
        if submitted:
            # create DataFrame row from values
            row = {}
            for k,v in values.items():
                # try to convert numeric strings to float/int where appropriate
                try:
                    # treat purely numeric strings as floats
                    if isinstance(v, str) and v.replace('.', '', 1).lstrip('-').isdigit():
                        row[k] = float(v) if '.' in v else int(v)
                    else:
                        row[k] = v
                except Exception:
                    row[k] = v
            input_data = pd.DataFrame([row], columns=feature_names)
else:
    st.markdown('Manual input (JSON). Example: `{"year": 2015, "km_driven": 50000, "fuel": "Petrol"}`')
    txt = st.text_area("Paste a JSON object with feature names and values", value='{}', height=150)
    if st.button("Predict from JSON"):
        try:
            j = json.loads(txt)
            input_data = pd.DataFrame([j])
        except Exception as e:
            st.error("Invalid JSON: " + str(e))

if input_data is not None:
    st.write("Input preview:")
    st.dataframe(input_data)
    try:
        # Some pickles are dicts with a 'model' key
        estimator = model
        if isinstance(model, dict) and "model" in model:
            estimator = model["model"]
        # If the saved object is a tuple like (model, columns...), try to extract first element as estimator
        if isinstance(model, (list, tuple)) and len(model) > 0:
            candidate = model[0]
            if hasattr(candidate, "predict"):
                estimator = candidate
        # Try prediction
        preds = estimator.predict(input_data)
        # if prediction returns array, take first element
        if hasattr(preds, "__len__"):
            pred = preds[0]
        else:
            pred = preds
        st.success(f"Predicted price: ₹{round(pred)}")
    except Exception as e:
        st.error("Prediction failed: " + str(e))

st.markdown('---')
st.caption("This Streamlit app was auto-generated. You can edit /mnt/data/streamlit_app.py to customize the UI.")



