
import streamlit as st
from src.pipeline.prediction_pipeline import CustomData
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logging

st.header("Cement Strength Predictor")

cement = st.number_input('Cement')
slag = st.number_input('Slag')
ash = st.number_input('Ash')
water = st.number_input('water')
Superplasticizer = st.number_input('Superplasticizer')
coarse_agg = st.number_input('coarse aggrigate')
fine_agg = st.number_input('fine aggrigate')
age = st.number_input('Age')

def perform_prediction():


    obj_customdata = CustomData(cement,slag,ash,water,Superplasticizer,coarse_agg,fine_agg,age)
    features = obj_customdata.get_custom_data()
    logging.info(features)

    obj_predict = PredictionPipeline()
    prediction = obj_predict.predict(features)
    return prediction

submit_button = st.button('Submit')
if submit_button:
    st.write(f'Stength:{perform_prediction()}')

