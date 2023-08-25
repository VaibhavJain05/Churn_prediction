import streamlit as st 
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


st.set_page_config(page_title="Churn Prediction", page_icon=":chair:", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():

    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAREDN;text-align:left;">Churn Prediction</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander("Information", expanded=True):
            st.write("""
            Churn Prediction
            """)



    with col2:
        st.subheader("Predicting churn based on user information")
        cust_id = st.number_input("Customer ID", 1,10000)
        name = st.text_input("Name")
        age = st.number_input("Age", 1,150)
        gender = st.text_input("Gender: Enter 'Male' or 'Female' ")
        if(gender=='Male'):
            gender = 1
        else:
            gender = 0
        locn = st.text_input("Location")
        if locn=='Chicago':
            locn = 0
        elif locn=='Houston':
            locn = 1
        elif locn=='Los Angeles':
            locn = 2
        elif locn=='Miamis':
            locn = 3
        else:
            locn = 4
        sub = st.number_input("Subscription Length (Month)",0,100)
        bill = st.number_input("Monthly Bill", 0.0,1000000.0)
        usage = st.number_input("Monthly Usage (GB)", 0,100000)

        feature_list = [age,gender,locn,sub,bill,usage]
        input = np.array(feature_list).reshape((1,len(feature_list)))
        model = xgb.XGBClassifier(n_estimators=500,random_state=42)
        model.load_model('model.json')
        scaler = load_model('scaler.pkl')
        
        if st.button('Predict'):
            input = scaler.transform(input)
            output = model.predict(input)
            if output[0] == 1:
                ans = "Churn"
            else:
                ans = "No Churn"

            col1.write('''
		    ## Results 
		    ''')
            col1.success(f"{ans}")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
