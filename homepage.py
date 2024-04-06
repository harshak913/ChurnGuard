import numpy as np
import streamlit as st
from data_visualizations import fig
import streamlit as st
import pandas as pd

st.markdown("""
    <style>
        .big-font {
            font-size:50px !important;
            background: -webkit-linear-gradient(45deg, orange, yellow);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .small-font {
            font-size:30px !important;
            background: -webkit-linear-gradient(45deg, purple, blue);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Churn Guard</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Locking in Loyalty</p>', unsafe_allow_html=True)


# Create DataFrame
df = pd.DataFrame({
    'Customer': ['Name', 'Age', 'DOB', 'Credit Score', 'Deposit Amount', 'Loan Amount', 'Mortgage'],
    'Info': ["John Doe", 35, "1986-07-15", 750, 5000.00, 25000.00, 200000.00]
})

# df.set_index('Customer', inplace=True)

st.dataframe(df) 

# # Dummy text
# st.title("Title in Purple")
# st.header("Header in Blue")


# Plot!
st.plotly_chart(fig, use_container_width=True)
