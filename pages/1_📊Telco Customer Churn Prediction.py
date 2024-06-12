import streamlit as st
from utils import st_def, t_churn

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Catboost Model👋',  page_icon="🚀",)
st.title('🔍 Data Processing')
st_def.st_logo()
tab1, tab2, tab3 = st.tabs(["1. Data Source", "2. Stratified Splitting", "3. Catboost"])
#-----------------------------------------------
with tab1: t_churn._datasource()
with tab2: t_churn._stratified_splitting()
with tab3: t_churn._catboost()