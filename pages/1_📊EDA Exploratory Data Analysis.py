import streamlit as st, numpy as np, pandas as pd, os
from streamlit_extras.stateful_button import button
from utils import st_def

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Data Preprocessing👋',  page_icon="🚀",)
st.title('🔍 Data Pre-Processing')
st_def.st_logo()
#-----------------------------------------------
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import missingno as msno
data_path = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)
df = df.drop('customerID', axis=1)
st.code(df)
if button("check missing value", key="but1"):
    st.code(df.isna().sum())
    st.write("There is currently no null value in the data set. However, we need to examine this output in a little more detail so that it is not misleading.")
    st.write(df.info())
    st.write("We can see the data consisting of 20 columns with 7043 instances. According to the first observations, there are no empty values in this data set. But we should note that the variable TotalCharges is of object type. We need to convert this variable to numeric type.")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    if button("# Missing values check - 2nd round", key="but2"):
        def missing_values(df):
            # Total missing values
            mis_val = df.isnull().sum()
            
            # Percentage
            mis_val_percent = 100 * df.isnull().sum() / len(df)
            
            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
            mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})
            
            # Sort the table by percentage of missing descending
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
            
            # Print some summary information
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                " columns that have missing values.")
            
            # Return the dataframe with missing information
            return mis_val_table_ren_columns
        
        missing_values_table = missing_values(df)
        st.code(missing_values_table)
        msno.matrix(df)
        fig = plt.gcf()
        st.pyplot(fig)
        # Fill the empty values in the TotalCharges variable by multiplying the tenure and MonthlyCharges values

        # df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
        df.fillna({'TotalCharges': df['tenure'] * df['MonthlyCharges']}, inplace=True)  # Option 1: Use the DataFrame-level method
        # df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges']) # Option 2: Assign the result back to the original column
        st.text(df[['TotalCharges', 'SeniorCitizen']])
        st.code('# Check cardinality control')

        st.code(df.nunique())
        st.code('There is no cardinality problem in categorical variables.')
        st.subheader('Let us review categorical and numerical values one last time')

        def filter_categorical_numeric_columns(dataframe):
            categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
            numeric_columns = dataframe.select_dtypes(include=['number']).columns
            return categorical_columns, numeric_columns

        # Filter categorical and numeric variables
        categorical_cols, numeric_cols = filter_categorical_numeric_columns(df)

        st.code("Categorics:")
        st.code(categorical_cols)
        st.code("\nNumerics:")
        st.code(numeric_cols)

        if button("Let's see the unique values of the categorical features.", key="but3"):
            for feature in df[categorical_cols]:
                st.code(f'{feature}: {df[feature].unique()}')
            st.write("In MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies variables 'No' and 'No internet service' are used repeatedly although they mean the same thing. These need to be merged during the model development phase.")
            
            df['MultipleLines'] = df['MultipleLines'].replace('No phone service','No')
            columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for column in columns_to_replace:
                df[column] = df[column].replace('No internet service', 'No')        
            
            if button("Visualization", key="but4"):
                import matplotlib.pyplot as plt
                import seaborn as sns 
                import plotly.express as px
                import warnings
                warnings.filterwarnings('ignore')

                churn_counts= df['Churn'].value_counts()
                fig2 = px.pie(names= churn_counts.keys(), values= churn_counts.values, title='Churn Distribution')    #fig2.show()
                # fig  = px.pie(names=churn_counts.index, values=churn_counts.values, title='Churn Distribution')
                st.plotly_chart(fig2)
                st.divider()
                
                # This function gives the distribution of variables, their relationship with the target variable and the probability of churn on a variable basis.'''
                if button("distribution of variables", key="but5"):
                    def analyze_category(df, category_column, target_column='Churn'):
                        # Value Counts Pie Chart
                        category_counts = df[category_column].value_counts()
                        fig = px.pie(names=category_counts.index, values=category_counts.values, title=f'{category_column} Distribution')
                        fig.show()

                        # Churn Probabilities
                        churn_probabilities = df.groupby(category_column)[target_column].value_counts(normalize=True) * 100
                        for category_value in df[category_column].unique():
                            churn_rate = churn_probabilities[category_value]['Yes'] if 'Yes' in churn_probabilities[category_value].index else 0
                            st.write(f"A {category_value} customer has a probability of {churn_rate:.2f}% churn")

                        # Histogram
                        fig = px.histogram(df, x=category_column, color=target_column, width=400, height=400)
                        # fig.show()
                        st.plotly_chart(fig)

                        # Grouping
                        grouped_data = df.groupby([category_column, target_column]).size().reset_index(name='count')

                        # Bar Chart
                        plt.figure(figsize=(10, 6))
                        sns.barplot(data=grouped_data, x=category_column, y='count', hue=target_column)
                        plt.title(f'Number of people with or without churn by {category_column} type')
                        plt.xlabel(category_column)
                        plt.ylabel('Count')
                        # plt.show() 
                        st.pyplot(plt)
                    
                    st.code(df["gender"].value_counts())
                    st.write(analyze_category(df, 'gender'))
                    
                    st.code(df["SeniorCitizen"].value_counts())
                    st.write(analyze_category(df, 'SeniorCitizen'))
                    
                    st.write(df["Contract"].value_counts())
                    st.write(analyze_category(df, 'Contract'))
                    
                    # st.write(df[['tenure', 'MonthlyCharges', 'TotalCharges']].iplot(kind='histogram',subplots=True,bins=50))
                    fig = px.histogram(df, x=['tenure', 'MonthlyCharges', 'TotalCharges'], marginal='box', nbins=50)
                    st.plotly_chart(fig)
                    
    # if button("", key="but6"):pass
    # if button("", key="but7"):pass
    # if button("", key="but8"):pass
