import streamlit as st
from streamlit_extras.stateful_button import button
# -----------------------------------------------
import pandas as pd, os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score,recall_score, roc_auc_score, precision_score)

from catboost import CatBoostClassifier

def _datasource():
    st.write("""
    This prototype consists of 7043 customers with 21 columns (features). 
    It contains customer account information, demographic information, and registered services. 
    The target variable (Churn) provides information on whether the customer has churned.""")

    if button("1. Click to show orignal data source", key="b1"):
        data_path = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = pd.read_csv(data_path)
        st.code(df)

        if button("2. Convert TotalCharges to numeric, filling NaN values", key="b1"):
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df.fillna({'TotalCharges': df['tenure'] * df['MonthlyCharges']}, inplace=True)  # Option 1: Use the DataFrame-level method
            st.write(df[['TotalCharges', 'SeniorCitizen']])

            if button("3. Convert SeniorCitizen to object", key="b1"):
                df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
                st.write(df[['TotalCharges', 'SeniorCitizen', 'MultipleLines']])

                if button("4. Replace 'No phone service' and 'No internet service' with 'No' for certain columns", key = 'b1'):
                    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
                    columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
                    for column in columns_to_replace:
                        df[column] = df[column].replace('No internet service', 'No')
                    st.write(df[['TotalCharges', 'SeniorCitizen', 'MultipleLines', 'Churn']])

                    if button("5. Convert 'Churn' categorical variable to numeric", key = 'b1'):
                        df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
                        st.write(df[['TotalCharges', 'SeniorCitizen', 'Churn']])
                        if 'df' not in st.session_state:
                            st.session_state['df'] = df


def _stratified_splitting():
    st.write("The unbalanced nature of the dataset was observed in the EDA study. \
        To ensure an unbiased distribution of classes in our train-test split, \
            we employ `StratifiedShuffleSplit` from scikit-learn. \
                This method preserves the proportion of classes in both training and testing sets, critical for reliable model evaluation.")

    st.subheader("1. Create the StratifiedShuffleSplit object")
    df = st.session_state.df
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
    train_index, test_index = next(strat_split.split(df, df["Churn"]))

    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

    X_train = strat_train_set.drop("Churn", axis=1)
    y_train = strat_train_set["Churn"].copy()

    X_test = strat_test_set.drop("Churn", axis=1)
    y_test = strat_test_set["Churn"].copy()

    if 'train_test' not in st.session_state:
        st.session_state['train_test'] = {'X_train': X_train,            'y_train': y_train,            'X_test': X_test,            'y_test': y_test}        
    st.text(X_train)
    st.text(X_test)
    st.code(y_train)
    st.code(y_test)

def _catboost():
    ################################################## CATBOOST ##################################################
    df = st.session_state.df
    X_train, y_train, X_test, y_test = (st.session_state['train_test'][key] for key in ['X_train', 'y_train', 'X_test', 'y_test'])
    
    st.code('# # Identify categorical columns')

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Initialize and fit CatBoostClassifier
    cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
    cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))

    # Predict on test set
    y_pred = cat_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

    # Create a DataFrame to store results
    model_names = ['CatBoost_Model']
    result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=model_names)

    # Print results
    st.write(result)

    # Save the model in the 'model' directory
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "leo_catboost_model.cbm")
    cat_model.save_model(model_path)