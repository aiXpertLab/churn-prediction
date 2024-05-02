import streamlit as st, numpy as np, pandas as pd, os
from streamlit_extras.stateful_button import button
from utils import st_def

st.set_page_config(page_title='Data Preprocessing👋',  page_icon="🚀",)
st.title('🔍 Data Pre-Processing')
st_def.st_logo()
# st.markdown("🚀) 🍨📄Rule Extraction📚: Python Libraries  Approaches📰🍨 ")
#-----------------------------------------------
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, Pool

if button("Data Loading and Editing", key="button1"):
    ################################################## Data Loading and Editing ##################################################
    data_path = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    st.text(df)
    # # Convert TotalCharges to numeric, filling NaN values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
    df.fillna({'TotalCharges': df['tenure'] * df['MonthlyCharges']}, inplace=True)  # Option 1: Use the DataFrame-level method
    # df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges']) # Option 2: Assign the result back to the original column
    st.text(df[['TotalCharges', 'SeniorCitizen']])

    st.divider()
    if button("Convert SeniorCitizen to object", key="button2"):
        # # Convert SeniorCitizen to object
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

        # # Replace 'No phone service' and 'No internet service' with 'No' for certain columns
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
        columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for column in columns_to_replace:
            df[column] = df[column].replace('No internet service', 'No')

        # # Convert 'Churn' categorical variable to numeric
        df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

        st.text(df[['TotalCharges', 'SeniorCitizen', 'Churn']])
        st.divider()

        if button("StratifiedShuffleSplit", key="button3"):
       
            ################################################## StratifiedShuffleSplit ##################################################
            # Create the StratifiedShuffleSplit object
            strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)

            train_index, test_index = next(strat_split.split(df, df["Churn"]))

            # Create train and test sets
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]

            X_train = strat_train_set.drop("Churn", axis=1)
            y_train = strat_train_set["Churn"].copy()

            X_test = strat_test_set.drop("Churn", axis=1)
            y_test = strat_test_set["Churn"].copy()

            if button("Catboost", key="button4"):
                ################################################## CATBOOST ##################################################
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