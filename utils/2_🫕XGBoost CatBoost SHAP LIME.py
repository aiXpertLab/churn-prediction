import streamlit as st, numpy as np, pandas as pd, os, joblib, sys
from streamlit_extras.stateful_button import button
from utils import st_def

st.set_page_config(page_title='👋 AI',  page_icon="🚀",)
st.title('🔍 AI')
st_def.st_logo()
#------------------------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,classification_report, recall_score,confusion_matrix, roc_auc_score, precision_score, f1_score, roc_curve, auc
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier, plot_importance

import shap, lime
import lime.lime_tabular
#------------------------------------------------------------------------------------------------
if button("Data Organization", key="button1"):
    data_path = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    categorical_columns = df.select_dtypes(include=['object', 'category'])
    for feature in categorical_columns:
        st.code(f"Column: {feature} -> {df[feature].unique()}")
        st.code(f"Unique Number: {df[feature].nunique()}")
        st.code(f"{df[feature].value_counts()} \n")
        st.code(df.isnull().sum())
    st.write("We need to drop the custoemerID column that each customer has unique, but I will leave it this way because I will use this value in my future applications. And make the TotalCharges variable numeric. We know from the EDA notebook that the TotalCharges variable has 11 empty values. Let's statistically fill these values using tenure and MonthlyCharges variables. In MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies variables 'No' and 'No internet-phone service' are used repeatedly although they mean the same thing. These need to be merged during the model development phase.")
#------------------------------------------------------------------------------------------------
    if button("TotalCharges and Churn", key="butto2"):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

        df['MultipleLines'] = df['MultipleLines'].replace('No phone service','No')
        columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for column in columns_to_replace:
            df[column] = df[column].replace('No internet service', 'No')
            
        # Changing categorical variables to numeric:
        df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
        
        st.write("""
                A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
                The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

                For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
                Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
            """)      
        
        if button("Save the parquet data", key="butto3"):
            df.to_parquet('./data/churn_data_regulated.parquet')
            
            if button("Preparation of Data for the CatBoost Model and Save dataset to .pkl", key="butto4"):
                # Create the StratifiedShuffleSplit object
                strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)

                train_index, test_index = next(strat_split.split(df, df["Churn"]))

                # Create train and test sets
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

                # Proportion of the target variable in the original data set
                st.code('Target Labels Ratio in Original Dataset\n')
                st.code(df["Churn"].value_counts(normalize=True).sort_index())

                # Proportion of the target variable in the test set
                st.code('\nTarget Labels Ratio in Test Dataset\n')
                st.code(strat_test_set["Churn"].value_counts(normalize=True).sort_index())

                X_train = strat_train_set.drop("Churn", axis=1)
                y_train = strat_train_set["Churn"].copy()

                X_test = strat_test_set.drop("Churn", axis=1)
                y_test = strat_test_set["Churn"].copy()
                
                # Save the datasets
                joblib.dump(X_train, './data/X_train.pkl')
                joblib.dump(y_train, './data/y_train.pkl')
                joblib.dump(X_test, './data/X_test.pkl')
                joblib.dump(y_test, './data/y_test.pkl')
                
                if button("CatBoost Model", key="butto5"):
                    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

                    cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)

                    cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))

                    y_pred = cat_model.predict(X_test)

                    accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

                    model_names = ['CatBoost_Model']

                    result = pd.DataFrame({'Accuracy':accuracy,
                                        'Recall':recall, 
                                        'Roc_Auc':roc_auc, 
                                        'Precision':precision}, index=model_names)

                    st.code(result)
                    st.info('cbm model saved.')
                    cat_model.save_model('./data/kaggle_cat_model.cbm')
                    pool = Pool(X_train, y_train, cat_features=categorical_columns)

                    feature_importance = pd.DataFrame({'feature_importance': cat_model.get_feature_importance(pool), 
                                        'feature_names': X_train.columns}).sort_values(by=['feature_importance'], ascending=False)

                    st.code(feature_importance)
                    
                    plt.figure(figsize=(10,10))
                    sns.barplot(x=feature_importance['feature_importance'], y=feature_importance['feature_names'], palette = 'rocket')
                    # plt.show()
                    st.pyplot(plt)
                    
                    cat_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
                    sns.heatmap(cat_confusion_matrix, annot=True, fmt="d")

                    plt.xlabel("Predicted Label", fontsize= 12)
                    plt.ylabel("True Label", fontsize= 12)
                    # plt.show()
                    st.pyplot(plt)
                    st.code(metrics.classification_report(y_test, y_pred, labels = [0, 1]))
                    
                    
                    st.divider()
                    if button("SHAP", key="butto6"):
                        explainercat = shap.TreeExplainer(cat_model)
                        shap_values_cat_train = explainercat.shap_values(X_train)
                        shap_values_cat_test = explainercat.shap_values(X_test)
                        # shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar",plot_size=(12,12))
                        st.pyplot(shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12, 12)))
                        
                        # summarize the effects of all the features

                        fig = plt.subplots(figsize=(6,6),dpi=200)
                        ax = shap.summary_plot(shap_values_cat_train, X_train,plot_type="dot")
                        st.pyplot(ax)
                        
                        # Contract

                        fig, ax= plt.subplots(figsize=(6,6),dpi=100)
                        shp_plt = shap.dependence_plot("Contract", shap_values_cat_test, X_test,ax=ax,interaction_index=None)
                        st.pyplot(shp_plt)
                        
                        # MonthlyCharges

                        fig, ax1= plt.subplots(figsize=(6,6),dpi=150)
                        shp_plt = shap.dependence_plot("MonthlyCharges", shap_values_cat_test,X_test,ax=ax1, interaction_index=None)
                        st.pyplot(shp_plt)
                        
                        st.divider()
                        if button("XGBoost", key="butto7"):
                            
                            data_path = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
                            df = pd.read_csv(data_path)
                            df.drop('customerID', axis=1, inplace=True)
                            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                            df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
                            df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

                            df['MultipleLines'] = df['MultipleLines'].replace('No phone service','No')
                            columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
                            for column in columns_to_replace:
                                df[column] = df[column].replace('No internet service', 'No')
                                
                            # Changing categorical variables to numeric:
                            df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})                            
                            st.markdown('''
                                A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
                                The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

                                For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.

                                Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`''')
                            
                            # OrdinalEncoder
                            encoder = OrdinalEncoder()
                            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

                            X = df.drop('Churn', axis=1).copy()
                            y = df['Churn'].copy()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Encode categorical columns in Train data
                            X_train_encoded = encoder.fit_transform(X_train[categorical_columns])

                            # Encode categorical columns in test data (no fit, only transform)
                            X_test_encoded = encoder.transform(X_test[categorical_columns])

                            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=categorical_columns)
                            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=categorical_columns)

                            xgb = XGBClassifier(random_state=0,scale_pos_weight=3)

                            xgb.fit(X_train_encoded_df, y_train)
                            y_pred = xgb.predict(X_test_encoded_df)

                            accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

                            model_names = ['XGBoost_adjusted_weight_3']
                            result_df3 = pd.DataFrame({'Accuracy':accuracy,'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_names)
                            st.code(result_df3)
                            
                            plot_importance(xgb)
                            # plt.show()
                            st.pyplot(plt)
                            
                            st.divider()
                            if button("LIME", key="butto8"):
                                
                                explainer = lime.lime_tabular.LimeTabularExplainer(X_train_encoded_df.values, feature_names=X_train_encoded_df.columns.values.tolist(),
                                                                                class_names=['Churn'], verbose=True, mode='classification')

                                # Choose the jth instance and use it to predict the results for that selection
                                j = 13
                                exp = explainer.explain_instance(X_train_encoded_df.values[j], xgb.predict_proba, num_features=5)
                                # exp.show_in_notebook(show_table=True)
                                for feat in exp.as_list():
                                    st.write(feat)
                                # fig = exp.as_pyplot_figure()
                                # st.pyplot(fig)