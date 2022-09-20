# Functions.py 
###############################################################################################
# Aula de deploy de modelos de machine learning usando streamlit #
# Created: 14/09/2022
# Author: Prof. Leandro Romualdo 
# Contact: leandroromualdo@uni9.prof.br
###############################################################################################

import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn import metrics
import shap 
import matplotlib.pyplot as plt 
import seaborn as sns 

## Criação do Pipeline de predição do modelo
def pipeline_predict(df, obj):

    if obj == 'Predição de Churn':

        df = df
        # Transformação das variáveis
        df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
        df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes

        df['international_plan'] = df['international_plan'].astype('category')
        df['international_plan'] = df['international_plan'].cat.codes

        df['area_code'] = df['area_code'].astype('category')
        df['area_code'] = df['area_code'].cat.codes

        df['state'] = df['state'].astype('category')
        df['state'] = df['state'].cat.codes

        # Carregar o arquivo do modelo preditivo para model_trained
        with open('xgb_model', 'rb') as files:
            model_trained = pkl.load(files)
        
        # Filtra as features que estão no modelo, ignora 
        features = model_trained.get_booster().feature_names
        df = df[features]
        
        return model_trained.predict(df), model_trained.predict_proba(df)[:,1]


    if obj == 'Explicabilidade' and df == '':

        df = pd.read_csv('churn_train.csv')
        df['churn'] = df['churn'].astype('category')
        df['churn'] = df['churn'].cat.codes
        df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
        df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes
        df['international_plan'] = df['international_plan'].astype('category')
        df['international_plan'] = df['international_plan'].cat.codes
        df['area_code'] = df['area_code'].astype('category')
        df['area_code'] = df['area_code'].cat.codes
        df['state'] = df['state'].astype('category')
        df['state'] = df['state'].cat.codes

        X = df.drop('churn', axis=1)
        y = df['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7565)

        with open('xgb_model', 'rb') as files:
            model_trained = pkl.load(files)

        # Pega as features usadas no modelo
        #features = model_trained.get_boosted().feature_names
        #df = df[features] # filtra as features usadas no modelo nos dados de entrada

        prob = model_trained.predict_proba(X_test)[:,1]
        pred = model_trained.predict(X_test)

        # Print das métricas
        st.title('Principais Métricas.')
        st.write("** AUC: **"+str(metrics.roc_auc_score(y_test, prob)))
        st.write("** Accuracy:** "+str(metrics.accuracy_score(y_test, pred)))
        st.write("** Recall:** "+str(metrics.recall_score(y_test, pred)))
        st.write("** F1-Measure:** "+str(metrics.f1_score(y_test, pred)))

        # Grafico de features mais importantes, sempre usar st.pyplot nos graficos
        st.title('Features mais importantes')
        feature_important = model_trained.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
        data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10))
        st.pyplot()

        st.title('Explicabilidade usando Shap values dos primeiros três registros de teste')
        explainer = shap.Explainer(model_trained)
        shap_values = explainer(X_test)

        shap.plots.waterfall(shap_values[0])
        st.pyplot()
        shap.plots.waterfall(shap_values[1])
        st.pyplot()
        shap.plots.waterfall(shap_values[2])
        st.pyplot()

        shap.summary_plot(shap_values)
        st.pyplot()

        shap.plots.force(shap_values[3])
        st.pyplot()
        shap.plots.force(shap_values[4])
        st.pyplot()
        shap.plots.force(shap_values[5])
        st.pyplot()

    if obj == 'Análise exploratória':

        df = pd.read_csv('churn_train.csv')

        df.international_plan.value_counts().plot(kind = 'pie')
        df.voice_mail_plan.value_counts().plot(kind = 'pie')
        churn_y = df.loc[df.churn == 'yes', 'churn'].count()
        churn_n = df.loc[df.churn == 'no', 'churn'].count()
        churn_total =churn_n+churn_y

        st.markdown('** Exploração do dataset, visão de churn **')
        st.write('Qtde total de clientes na base : '+ str(churn_total))
        st.write('Qtde de clientes com churn '+ str(churn_y) +' que representam '+ str(round(100*churn_y/churn_total,0)) +'% da base de clientes.' )
        st.write('Qtde de clientes sem churn  '+ str(churn_n) +' que representam '+ str(round(100*churn_n/churn_total,0)) +'% da base de clientes' )

        st.markdown('** Análise Exploratória **')
        st.title('Quantidade de clientes com churn por estado')
        fig, ax=plt.subplots(figsize=(20,5))
        sns.countplot(data = df, x='state', order=df['state'].value_counts().index, palette='viridis', hue='churn')
        plt.xticks(rotation=90)
        plt.xlabel('State', fontsize=10, fontweight='bold')
        plt.ylabel('Customers', fontsize=10, fontweight='bold')
        plt.title('State wise Customers', fontsize=12, fontweight='bold')
        st.pyplot()

        st.title('** Quantidade de clientes com e sem churn por código de área **')
        sns.countplot(data = df, x='area_code', order=df['area_code'].value_counts().index, palette='viridis', hue='churn')
        plt.xlabel('Area Code', fontsize=10, fontweight='bold')
        plt.ylabel('Customers', fontsize=10, fontweight='bold')
        plt.title('Area Code wise Customers', fontsize=12, fontweight='bold')
        st.pyplot()

        st.title('Correlação entre variáveis')
        corr = df.corr()
        fig4, ax = plt.subplots(figsize=(15,7))
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True,cmap="YlGnBu",annot_kws={'size': 12},fmt=".2f")
        st.pyplot()