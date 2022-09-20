# app. py

###############################################################################################
# Aula de deploy de modelos de machine learning usando streamlit #
# Created: 14/09/2022
# Author: Prof. Leandro Romualdo 
# Contact: leandroromualdo@uni9.prof.br
###############################################################################################

# Import das bibliotecas
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import metrics
import shap
import matplotlib.pyplot as plt 
import seaborn as sns 
from functions import * 

st.set_option('deprecation.showPyplotGlobalUse', False)

image = Image.open('uninove.png') # Imagem do front com logo da Uninove

# Cabeçalho da tela
st.image(image)
html_temp = """
<div style ="background-color:blue;padding:13px">
<h1 style ="color:white;text-align:center;">Modelo de predição de Churn de clientes de telefonia móvel</h1>
</div>
"""

st.markdown(html_temp, unsafe_allow_html = True)
st.subheader('**Modelo de predição de churn usando Xtreme Gradient Boosting**')
st.markdown('**Este modelo foi treinado usando informações históricas de clientes que tiveram churn e clientes que não tiveram churn.**')

# Função principal
def main():

    st.subheader('** Selecione uma das opções abaixo: **')
    options =st.radio('O que voce deseja fazer? ', ('', 'Análise exploratória', 'Predição de Churn', 'Explicabilidade'))

    if options == 'Predição de Churn':
        st.subheader('Insira os dados abaixo:')
        state=st.selectbox('Escolha a sigla do estado :', ['','AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA','ID',\
		'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',\
		'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV','WY'])
        account_length=st.number_input('Selecione o tempo como cliente :', min_value=0, max_value=250, value=0)
        area_code=st.selectbox('Selecione o codigo de area :', ['','area_code_408', 'area_code_415', 'area_code_510'])
        international_plan=st.selectbox('Selecione se o cliente tem plano internacional :', ['', 'yes', 'no'])
        voice_mail_plan=st.selectbox('Selecione se o cliente tem plano de caixa postal :',  ['', 'yes', 'no'])
        number_vmail_messages=st.slider('Insira o numero de mensagens na caixa postal', min_value=0, max_value=250, value=0)
        total_day_minutes=st.slider('Insira o numero de minutos por dia :', min_value=0, max_value=250, value=0)
        total_day_calls=st.slider('Insira o numero de ligações por dia :', min_value=0, max_value=250, value=0)
        total_day_charge=st.slider('Insira o numero de recargas por dia :', min_value=0, max_value=250, value=0)
        total_eve_minutes=st.slider('Insira o numero de minutos por tarde :', min_value=0, max_value=250, value=0)
        total_eve_calls=st.slider('Insira o numero de ligações por tarde :', min_value=0, max_value=250, value=0)
        total_eve_charge=st.slider('Insira o numero de recargas por tarde :', min_value=0, max_value=250, value=0)
        total_night_minutes=st.slider('Insira o numero de minutos por noite :', min_value=0, max_value=250, value=0)
        total_night_calls=st.slider('Insira o numero de ligações por noite :', min_value=0, max_value=250, value=0)
        total_night_charge=st.slider('Insira o numero de recargas por noite :', min_value=0, max_value=250, value=0)
        total_intl_minutes=st.slider('Insira o numero de minutos internacionais :', min_value=0, max_value=250, value=0)
        total_intl_calls=st.slider('Insira o numero de ligações internacionais :', min_value=0, max_value=250, value=0)
        total_intl_charge=st.slider('Insira o numero de recargas internacionais :', min_value=0, max_value=250, value=0)
        number_customer_service_calls=st.slider('Insira o numero de ligações para atendimento ao cliente :', min_value=0, max_value=250, value=0)

        # Dicionário para gerar o dataset
        input_dict={'state':state,'account_length': account_length,'area_code':area_code,'international_plan':international_plan,'voice_mail_plan':voice_mail_plan\
		,'number_vmail_messages':number_vmail_messages,'total_day_minutes':total_day_minutes,'total_day_calls':total_day_calls\
        ,'total_day_charge':total_day_charge, 'total_eve_minutes':total_eve_minutes,'total_eve_calls':total_eve_calls\
        ,'total_eve_charge':total_eve_charge,'total_night_minutes':total_night_minutes,'total_night_calls':total_night_calls\
        ,'total_night_charge':total_night_charge,'total_intl_minutes':total_intl_minutes,'total_intl_calls':total_intl_calls\
		,'total_intl_charge':total_intl_charge ,'number_customer_service_calls':number_customer_service_calls}

        # Gerando dataset para predição
        df_test = pd.DataFrame([input_dict])

        if st.button('Predict'):
            predict_value, proba_value = pipeline_predict(df_test, 'Predição de Churn')
            if predict_value == '[1]':
                predict_value = 'Churn'
            else:
                predict_value = '** Não Churn **'
            proba_value = str(proba_value)
            proba_value.replace('[]', '')
            st.subheader('Dados inseridos pelo usuário:')
            st.write(df_test)
            st.write('A predição de churn deste cliente é {}, com a probabilidade de {}.'.format(predict_value, proba_value))
        
    if options == 'Análise exploratória':
        pipeline_predict('', 'Análise exploratória')

    if options == 'Explicabilidade':
        pipeline_predict('', 'Explicabilidade')
        
if __name__ == '__main__':
    main()