import streamlit as st 
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
from pandasai import PandasAI

#calling constructior
#calling constructior
load_dotenv()
#Assessing openapi key from .env file
open_api_key = os.getenv('OPEN_API_KEY')

def chatWithCSV(df, prompt):
    llm = OpenAI(open_api_key)
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=prompt)
    print(result)
    return result



st.title("ChatCSV Powered by LLM | PandasAI")

llm_selected = st.selectbox('Choose the LLM', ['OpenAI', 'Google GenAI', 'AWS GenAI'])


input_csv = st.file_uploader("Upload your csv", type=['csv'])

if input_csv is not None and llm_selected == "OpenAI":
    st.info("CSV file uploaded successfully!!")
    data = pd.read_csv(input_csv)
    st.dataframe(data)

    st.info("Chat with your csv")
    input_text = st.text_area("Enter your query:")
    if input_text is not None:
        if st.button("Chat with CSV"):
            st.info("Your query: "+ input_text)
            result = chatWithCSV(data, input_text)
            final_result = st.dataframe(result)




