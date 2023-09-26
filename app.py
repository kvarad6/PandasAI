import streamlit as st 
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
from pandasai import PandasAI
from pandasai.llm.google_palm import GoogleVertexai


#calling constructior
load_dotenv()


llm_selected = st.selectbox('Choose the LLM', ['OpenAI', 'Google GenAI'])

def chatWithCSV(df, prompt, open_api_key):
    # llm = OpenAI(open_api_key)
    if llm_selected == "Google GenAI":
        Googlellm = GoogleVertexai(project_id="",
                                location="us-central1",
                                model="text-bison@001")
        pandas_ai = PandasAI(Googlellm, enable_cache=False) 
        result = pandas_ai.run(df, prompt=prompt)
        print(result)
    elif llm_selected == "OpenAI":
        #Assessing openapi key from .env file
        # open_api_key = os.getenv('OPEN_API_KEY')
        llm = OpenAI(open_api_key)
        pandas_ai = PandasAI(llm)
        result = pandas_ai.run(df, prompt=prompt)
        print(result)
        
    return result

st.title("ChatCSV Powered by LLM | PandasAI")

input_csv = st.file_uploader("Upload your csv", type=['csv'])

if input_csv is not None:
    st.info("CSV file uploaded successfully!!")
    data = pd.read_csv(input_csv)
    st.dataframe(data)

    st.info("Chat with your csv")
    open_api_key = st.text_area("Enter OpenAI Key:")
    input_text = st.text_area("Enter your query:")
    if input_text is not None:
        if st.button("Chat with CSV"):
            st.info("Your query: "+ input_text)
            result = chatWithCSV(data, input_text, open_api_key)
            final_result = st.dataframe(result)




