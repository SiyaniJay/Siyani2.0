# chatbot.py

import streamlit as st
from huggingface_hub import InferenceClient
import sys

def get_featherless_response(prompt: str) -> str:
    """
    Connects to the featherless-ai provider and gets a response.
    """
    try:
        # Initialize the client, getting the key from Streamlit secrets
        client = InferenceClient(
            provider="featherless-ai",
            api_key=st.secrets["HF_TOKEN"],
        )
        
        # Generate text and return the response
        response = client.text_generation(
            prompt,
            model="fdtn-ai/Foundation-Sec-8B",
            max_new_tokens=250,
        )
        return response
    except KeyError:
        # This error occurs if the HF_TOKEN is not in the secrets file
        return "ERROR: The 'HF_TOKEN' is not configured in your secrets.toml file."
    except Exception as e:
        return f"An error occurred with the AI service: {e}"