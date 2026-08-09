import streamlit as st
import requests

st.title("Topic Classifier")

item_type = st.selectbox(
    "Item type",
    ["Question", "Variable"]
)


label_text = st.text_area(
    "Enter a label text:",
    height=50
)

categories_text = st.text_area(
    "Enter any category reponses associated with the item, separated by spaces (e.g. 'yes no'):",
    height=50
)

if st.button("Submit"):

    payload = {
        "items": [
            {
                "TextLabel": label_text,
                "ItemType": item_type,
                "ItemCategories": categories_text
            }
        ]
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/categorise_items",
            json=payload
        )

        response.raise_for_status()

        st.subheader("Predicted topic is:")

        st.json(response.json())

    except Exception as e:
        st.error(f"Request failed: {e}")