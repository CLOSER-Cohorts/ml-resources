import streamlit as st
import requests

st.title("Topic Classifier")

item_type = st.selectbox(
    "Item type",
    ["Question", "Variable"]
)

study = st.selectbox(
    "Study",
    ["uk.iser.ukhls",
    "uk.whitehall2",
    "uk.cls.nextsteps",
    "uk.lha",
    "uk.wchads",
    "uk.cls.bcs70",
    "uk.alspac",
    "uk.mrcleu-uos.sws",
    "uk.mrcleu-uos.hcs",
    "uk.mrcleu-uos.heaf"]
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

    print(payload)

    try:
        response = requests.post(
            f"http://127.0.0.1:8000/categorise_items/{study}",
            json=payload
        )

        response.raise_for_status()

        st.subheader("Predicted topic is:")

        st.json(response.json())

    except Exception as e:
        st.error(f"Request failed: {e}")