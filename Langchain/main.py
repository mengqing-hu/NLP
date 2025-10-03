import streamlit as st
import langchain_helper

st.title("Resturan Name and Menu Generator")
cuisine= st.sidebar.selectbox("Select Cuisine", ["Chinese", "Indian", "Mexican", "Italian", "French"])


if cuisine:
    response = langchain_helper.generate_name_and_menu(cuisine)
    st.header(response["restaurant_name"].strip())
    menu_items = response["menu_items"].strip().split(",")
    st.write("##### Menu")
    for item in menu_items:
        st.write("- " + item)
