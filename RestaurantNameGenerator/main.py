import streamlit as st
from cuisine_helper import get_menu

st.title("Menu List Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ("Indian", "Mexican", "American"))

if cuisine:
    response = get_menu(cuisine)
    st.header(response["restaurant_name"].strip())
    menu_items = response["menu_items"].split(",")
    st.write("**Menu Items**")
    for item in menu_items:
        st.write("-", item)


