import streamlit as st

st.set_page_config(
    page_title="Balloon Detection App",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image("./images/banner.png", use_column_width="always")
st.divider()
st.title("Balloon Detection App")

st.markdown(
    """"
    This is a streamlit application for Balloon Detection

    """
)

st.divider()