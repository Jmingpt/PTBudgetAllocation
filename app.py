import streamlit as st
from budget import *


def run():
    st.set_page_config(
        page_title="Data Modelling",
        page_icon="📈",
        layout="wide",  # centered, wide
        initial_sidebar_state="auto"  # auto, expanded, collapsed
    )
    st.title('Budget Allocation')

    tabs = st.tabs(['Market Mix Modeling', 'Spend Optimisation', 'Budget Reallocation'])
    with tabs[0]:
        uploaded_files = st.file_uploader('Upload your files', accept_multiple_files=True, type=['csv'])
        if uploaded_files:
            df, mmm_df, date_range = upload_files(uploaded_files)
            mmm_model(mmm_df, date_range)

    with tabs[1]:
        if uploaded_files:
            spend_optimisation(df)

    with tabs[2]:
        if uploaded_files:
            budget_allocate(df, date_range)


if __name__ == "__main__":
    run()
