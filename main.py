import streamlit as st
from app.quant_a.ui_quant_a import render_quant_a_page
from app.quant_b.page_quant_b import render as render_quant_b_page


def main():
    st.set_page_config(
        page_title="Quant Dashboard - Indices",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Choisir le module :",
        [
            "Single Asset - Quant A (CAC40)",
            "Portfolio - Quant B",
        ],
    )

    if mode.startswith("Single Asset"):
        render_quant_a_page()
    else:
        render_quant_b_page()


if __name__ == "__main__":
    main()
