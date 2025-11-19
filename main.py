import streamlit as st
from app.quant_a.ui_quant_a import render_quant_a_page


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
            "Portfolio - Quant B (à venir)",
        ],
    )

    if mode.startswith("Single Asset"):
        render_quant_a_page()
    else:
        st.title("Quant B - Portfolio (à venir)")
        st.info("La partie Quant B sera intégrée plus tard.")

if __name__ == "__main__":
    main()
