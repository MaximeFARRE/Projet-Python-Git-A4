import streamlit as st
import datetime as dt

from app.quant_a.ui_quant_a import render_quant_a_page


def main():
    # Configuration globale
    st.set_page_config(
        page_title="Quant Dashboard - Indices",
        layout="wide",
    )

    # 🔄 Auto-refresh toutes les 5 minutes (300 secondes)
    st.markdown(
        "<meta http-equiv='refresh' content='300'>",
        unsafe_allow_html=True
    )

    # Info visuelle (utile en soutenance)
    st.caption(
        f"⏱️ Dernière actualisation automatique : "
        f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Choisir le module :",
        [
            "Single Asset - Quant A (CAC40)",
            "Portfolio - Quant B (à venir)",
        ],
    )

    # Routing
    if mode.startswith("Single Asset"):
        render_quant_a_page()
    else:
        st.title("Quant B - Portfolio (à venir)")
        st.info("La partie Quant B sera intégrée plus tard.")

if __name__ == "__main__":
    main()
