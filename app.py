from __future__ import annotations

import streamlit as st

from hitl_qualitative.ui import inject_styles, progress_page, review_page, setup_page


st.set_page_config(
    page_title="Qualitative Coding Review",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

navigation = st.navigation(
    [
        st.Page(setup_page, title="Setup", icon="⚙️", default=True),
        st.Page(review_page, title="Review", icon="📝"),
        st.Page(progress_page, title="Progress and export", icon="📊"),
    ]
)
navigation.run()

