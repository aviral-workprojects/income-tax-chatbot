"""
app.py
------
Main Streamlit application for the Indian Income Tax RAG Assistant.

Pages
-----
1. Chat Interface   – ask questions about the Income Tax Act
2. Tax Calculator   – compute slab-wise tax for old or new regime
3. Regime Comparison – side-by-side comparison of both regimes

The RAG pipeline is initialised once at startup using st.session_state
so it persists across Streamlit re-runs caused by user interactions.
"""

import os
import streamlit as st

from config import GROQ_API_KEY
from rag_pipeline import RAGPipeline
from tax_calculator import (
    compute_old_regime,
    compute_new_regime,
    compare_regimes,
    format_inr,
    TaxResult,
)

# ---------------------------------------------------------------------------
# Page configuration – must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Indian Income Tax Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS – no flashy effects, just clean typography
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Slightly increase base font size for readability */
        html, body, [class*="css"] { font-size: 15px; }

        /* Tighten up the sidebar */
        section[data-testid="stSidebar"] { min-width: 220px; max-width: 260px; }

        /* Source excerpt boxes */
        .source-box {
            background-color: #f7f7f7;
            border-left: 3px solid #888;
            padding: 10px 14px;
            margin-bottom: 12px;
            font-size: 0.88em;
            line-height: 1.55;
            color: #333;
        }

        /* Answer block */
        .answer-block {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 16px 20px;
            line-height: 1.7;
        }

        /* Table styling */
        table { width: 100%; border-collapse: collapse; }
        th { background-color: #f0f0f0; text-align: left; padding: 8px; }
        td { padding: 7px 8px; border-bottom: 1px solid #e0e0e0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Initialise the RAG pipeline (runs only once per session)
# ---------------------------------------------------------------------------

def get_pipeline() -> RAGPipeline:
    """
    Retrieve the RAGPipeline from session state, creating and initialising
    it if this is the first run.
    """
    if "pipeline" not in st.session_state:
        if not GROQ_API_KEY:
            st.error(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it before running the app:\n\n"
                "    export GROQ_API_KEY='your_key_here'"
            )
            st.stop()

        pdf_path = os.path.join(os.path.dirname(__file__), "data", "income_tax_act.pdf")
        if not os.path.exists(pdf_path):
            st.error(
                f"PDF not found at: {pdf_path}\n\n"
                "Please place the Income Tax Act PDF at data/income_tax_act.pdf "
                "before running the app."
            )
            st.stop()

        with st.spinner("Initialising the knowledge base. This may take a few minutes on the first run …"):
            pipeline = RAGPipeline()
            pipeline.initialise()

        st.session_state["pipeline"] = pipeline

    return st.session_state["pipeline"]


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Indian Income Tax Assistant")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    options=["Chat Interface", "Tax Calculator", "Regime Comparison"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Knowledge base:** Income Tax Act 1961, amended by Finance Act 2024.\n\n"
    "Answers are generated from the Act only. Always verify with a qualified CA."
)


# ===========================================================================
# PAGE 1 – Chat Interface
# ===========================================================================

def render_chat_page():
    st.title("Income Tax Query Assistant")
    st.markdown(
        "Ask any question about Indian income tax law. "
        "Answers are drawn exclusively from the Income Tax Act 1961 "
        "(as amended by Finance Act 2024)."
    )
    st.markdown("---")

    pipeline = get_pipeline()

    # Initialise chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"], expanded=False)

    # New question input
    question = st.chat_input("Type your question here …")

    if question:
        # Show the user's message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and display the answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant sections and generating answer …"):
                result = pipeline.ask(question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            _render_sources(sources, expanded=True)

        # Persist to chat history
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # Clear chat button
    if st.session_state.get("messages"):
        if st.button("Clear conversation"):
            st.session_state["messages"] = []
            st.rerun()


def _render_sources(sources: list, expanded: bool = False):
    """Render retrieved source excerpts inside an expander."""
    if not sources:
        return

    with st.expander(f"Retrieved source excerpts ({len(sources)} chunks)", expanded=expanded):
        for i, src in enumerate(sources, start=1):
            section_note = f"  |  Section {src['section']}" if src["section"] else ""
            header = f"Source {i}  |  Page {src['page']}{section_note}"
            st.markdown(
                f"<div class='source-box'>"
                f"<strong>{header}</strong><br><br>{src['text'][:600]}{'…' if len(src['text']) > 600 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ===========================================================================
# PAGE 2 – Tax Calculator
# ===========================================================================

def render_calculator_page():
    st.title("Income Tax Calculator")
    st.markdown(
        "Calculate your income tax liability for AY 2025-26 (FY 2024-25). "
        "Figures are for an individual taxpayer below 60 years of age."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Details")

        gross_income = st.number_input(
            "Annual Gross Income (₹)",
            min_value=0,
            max_value=100_000_000,
            value=800_000,
            step=10_000,
            format="%d",
            help="Enter your total annual income before any deductions.",
        )

        regime = st.selectbox(
            "Select Tax Regime",
            options=["New Regime (Default)", "Old Regime"],
        )

        other_deductions = 0
        if "Old" in regime:
            other_deductions = st.number_input(
                "Other Deductions (₹)",
                min_value=0,
                max_value=10_000_000,
                value=150_000,
                step=5_000,
                format="%d",
                help=(
                    "Sum of all eligible deductions such as 80C (up to ₹1.5L), "
                    "80D, HRA, etc. Standard deduction of ₹50,000 is added automatically."
                ),
            )

        calculate = st.button("Calculate Tax", type="primary")

    with col2:
        if calculate:
            if "Old" in regime:
                result = compute_old_regime(float(gross_income), float(other_deductions))
            else:
                result = compute_new_regime(float(gross_income))

            _render_tax_result(result)


def _render_tax_result(result: TaxResult):
    """Display a structured tax computation result."""
    st.subheader(f"Tax Computation – {result.regime}")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross Income", format_inr(result.gross_income))
    m2.metric("Deductions", format_inr(result.deductions))
    m3.metric("Taxable Income", format_inr(result.taxable_income))
    m4.metric("Total Tax Payable", format_inr(result.total_tax))

    st.markdown("---")

    # Slab-wise breakdown table
    st.markdown("**Slab-wise Tax Breakdown**")
    table_rows = "".join(
        f"<tr>"
        f"<td>{s.slab_label}</td>"
        f"<td style='text-align:right'>{format_inr(s.taxable_in_slab)}</td>"
        f"<td style='text-align:center'>{s.rate_pct:.0f}%</td>"
        f"<td style='text-align:right'>{format_inr(s.tax_in_slab)}</td>"
        f"</tr>"
        for s in result.slab_breakdown
    )
    st.markdown(
        f"""
        <table>
          <thead>
            <tr>
              <th>Slab</th>
              <th style='text-align:right'>Income in Slab</th>
              <th style='text-align:center'>Rate</th>
              <th style='text-align:right'>Tax</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Final computation summary
    st.markdown("**Final Computation**")
    rows = [
        ("Base Tax (before rebate)", format_inr(result.base_tax)),
        ("Less: Rebate u/s 87A", f"({format_inr(result.rebate_87a)})"),
        ("Tax after Rebate", format_inr(result.tax_after_rebate)),
        ("Add: Education Cess @ 4%", format_inr(result.cess)),
        ("**Total Tax Payable**", f"**{format_inr(result.total_tax)}**"),
        ("Effective Tax Rate", f"{result.effective_rate_pct:.2f}%"),
    ]
    for label, value in rows:
        c1, c2 = st.columns([3, 1])
        c1.markdown(label)
        c2.markdown(value)


# ===========================================================================
# PAGE 3 – Regime Comparison
# ===========================================================================

def render_comparison_page():
    st.title("Regime Comparison")
    st.markdown(
        "Compare your tax liability under the Old and New tax regimes "
        "to decide which is more beneficial for you."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Details")

        gross_income = st.number_input(
            "Annual Gross Income (₹)",
            min_value=0,
            max_value=100_000_000,
            value=1_200_000,
            step=10_000,
            format="%d",
            key="compare_income",
        )

        deductions_80c = st.number_input(
            "Section 80C Investments (₹)",
            min_value=0,
            max_value=150_000,
            value=150_000,
            step=5_000,
            format="%d",
            help="Maximum allowed: ₹1,50,000",
            key="compare_80c",
        )

        deductions_80d = st.number_input(
            "Section 80D – Health Insurance Premium (₹)",
            min_value=0,
            max_value=100_000,
            value=25_000,
            step=1_000,
            format="%d",
            key="compare_80d",
        )

        other_ded = st.number_input(
            "Other Deductions (₹)",
            min_value=0,
            max_value=5_000_000,
            value=0,
            step=5_000,
            format="%d",
            help="HRA, NPS (80CCD), home loan interest (Section 24), etc.",
            key="compare_other",
        )

        compare_btn = st.button("Compare Regimes", type="primary")

    with col2:
        if compare_btn:
            total_old_deductions = deductions_80c + deductions_80d + other_ded
            old_result, new_result = compare_regimes(
                float(gross_income), float(total_old_deductions)
            )

            _render_comparison_table(old_result, new_result)


def _render_comparison_table(old: TaxResult, new: TaxResult):
    """Render a side-by-side comparison table."""
    st.subheader("Side-by-Side Comparison")

    savings = new.total_tax - old.total_tax
    better = "Old Regime" if savings > 0 else "New Regime" if savings < 0 else "Equal"
    saved_amount = abs(savings)

    if savings > 0:
        st.info(
            f"The Old Regime saves you {format_inr(saved_amount)} "
            f"compared to the New Regime."
        )
    elif savings < 0:
        st.info(
            f"The New Regime saves you {format_inr(saved_amount)} "
            f"compared to the Old Regime."
        )
    else:
        st.info("Both regimes result in the same tax liability.")

    # Comparison table
    st.markdown(
        f"""
        <table>
          <thead>
            <tr>
              <th>Particulars</th>
              <th style='text-align:right'>Old Regime</th>
              <th style='text-align:right'>New Regime</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Gross Income</td>
                <td style='text-align:right'>{format_inr(old.gross_income)}</td>
                <td style='text-align:right'>{format_inr(new.gross_income)}</td></tr>
            <tr><td>Total Deductions</td>
                <td style='text-align:right'>{format_inr(old.deductions)}</td>
                <td style='text-align:right'>{format_inr(new.deductions)}</td></tr>
            <tr><td>Taxable Income</td>
                <td style='text-align:right'>{format_inr(old.taxable_income)}</td>
                <td style='text-align:right'>{format_inr(new.taxable_income)}</td></tr>
            <tr><td>Base Tax</td>
                <td style='text-align:right'>{format_inr(old.base_tax)}</td>
                <td style='text-align:right'>{format_inr(new.base_tax)}</td></tr>
            <tr><td>Rebate u/s 87A</td>
                <td style='text-align:right'>({format_inr(old.rebate_87a)})</td>
                <td style='text-align:right'>({format_inr(new.rebate_87a)})</td></tr>
            <tr><td>Education Cess (4%)</td>
                <td style='text-align:right'>{format_inr(old.cess)}</td>
                <td style='text-align:right'>{format_inr(new.cess)}</td></tr>
            <tr style='font-weight:bold'>
                <td>Total Tax Payable</td>
                <td style='text-align:right'>{format_inr(old.total_tax)}</td>
                <td style='text-align:right'>{format_inr(new.total_tax)}</td></tr>
            <tr><td>Effective Tax Rate</td>
                <td style='text-align:right'>{old.effective_rate_pct:.2f}%</td>
                <td style='text-align:right'>{new.effective_rate_pct:.2f}%</td></tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Per-slab detail for both regimes
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Old Regime – Slab Detail**")
        for s in old.slab_breakdown:
            st.text(
                f"{s.slab_label}  |  {s.rate_pct:.0f}%  |  {format_inr(s.tax_in_slab)}"
            )

    with c2:
        st.markdown("**New Regime – Slab Detail**")
        for s in new.slab_breakdown:
            st.text(
                f"{s.slab_label}  |  {s.rate_pct:.0f}%  |  {format_inr(s.tax_in_slab)}"
            )


# ===========================================================================
# Router
# ===========================================================================

if page == "Chat Interface":
    render_chat_page()
elif page == "Tax Calculator":
    render_calculator_page()
elif page == "Regime Comparison":
    render_comparison_page()
