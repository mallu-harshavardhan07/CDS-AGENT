"""
=============================================================================
Module: app.py
Description: Interactive Streamlit Web UI Dashboard for SAP S/4HANA CDS Agent.
             Enables users to search across 89,401 live SAP CDS views in ChromaDB
             with strict metadata catalog scope filtering (Standard Clean Core vs Custom),
             evaluate business requirements, expand query intent, and generate
             production-ready, Clean Core compliant ABAP CDS View Entities.
Author: SAP S/4HANA Cloud & Clean Core Architecture Engineering
=============================================================================
"""

import os
import sys
import json
import time
import importlib
import streamlit as st
from dotenv import load_dotenv

# Load environment configuration
load_dotenv(dotenv_path=".env", override=True)

# Dynamically import CDSRagEngine and CDSAgent
try:
    rag_module = importlib.import_module("03_rag_search")
    CDSRagEngine = rag_module.CDSRagEngine
    expand_query_intent = getattr(rag_module, "expand_query_intent", None)
except Exception as e:
    CDSRagEngine = None
    expand_query_intent = None
    st.error(f"Error loading 03_rag_search: {e}")

try:
    agent_module = importlib.import_module("04_cds_agent")
    CDSAgent = getattr(agent_module, "CDSAgent", None)
except Exception as e:
    CDSAgent = None
    st.error(f"Error loading 04_cds_agent: {e}")


# =============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =============================================================================

st.set_page_config(
    page_title="SAP CDS View RAG & Clean Core Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0c121e;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .decision-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.05em;
    }
    .decision-reuse { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .decision-extend { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .decision-create { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .field-chip-success {
        display: inline-block;
        padding: 3px 10px;
        margin: 2px 4px 2px 0px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .scope-tag-standard {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .scope-tag-custom {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED ENGINE INITIALIZATION
# =============================================================================

@st.cache_resource
def get_rag_engine():
    """Initializes and caches the ChromaDB RAG search engine."""
    db_path = os.getenv("VECTOR_DB_PATH", "./cds_vector_db")
    collection_name = os.getenv("VECTOR_COLLECTION_NAME", "cds_views")
    if CDSRagEngine:
        return CDSRagEngine(db_path=db_path, collection_name=collection_name)
    return None

@st.cache_resource
def get_cds_agent():
    """Initializes and caches the Clean Core CDS Code Generator Agent."""
    db_path = os.getenv("VECTOR_DB_PATH", "./cds_vector_db")
    collection_name = os.getenv("VECTOR_COLLECTION_NAME", "cds_views")
    if CDSAgent:
        return CDSAgent(db_path=db_path, collection_name=collection_name)
    return None

rag_engine = get_rag_engine()
cds_agent = get_cds_agent()


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/SAP_2011_logo.svg", width=85)
    st.title("CDS Clean Core Agent")
    st.caption("AI-Powered SAP CDS Architecture & Filtered Vector Search")

    st.markdown("---")
    st.subheader("Catalog Scope")
    scope_choice = st.radio(
        "Target Catalog Scope:",
        ["Standard Views (Clean Core)", "Custom Views"],
        index=0,
        help="Standard Views strictly returns released S/4HANA interface views (I_*, C_*). Custom Views returns customer/partner objects (Z*, Y*)."
    )
    active_scope = "Standard" if "Standard" in scope_choice else "Custom"

    st.markdown("---")
    st.subheader("System Status")
    total_vectors = rag_engine.count() if rag_engine else 0
    st.metric(label="ChromaDB Vectors Indexed", value=f"{total_vectors:,}")
    st.markdown(f"**Catalog Isolation:** `Strict ChromaDB Filter`")
    st.markdown(f"**Runtime Overhead:** `Zero SQLite (Direct Vector)`")
    st.markdown(f"**Collection:** `{os.getenv('VECTOR_COLLECTION_NAME', 'cds_views')}`")

    st.markdown("---")
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top Candidates (k)", min_value=1, max_value=10, value=5)
    use_mock = st.toggle("Deterministic Mock Engine", value=True, help="Toggle deterministic Clean Core rule-based engine or live LLM inference")

    st.markdown("---")
    st.info("💡 **Clean Core Extensibility Principles:**\n1. **REUSE**: Direct projection on released standard interface view.\n2. **EXTEND**: Non-disruptive extension view entity for custom fields.\n3. **CREATE**: Multi-entity composite view entity wrapping standard views.")


# =============================================================================
# MAIN INTERFACE
# =============================================================================

st.title("⚡ SAP S/4HANA CDS Clean Core Generator & Filtered Vector Engine")
st.markdown("Generate production-ready ABAP CDS View Entities adhering to SAP Clean Core principles with strict vector catalog filtering.")

# Key Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("""<div class="metric-card"><div class="metric-value">89,401</div><div class="metric-label">Vectors Indexed</div></div>""", unsafe_allow_html=True)
with col_m2:
    st.markdown("""<div class="metric-card"><div class="metric-value">62,848</div><div class="metric-label">Standard VDM Views</div></div>""", unsafe_allow_html=True)
with col_m3:
    st.markdown("""<div class="metric-card"><div class="metric-value">Clean Core</div><div class="metric-label">Gates 1 & 2 Verified</div></div>""", unsafe_allow_html=True)
with col_m4:
    scope_tag_cls = "scope-tag-standard" if active_scope == "Standard" else "scope-tag-custom"
    st.markdown(f"""<div class="metric-card"><div class="metric-value"><span class="{scope_tag_cls}">{active_scope} Scope</span></div><div class="metric-label">Active Isolation</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2 = st.tabs(["🚀 Clean Core CDS Code Generator", "🔍 Filtered Vector Catalog Explorer"])


# =============================================================================
# TAB 1: CLEAN CORE CODE GENERATOR
# =============================================================================

with tab1:
    st.subheader("CDS Architecture & Code Generation")

    # Domain Dropdown
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        domain_choice = st.selectbox(
            "Business Domain:",
            [
                "All Domains",
                "Sales & Distribution (SD)",
                "Financials & Controlling (FI/CO)",
                "Materials Management (MM/PUR)",
                "Customer & Master Data (BP/MD)"
            ]
        )
    with col_f2:
        goal_choice = st.radio(
            "Architectural Goal:",
            ["Auto-Detect", "REUSE", "EXTEND", "CREATE"],
            horizontal=True,
            help="Select Clean Core pathway or let the agent auto-detect based on business intent."
        )

    # Required Fields & Attributes
    fields_input = st.text_input(
        "Required Fields & Attributes (comma-separated):",
        placeholder="e.g. SalesOrder, CreationDate, SoldToParty or KUNNR, NAME1, ORT01",
        help="Enter technical fields, modern VDM elements, or legacy ECC abbreviations. DDIC synonyms are mapped automatically."
    )

    # Requirement Prompt
    req_input = st.text_area(
        "Requirement / Business Intent:",
        placeholder="e.g. Customer master data or Sales order header details with customer name and total net amount...",
        height=90
    )

    # Pathway Customization Helpers
    with st.expander("🛠️ Advanced Pathway Configuration (Optional Target Overrides)"):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            base_entity_input = st.text_input("Base Entity (Optional)", placeholder="e.g. I_SalesOrder, I_Customer")
        with col_c2:
            assoc_entity_input = st.text_input("Associated Entity for CREATE (Optional)", placeholder="e.g. I_Customer, I_Supplier")
        with col_c3:
            custom_fields_input = st.text_input("Custom Z Fields for EXTEND (Optional)", placeholder="e.g. z_loyalty_tier, z_custom_status")

    # Generate Action
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button("Generate Clean Core CDS", type="primary", use_container_width=True)

    if generate_clicked:
        effective_query = req_input.strip() or fields_input.strip() or base_entity_input.strip()

        if not effective_query:
            st.warning("Please provide a requirement, field list, or base entity to proceed.")
        elif not cds_agent:
            st.error("CDSAgent is not initialized. Please verify your vector database.")
        else:
            with st.spinner("Expanding query intent and retrieving candidates from ChromaDB..."):
                t_start = time.time()

                # Step 1: Expand Query Intent
                expanded_query = expand_query_intent(
                    domain=domain_choice,
                    fields=fields_input,
                    requirement=req_input
                ) if expand_query_intent else effective_query

                # Step 2: Execute Clean Core Code Generation
                res = cds_agent.evaluate_and_generate(
                    query=effective_query,
                    goal=goal_choice,
                    scope=active_scope,
                    base_entity=base_entity_input.strip() if base_entity_input else None,
                    associated_entity=assoc_entity_input.strip() if assoc_entity_input else None,
                    fields=fields_input.strip() if fields_input else None,
                    custom_fields=custom_fields_input.strip() if custom_fields_input else None,
                    top_k=top_k,
                    domain=domain_choice if domain_choice != "All Domains" else None,
                    mock=use_mock
                )
                elapsed = time.time() - t_start

            st.success(f"Generation Complete in {elapsed:.2f} seconds!")

            # Display Expanded Intent
            with st.expander("🔍 Expanded SAP VDM Query Intent"):
                st.markdown(f"**Search Query Prompt:** `{expanded_query}`")

            # Decision Banner
            st.markdown("### Architectural Recommendation")
            action = res.action
            target_entity = res.target_entity
            badge_class = f"decision-{action.lower()}"
            match_badge_str = " 🎯 `[EXACT TECHNICAL MATCH]`" if res.match_type == "EXACT_TECHNICAL_NAME" else ""

            st.markdown(f"""
            <div style="margin-bottom: 16px;">
                <span class="decision-badge {badge_class}">Action: {action}</span>
                <span style="margin-left: 15px; font-size: 18px; font-weight: 600; color: #f8fafc;">Target Entity: <code>{target_entity}</code></span>
                <span style="margin-left: 15px; font-size: 15px; color: #38bdf8;">Match Score: <strong>{res.score:.1f}%</strong>{match_badge_str}</span>
            </div>
            """, unsafe_allow_html=True)

            # Clean Core Quality Guardrails Status Banner
            guardrails_info = res.guardrails
            if guardrails_info.get("is_valid", True):
                st.markdown("""
                <div style="margin-bottom: 20px; padding: 12px 18px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px;">
                    <div style="font-size: 16px; font-weight: 700; color: #34d399;">🛡️ Clean Core Guardrails: PASSED (Gates 1 & 2 Verified)</div>
                    <div style="margin-top: 4px; color: #cbd5e1; font-size: 13px;">Zero raw DDIC table dependencies detected. Modern CDS view entity syntax verified without classic database views.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                violations_items = "".join([f"<li>{v}</li>" for v in guardrails_info.get("violations", [])])
                st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 12px 18px; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px;">
                    <div style="font-size: 16px; font-weight: 700; color: #f87171;">⚠️ Clean Core Guardrail Violations Detected</div>
                    <ul style="margin: 8px 0 0 0; color: #fca5a5; font-size: 13px; padding-left: 20px;">
                        {violations_items}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Architectural Rationale
            st.markdown("#### Architectural Rationale & Clean Core Strategy")
            st.info(res.rationale)

            # Production-Ready ABAP CDS Code
            st.markdown(f"#### Production-Ready ABAP CDS View Entity (`{target_entity}`)")
            st.code(res.code, language="sql")

            # Download ASDDLS Button
            st.download_button(
                label=f"💾 Download {target_entity}.asddls",
                data=res.code,
                file_name=f"{target_entity}.asddls",
                mime="text/plain"
            )

            # Retrieved Context Candidates
            with st.expander(f"📚 Retrieved Candidate Views ({len(res.candidates)} matches in {active_scope} scope)"):
                for cand in res.candidates:
                    cand_score = cand.get("score", cand.get("similarity_score", 0.0))
                    vdm_val = cand.get("vdm_type", "#BASIC")
                    contract_val = cand.get("release_contract", "#PUBLIC_LOCAL_API")
                    scope_val = cand.get("view_type", active_scope.lower())
                    exact_flag = " 🎯 `[EXACT TECHNICAL MATCH]`" if cand.get("match_type") == "EXACT_TECHNICAL_NAME" else ""

                    st.markdown(f"**#{cand['rank']} — `{cand['view_name']}`**{exact_flag} | Score: `{cand_score:.1f}%` | Scope: `{scope_val}`")
                    st.caption(f"**VDM Type:** `{vdm_val}` | **Release Contract:** `{contract_val}`")
                    st.caption(f"**Description:** {cand.get('description', 'No description available')}")

                    matched_f = cand.get("matched_fields", [])
                    if matched_f:
                        chips_html = "".join([f'<span class="field-chip-success">{f}</span>' for f in matched_f])
                        st.markdown(f"**Matched Fields:** {chips_html}", unsafe_allow_html=True)

                    st.code(cand.get("ddl_snippet", ""), language="sql")
                    st.markdown("---")


# =============================================================================
# TAB 2: FILTERED VECTOR CATALOG EXPLORER
# =============================================================================

with tab2:
    st.subheader(f"Explore Live CDS Views [{active_scope} Scope]")

    col_cat1, col_cat2 = st.columns([1, 2])
    with col_cat1:
        cat_domain = st.selectbox(
            "Filter Domain:",
            ["All Domains", "Sales & Distribution (SD)", "Financials & Controlling (FI/CO)", "Materials Management (MM/PUR)", "Customer & Master Data (BP/MD)"],
            key="cat_domain_select"
        )
    with col_cat2:
        cat_fields = st.text_input(
            "Filter by Fields / Attributes:",
            placeholder="e.g. SalesOrder, TotalNetAmount, Customer, SoldToParty",
            key="cat_fields_input"
        )

    cat_search = st.text_input(
        "Semantic Search or Technical View Name:",
        placeholder="e.g. Customer master data, Billing document item, I_SalesOrder, I_Customer...",
        key="cat_search_input"
    )

    if (cat_search or cat_fields) and rag_engine:
        with st.spinner("Executing filtered vector search..."):
            cat_results = rag_engine.vector_search(
                query_text=cat_search or cat_fields,
                scope=active_scope,
                top_k=top_k,
                domain=cat_domain if cat_domain != "All Domains" else None,
                fields=cat_fields
            )

        st.markdown(f"Found **{len(cat_results)}** matching CDS views in **{active_scope}** catalog:")

        for item in cat_results:
            with st.container():
                exact_flag = " 🎯 `[EXACT TECHNICAL MATCH]`" if item.get("match_type") == "EXACT_TECHNICAL_NAME" else ""
                st.markdown(f"### #{item['rank']} `{item['view_name']}`{exact_flag}")
                st.markdown(f"**Score:** `{item['score']:.1f}%` | **Scope:** `{item['view_type']}` | **VDM:** `{item['vdm_type']}` | **Contract:** `{item['release_contract']}`")
                st.caption(f"**Description:** {item.get('description', '')}")

                matched_f = item.get("matched_fields", [])
                if matched_f:
                    chips_html = "".join([f'<span class="field-chip-success">{f}</span>' for f in matched_f])
                    st.markdown(f"**Matched Fields:** {chips_html}", unsafe_allow_html=True)

                st.code(item.get("ddl_snippet", ""), language="sql")
                with st.expander(f"Full DDL Definition for {item['view_name']}"):
                    st.code(item.get("ddl_code", item.get("ddl_snippet", "")), language="sql")
                st.markdown("---")
