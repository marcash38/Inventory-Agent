"""
Supply Chain Analytics - Inventory Decision Agent
Interactive Web Application powered by Streamlit and pure Python inventory models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review (Periodic Review Order-Up-To)
5. Newsvendor Model (Single-Period Decision)
6. AI Natural Language Decision Agent
"""

import math
import streamlit as st
import pandas as pd
import numpy as np

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_newsvendor,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
)

from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
    AgentResponse,
)

# =====================================================================
# Page Configuration & Global Styling
# =====================================================================

st.set_page_config(
    page_title="Supply Chain Analytics | Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive cards, metrics, and clean typography
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rec-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 15px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Overview
st.markdown('<div class="main-header">📦 Supply Chain Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Inventory Optimization, Verification & Decision Support System</div>',
    unsafe_allow_html=True
)

# =====================================================================
# Sidebar: System Context & Quick Navigation
# =====================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/warehouse-window.png", width=64)
    st.title("Decision Framework")
    st.markdown("""
    **Deterministic Execution Pipeline:**
    1. **Customer Query**
    2. **Identify Model**
    3. **Extract Inputs**
    4. **Select Python Tool**
    5. **Calculate**
    6. **Verify Optimality**
    7. **Explain & Insights**
    """)
    st.divider()
    st.markdown("### Core Inventory Policies")
    st.caption("• **EOQ:** Optimal continuous replenishment lot size")
    st.caption("• **Quantity Discount:** Multi-tier bulk price optimization")
    st.caption("• **Safety Stock & ROP:** 4 uncertainty demand/lead-time regimes")
    st.caption("• **Fixed-Period Review:** Periodic review order-up-to (R, S)")
    st.caption("• **Newsvendor:** Single-period critical fractile stocking")
    st.caption("• **AI Agent:** Natural language supply chain inquiry")
    st.divider()
    st.caption("Powered by `inventory_tools.py` & `inventory_agent.py`")

    # Data Upload Section (Last section in the sidebar)
    st.divider()
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# =====================================================================
# CSV Data Processing & Multicriteria Classification Helpers
# =====================================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Safely converts string or numeric series to float, stripping currency symbols and commas."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0)
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

def assign_pareto_classes(series: pd.Series, cutoffs=(0.70, 0.90, 0.98)) -> pd.Series:
    """
    Assigns Pareto inventory classes (A, B, C, D) to a numeric series sorted descending.
    Cutoffs:
      - Cumulative volume <= 70% -> Class A (Fast movers / High monetary value)
      - Cumulative volume 70% - 90% -> Class B (Moderate importance)
      - Cumulative volume 90% - 98% -> Class C (Low importance)
      - Cumulative volume > 98% -> Class D (Very low / trailing velocity)
    """
    sorted_s = series.sort_values(ascending=False, kind="mergesort")
    total = float(sorted_s.sum())
    if total <= 0:
        return pd.Series("D", index=series.index)
    shares = (sorted_s / total).round(6)
    cum_shares = shares.cumsum().round(6)
    prev_cum = (cum_shares - shares).round(6)

    class_list = []
    for cum, prev in zip(cum_shares, prev_cum):
        if prev < cutoffs[0] or cum <= cutoffs[0]:
            class_list.append("A")
        elif prev < cutoffs[1] or cum <= cutoffs[1]:
            class_list.append("B")
        elif prev < cutoffs[2] or cum <= cutoffs[2]:
            class_list.append("C")
        else:
            class_list.append("D")

    assigned_series = pd.Series(class_list, index=sorted_s.index)
    return assigned_series.loc[series.index]

def get_top_items_by_sales(df_classified: pd.DataFrame, class_label: str, top_n: int = 5) -> pd.DataFrame:
    """
    Extracts the top N items sorted by Sales ($) descending for a specified final multicriteria class.
    Returns a DataFrame with columns:
    Item Number, Item Name, Sales ($), Hits, Final Multicriteria Class
    """
    sub = df_classified[df_classified["Multicriteria Class"] == class_label].copy()
    sub_sorted = sub.sort_values(by="_Sales_Numeric", ascending=False).head(top_n)

    top_df = pd.DataFrame({
        "Item Number": sub_sorted["Item Number"],
        "Item Name": sub_sorted["Item Name"],
        "Sales ($)": sub_sorted["_Sales_Numeric"].apply(lambda x: f"${x:,.2f}"),
        "Hits": sub_sorted["_Hits_Numeric"].apply(lambda x: f"{x:,.0f}"),
        "Final Multicriteria Class": sub_sorted["Multicriteria Class"]
    })
    return top_df

def perform_multicriteria_classification(
    df: pd.DataFrame,
    sales_col: str = "Sales ($)",
    hits_col: str = "Hits",
    w_sales: float = 0.5,
    w_hits: float = 0.5
) -> pd.DataFrame:
    """
    Performs multicriteria inventory classification into A, B, C, and D classes
    using Sales ($) (revenue volume) and Hits (pick/transaction frequency).
    Adds classification results and cumulative shares to the dataframe.
    Guarantees 100% record retention with zero dropped or duplicated items.
    """
    res = df.copy()
    original_index = res.index
    res = res.reset_index(drop=True)

    clean_sales = clean_numeric_series(res[sales_col])
    clean_hits = clean_numeric_series(res[hits_col])

    res["_Sales_Numeric"] = clean_sales
    res["_Hits_Numeric"] = clean_hits

    # Single-dimension Pareto classifications
    res["Sales Class"] = assign_pareto_classes(clean_sales)
    res["Hits Class"] = assign_pareto_classes(clean_hits)

    # Share calculations
    tot_sales = float(clean_sales.sum())
    tot_hits = float(clean_hits.sum())

    sales_share = (clean_sales / tot_sales) if tot_sales > 0 else pd.Series(0.0, index=res.index)
    hits_share = (clean_hits / tot_hits) if tot_hits > 0 else pd.Series(0.0, index=res.index)

    # Cumulative percentages for both dimensions
    sales_sorted_idx = clean_sales.sort_values(ascending=False, kind="mergesort").index
    hits_sorted_idx = clean_hits.sort_values(ascending=False, kind="mergesort").index

    cum_sales_pct = (clean_sales.loc[sales_sorted_idx] / tot_sales * 100).cumsum().round(2) if tot_sales > 0 else pd.Series(0.0, index=sales_sorted_idx)
    cum_hits_pct = (clean_hits.loc[hits_sorted_idx] / tot_hits * 100).cumsum().round(2) if tot_hits > 0 else pd.Series(0.0, index=hits_sorted_idx)

    res["Cumulative Sales %"] = cum_sales_pct.reindex(res.index)
    res["Cumulative Hits %"] = cum_hits_pct.reindex(res.index)

    # Multi-attribute composite score and Pareto classification
    composite_score = (w_sales * sales_share + w_hits * hits_share).round(6)
    res["Composite Score"] = composite_score
    res["Multicriteria Class"] = assign_pareto_classes(composite_score)

    # 2D Cross-criteria matrix identifier (e.g. AA, AB, BA, CD)
    res["Matrix Class"] = res["Sales Class"] + res["Hits Class"]

    res.index = original_index
    return res

# =====================================================================
# CSV Data Processing & Analysis (when CSV is uploaded in sidebar)
# =====================================================================

if uploaded_file is not None:
    st.header("📁 Uploaded Inventory Data & Analysis")
    try:
        # 1. Read CSV file into pandas DataFrame
        df_uploaded = pd.read_csv(uploaded_file)

        # Normalize column names by stripping whitespace
        df_uploaded.columns = df_uploaded.columns.str.strip()

        # Schema 1: Inventory Overview (Product, Category, Units, Unit Cost)
        overview_required = ["Product", "Category", "Units", "Unit Cost"]
        has_overview_schema = set(overview_required).issubset(df_uploaded.columns)

        # Schema 2: Multicriteria Classification (Item Number, Item Name, UoM, Sales ($), Hits)
        classification_required = ["Item Number", "Item Name", "UoM", "Hits"]
        sales_col = None
        for candidate in ["Sales ($)", "Sales", "Sales($)", "Sales Amount", "Sales_Amount", "Revenue"]:
            if candidate in df_uploaded.columns:
                sales_col = candidate
                break

        has_classification_schema = set(classification_required).issubset(df_uploaded.columns) and (sales_col is not None)

        if not has_overview_schema and not has_classification_schema:
            st.error(
                "⚠️ Uploaded CSV file does not match either supported inventory schema:\n\n"
                "• **Multicriteria Classification Schema:** Requires `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
                "• **Inventory Overview Schema:** Requires `Product`, `Category`, `Units`, `Unit Cost`\n\n"
                f"**Columns detected in uploaded file:** {', '.join(df_uploaded.columns)}"
            )

        # =================================================================
        # MODULE A: Multicriteria Inventory Classification (Sales & Hits)
        # =================================================================
        if has_classification_schema:
            st.subheader("🏷️ Multicriteria Inventory Classification (A, B, C, D Classes)")
            st.caption(
                "Combines **Sales ($)** (monetary revenue impact) and **Hits** (pick transaction velocity) "
                "to perform multi-attribute Pareto inventory classification."
            )

            # Execute Multicriteria Classification using pandas
            df_classified = perform_multicriteria_classification(
                df_uploaded,
                sales_col=sales_col,
                hits_col="Hits",
                w_sales=0.5,
                w_hits=0.5
            )

            # Metadata & Data Integrity Audit
            total_items = len(df_uploaded)
            classified_items = len(df_classified)
            missing_items = int(df_classified["Multicriteria Class"].isna().sum())
            duplicate_records = int(df_uploaded.duplicated().sum())
            duplicate_item_nums = int(df_uploaded["Item Number"].duplicated().sum()) if "Item Number" in df_uploaded.columns else 0

            meta_c1, meta_c2, meta_c3, meta_c4 = st.columns([1, 1, 1, 1.5])
            meta_c1.metric("Total Items Uploaded", f"{total_items:,}")
            meta_c2.metric("Total Items Classified", f"{classified_items:,}")
            meta_c3.metric("Missing Records", f"{missing_items}")
            meta_c4.metric("Duplicate Items", f"{duplicate_item_nums}")

            # Audit confirmation badge
            if total_items == classified_items and missing_items == 0 and duplicate_records == 0:
                st.success(
                    f"✅ **Audit Verified:** All **{total_items}** inventory items were classified "
                    f"without missing or duplicated records (100% record retention and completeness)."
                )
            else:
                st.warning(
                    f"⚠️ **Audit Notice:** {classified_items}/{total_items} items classified. "
                    f"Missing: {missing_items}, Duplicate records: {duplicate_records}."
                )

            # Display Classification Rules and Thresholds
            with st.expander("📐 Classification Rules & Thresholds Used to Assign Each Class", expanded=True):
                st.markdown(r"""
                **Multicriteria Inventory Classification Rules & Thresholds:**

                1. **Monetary Dimension — Sales ($) Pareto Thresholds:**
                   - **Class A:** Cumulative Sales $\le 70\%$ (Top revenue drivers / Highest monetary velocity)
                   - **Class B:** Cumulative Sales $70\% - 90\%$ (Next 20% of revenue / Moderate monetary velocity)
                   - **Class C:** Cumulative Sales $90\% - 98\%$ (Next 8% of revenue / Low monetary velocity)
                   - **Class D:** Cumulative Sales $> 98\%$ (Trailing 2% or negligible revenue)

                2. **Transaction Velocity Dimension — Hits Pareto Thresholds:**
                   - **Class A:** Cumulative Hits $\le 70\%$ (Fast movers / High pick frequency lines)
                   - **Class B:** Cumulative Hits $70\% - 90\%$ (Next 20% of picks / Moderate pick frequency)
                   - **Class C:** Cumulative Hits $90\% - 98\%$ (Next 8% of picks / Slow pick frequency)
                   - **Class D:** Cumulative Hits $> 98\%$ (Trailing 2% / Dead stock / Rare picks)

                3. **Multi-Attribute Composite Scoring Formula:**
                   $$\text{Composite Score}_i = 0.50 \times \left(\frac{\text{Sales}_i}{\sum \text{Sales}}\right) + 0.50 \times \left(\frac{\text{Hits}_i}{\sum \text{Hits}}\right)$$
                   - Equal weighting ($50\%$ Sales contribution $+ 50\%$ Hits contribution).
                   - Items are sorted by Composite Score descending and assigned to **Final Multicriteria Class** using the standard Pareto cumulative cutoffs ($\le 70\% \to \text{A}$, $70\%-90\% \to \text{B}$, $90\%-98\% \to \text{C}$, $> 98\% \to \text{D}$).

                4. **2D Operational Cross-Classification Matrix (`Matrix Class`):**
                   - Combines individual `Sales Class` and `Hits Class` (e.g. `AA`, `AB`, `AD`, `DA`, `DD`) for warehouse slotting, cycle count frequencies, and replenishment review policies.
                """)

            tot_portfolio_sales = float(df_classified["_Sales_Numeric"].sum())
            tot_portfolio_hits = float(df_classified["_Hits_Numeric"].sum())

            # Count of items in each Multicriteria Class
            class_counts = df_classified["Multicriteria Class"].value_counts().to_dict()
            cnt_a = class_counts.get("A", 0)
            cnt_b = class_counts.get("B", 0)
            cnt_c = class_counts.get("C", 0)
            cnt_d = class_counts.get("D", 0)
            n_items = len(df_classified) if len(df_classified) > 0 else 1

            # Sales in each class
            class_sales = df_classified.groupby("Multicriteria Class")["_Sales_Numeric"].sum().to_dict()
            sales_a = class_sales.get("A", 0.0)
            sales_b = class_sales.get("B", 0.0)
            sales_c = class_sales.get("C", 0.0)
            sales_d = class_sales.get("D", 0.0)

            # Executive KPI Cards
            st.markdown("##### Executive Portfolio Summary")
            kpi_cols = st.columns(6)
            kpi_cols[0].metric("Total Sales ($)", f"${tot_portfolio_sales:,.2f}")
            kpi_cols[1].metric("Total Pick Hits", f"{tot_portfolio_hits:,.0f}")
            kpi_cols[2].metric("Class A Items", f"{cnt_a} ({cnt_a/n_items*100:.1f}%)", f"${sales_a:,.0f}")
            kpi_cols[3].metric("Class B Items", f"{cnt_b} ({cnt_b/n_items*100:.1f}%)", f"${sales_b:,.0f}")
            kpi_cols[4].metric("Class C Items", f"{cnt_c} ({cnt_c/n_items*100:.1f}%)", f"${sales_c:,.0f}")
            kpi_cols[5].metric("Class D Items", f"{cnt_d} ({cnt_d/n_items*100:.1f}%)", f"${sales_d:,.0f}")

            # =============================================================
            # Top 5 Items by Sales for Each Final Inventory Class
            # =============================================================
            st.markdown("---")
            st.subheader("Top 5 Items by Sales for Each Final Inventory Class")
            st.caption(
                "Separate tables for Classes A, B, C, and D, each showing Item Number, Item Name, "
                "Sales ($), Hits, and Final Multicriteria Class, sorted from highest to lowest Sales ($)."
            )

            col_top_a, col_top_b = st.columns(2)
            with col_top_a:
                st.markdown("##### 🥇 Class A — Top 5 Items by Sales")
                tbl_a = get_top_items_by_sales(df_classified, "A", top_n=5)
                if len(tbl_a) > 0:
                    st.dataframe(tbl_a, width="stretch")
                else:
                    st.info("No items in Class A.")

                st.markdown("##### 🥉 Class C — Top 5 Items by Sales")
                tbl_c = get_top_items_by_sales(df_classified, "C", top_n=5)
                if len(tbl_c) > 0:
                    st.dataframe(tbl_c, width="stretch")
                else:
                    st.info("No items in Class C.")

            with col_top_b:
                st.markdown("##### 🥈 Class B — Top 5 Items by Sales")
                tbl_b = get_top_items_by_sales(df_classified, "B", top_n=5)
                if len(tbl_b) > 0:
                    st.dataframe(tbl_b, width="stretch")
                else:
                    st.info("No items in Class B.")

                st.markdown("##### 📦 Class D — Top 5 Items by Sales")
                tbl_d = get_top_items_by_sales(df_classified, "D", top_n=5)
                if len(tbl_d) > 0:
                    st.dataframe(tbl_d, width="stretch")
                else:
                    st.info("No items in Class D.")

            st.markdown("---")

            # Analytics Tabs
            tab_summary, tab_matrix, tab_table = st.tabs([
                "📊 Classification Distribution & Charts",
                "🎯 2D Multicriteria Matrix (Sales × Hits)",
                "📋 Complete Classified Inventory Table"
            ])

            with tab_summary:
                # Grouped summary table
                summary_records = []
                for cls_label in ["A", "B", "C", "D"]:
                    sub = df_classified[df_classified["Multicriteria Class"] == cls_label]
                    cls_items = len(sub)
                    cls_sales_val = float(sub["_Sales_Numeric"].sum())
                    cls_hits_val = float(sub["_Hits_Numeric"].sum())
                    summary_records.append({
                        "Class": cls_label,
                        "Item Count": cls_items,
                        "% of Items": f"{(cls_items / n_items * 100):.1f}%",
                        "Total Sales ($)": f"${cls_sales_val:,.2f}",
                        "% of Sales": f"{(cls_sales_val / tot_portfolio_sales * 100 if tot_portfolio_sales > 0 else 0.0):.1f}%",
                        "Total Hits": f"{cls_hits_val:,.0f}",
                        "% of Hits": f"{(cls_hits_val / tot_portfolio_hits * 100 if tot_portfolio_hits > 0 else 0.0):.1f}%"
                    })
                df_summary = pd.DataFrame(summary_records)
                st.dataframe(df_summary, width="stretch")

                # Charts
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("**Item Count by Multicriteria Class**")
                    chart_counts = pd.DataFrame({
                        "Class": ["A", "B", "C", "D"],
                        "Items": [cnt_a, cnt_b, cnt_c, cnt_d]
                    }).set_index("Class")
                    st.bar_chart(chart_counts["Items"])

                with chart_col2:
                    st.markdown("**Total Sales ($) by Multicriteria Class**")
                    chart_sales_df = pd.DataFrame({
                        "Class": ["A", "B", "C", "D"],
                        "Sales ($)": [sales_a, sales_b, sales_c, sales_d]
                    }).set_index("Class")
                    st.bar_chart(chart_sales_df["Sales ($)"])

            with tab_matrix:
                st.markdown("##### 2D Cross-Classification Matrix (`Sales Class` × `Hits Class`)")
                st.caption("Distribution of SKUs across monetary volume (Sales) and physical pick frequency (Hits).")
                matrix_df = pd.crosstab(
                    df_classified["Sales Class"],
                    df_classified["Hits Class"],
                    margins=True,
                    margins_name="Total"
                )
                st.dataframe(matrix_df, width="stretch")

                st.markdown("""
                **Operational Management Strategies:**
                - **Class A-A (High Sales, High Hits):** Fast-moving top earners. Place near shipping bays; implement continuous review, automated replenishment, and tight cycle counts.
                - **Class A-D (High Sales, Low Hits):** High dollar value, slow movement. Expensive capital risk; store in secure high-bay locations with periodic review.
                - **Class D-A (Low Sales, High Hits):** Low unit revenue, high pick frequency. Essential consumable components; position at ergonomic pick faces to minimize picker travel.
                - **Class D-D (Low Sales, Low Hits):** Dead or obsolete stock candidates. Review for vendor rationalization, minimum batch sizing, or liquidation.
                """)

            with tab_table:
                st.markdown("##### Enriched Inventory Dataset")
                filter_choice = st.radio(
                    "Filter by Multicriteria Class:",
                    ["All", "A", "B", "C", "D"],
                    horizontal=True
                )

                df_view = df_classified if filter_choice == "All" else df_classified[df_classified["Multicriteria Class"] == filter_choice]

                # Prepare clean display DataFrame
                cols_to_show = [
                    "Item Number", "Item Name", "UoM", sales_col, "Hits",
                    "Sales Class", "Hits Class", "Multicriteria Class", "Matrix Class",
                    "Cumulative Sales %", "Cumulative Hits %"
                ]
                existing_cols_to_show = [c for c in cols_to_show if c in df_view.columns]
                df_display = df_view[existing_cols_to_show].copy()

                # Currency formatting for display
                df_display[sales_col] = df_view["_Sales_Numeric"].apply(lambda x: f"${x:,.2f}")
                df_display["Hits"] = df_view["_Hits_Numeric"].apply(lambda x: f"{x:,.0f}")
                df_display["Cumulative Sales %"] = df_view["Cumulative Sales %"].apply(lambda x: f"{x:.2f}%")
                df_display["Cumulative Hits %"] = df_view["Cumulative Hits %"].apply(lambda x: f"{x:.2f}%")

                st.dataframe(df_display, width="stretch")

                # Export Classified CSV
                csv_download = df_classified.drop(columns=["_Sales_Numeric", "_Hits_Numeric"]).to_csv(index=False)
                st.download_button(
                    label="📥 Download Classified Inventory CSV",
                    data=csv_download,
                    file_name="classified_inventory_results.csv",
                    mime="text/csv",
                    width="stretch"
                )

        # =================================================================
        # MODULE B: Existing Inventory Overview (Product, Category, Units, Unit Cost)
        # =================================================================
        if has_overview_schema:
            st.subheader("📊 Basic Inventory Portfolio Overview")
            meta_c1, meta_c2, meta_c3 = st.columns([1, 1, 2])
            meta_c1.metric("Number of Rows", f"{len(df_uploaded):,}")
            meta_c2.metric("Number of Columns", f"{len(df_uploaded.columns):,}")
            meta_c3.write(f"**Column Names:** {', '.join(df_uploaded.columns)}")

            st.markdown("##### Uploaded Dataframe Preview")
            st.dataframe(df_uploaded, width="stretch")

            # DATA ANALYSIS using pandas and the uploaded dataframe:
            df_analysis = df_uploaded.copy()
            df_analysis["Units"] = pd.to_numeric(df_analysis["Units"], errors="coerce").fillna(0)
            df_analysis["Unit Cost"] = pd.to_numeric(df_analysis["Unit Cost"], errors="coerce").fillna(0.0)

            # 1. Calculate total Units
            total_units = float(df_analysis["Units"].sum())

            # 2. Calculate average Unit Cost
            avg_unit_cost = float(df_analysis["Unit Cost"].mean()) if len(df_analysis) > 0 else 0.0

            # 3. Identify Product with highest Units
            if len(df_analysis) > 0:
                highest_idx = df_analysis["Units"].idxmax()
                highest_product = str(df_analysis.loc[highest_idx, "Product"])
                highest_units = float(df_analysis.loc[highest_idx, "Units"])
            else:
                highest_product = "N/A"
                highest_units = 0.0

            # 4. Group data by Category and calculate total Units for each Category
            category_units_df = (
                df_analysis.groupby("Category")["Units"]
                .sum()
                .reset_index()
                .sort_values(by="Units", ascending=False)
            )

            # 5. Create new calculated column: Inventory Value = Units × Unit Cost
            df_analysis["Inventory Value"] = df_analysis["Units"] * df_analysis["Unit Cost"]

            # 6. Calculate total Inventory Value across all products
            total_inventory_value = float(df_analysis["Inventory Value"].sum())

            # RESULTS DISPLAY
            st.markdown("##### Key Inventory Metrics")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            kpi_c1.metric("Total Units", f"{total_units:,.0f}")
            kpi_c2.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
            kpi_c3.metric("Product with Highest Units", f"{highest_product}", f"{highest_units:,.0f} units")
            kpi_c4.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

            # Total Units by Category
            st.markdown("##### Total Units by Category")
            cat_col1, cat_col2 = st.columns([1, 1.2])
            with cat_col1:
                st.dataframe(category_units_df, width="stretch")
            with cat_col2:
                st.bar_chart(category_units_df.set_index("Category")["Units"])

            # Dataframe including the new Inventory Value column with currency formatting
            st.markdown("##### Complete Inventory Dataframe (with Calculated Inventory Value)")
            df_display = df_analysis.copy()
            df_display["Unit Cost"] = df_display["Unit Cost"].apply(lambda x: f"${x:,.2f}")
            df_display["Inventory Value"] = df_display["Inventory Value"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(df_display, width="stretch")

    except Exception as e:
        st.error(f"Error reading or processing CSV file: {str(e)}")

    st.divider()

# =====================================================================
# Model Selection Dropdown
# =====================================================================

model = st.selectbox(
    "Select an Inventory Model",
    [
        "Economic Order Quantity (EOQ)",
        "Quantity Discount Model",
        "Safety Stock and Reorder Point",
        "Fixed-Period Review",
        "Newsvendor Model",
        "🤖 AI Natural Language Decision Agent"
    ],
    index=0
)

st.divider()


# =====================================================================
# Model 1: Economic Order Quantity (EOQ)
# =====================================================================

if model == "Economic Order Quantity (EOQ)":
    st.header("📦 Economic Order Quantity (EOQ) Calculator")
    st.write(
        "Find the optimal order quantity ($Q^*$) that minimizes the total annual inventory cost "
        "by balancing setup/ordering costs with inventory holding costs."
    )

    col1, col2, col3 = st.columns([1.1, 1.1, 1.0])

    with col1:
        st.subheader("1. Demand & Ordering")
        annual_demand = st.number_input(
            "Annual Demand (D, units/year)",
            min_value=1.0,
            value=12000.0,
            step=500.0,
            help="Total annual customer demand in physical units."
        )
        ordering_cost = st.number_input(
            "Ordering / Setup Cost (S, $ per order)",
            min_value=0.01,
            value=50.0,
            step=5.0,
            help="Fixed administrative and handling expense incurred each time an order is placed."
        )

    with col2:
        st.subheader("2. Holding / Carrying Cost")
        holding_mode = st.radio(
            "Holding Cost Specification",
            ["Fixed Holding Cost ($/unit/year)", "Percentage of Unit Purchase Price (%)"],
            help="Specify carrying cost as either a fixed dollar amount per unit or a percentage of purchase price."
        )

        unit_cost = None
        holding_cost_rate = None
        holding_cost_per_unit = None

        if holding_mode == "Fixed Holding Cost ($/unit/year)":
            holding_cost_per_unit = st.number_input(
                "Holding Cost per Unit (H, $/unit/year)",
                min_value=0.01,
                value=4.0,
                step=0.5,
                help="Annual capital, warehousing, insurance, and obsolescence expense per unit."
            )
        else:
            unit_cost = st.number_input(
                "Unit Purchase Price (C, $)",
                min_value=0.01,
                value=25.0,
                step=1.0,
                help="Purchase cost paid per unit to the supplier."
            )
            holding_rate_pct = st.number_input(
                "Annual Holding Rate (h, % of unit price)",
                min_value=0.1,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Annual carrying cost rate as a percentage of unit purchase cost."
            )
            holding_cost_rate = holding_rate_pct / 100.0

    with col3:
        st.subheader("3. Operational Parameters")
        lead_time_days = st.number_input(
            "Replenishment Lead Time (days, optional)",
            min_value=0.0,
            value=5.0,
            step=1.0,
            help="Time required from order placement to delivery arrival."
        )
        work_days = st.number_input(
            "Operating Days per Year",
            min_value=1.0,
            max_value=365.0,
            value=365.0,
            step=5.0,
            help="Annual operating or business days (default: 365)."
        )

    if st.button("Calculate EOQ", type="primary", width="stretch"):
        try:
            # Deterministic execution via inventory_tools.py
            result = calculate_eoq(
                annual_demand=annual_demand,
                ordering_cost=ordering_cost,
                holding_cost_per_unit=holding_cost_per_unit,
                holding_cost_rate=holding_cost_rate,
                unit_cost=unit_cost,
                lead_time_days=lead_time_days if lead_time_days > 0 else None,
                work_days=work_days
            )

            st.markdown(
                f'<div class="rec-box">'
                f'<h4>🎯 Executive Recommendation</h4>'
                f'<b>Order {result.optimal_order_quantity:,.0f} units</b> per replenishment order '
                f'(approx. <b>{result.orders_per_year:.1f} orders/year</b>). '
                f'Total annual inventory policy cost is <b>${result.total_inventory_cost:,.2f}</b>.'
                f'</div>',
                unsafe_allow_html=True
            )

            # Key Performance Indicators
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Optimal Batch Size (Q*)", f"{result.optimal_order_quantity:,.1f} units", f"{round(result.optimal_order_quantity)} units")
            kpi2.metric("Total Inventory Cost", f"${result.total_inventory_cost:,.2f}")
            kpi3.metric("Annual Ordering Cost", f"${result.annual_ordering_cost:,.2f}")
            kpi4.metric("Annual Holding Cost", f"${result.annual_holding_cost:,.2f}")

            kpi5, kpi6, kpi7, kpi8 = st.columns(4)
            kpi5.metric("Order Frequency", f"{result.orders_per_year:.2f} orders/yr")
            kpi6.metric("Cycle Time Between Orders", f"{result.cycle_time_days:.1f} days", f"{result.cycle_time_years*52:.1f} weeks")
            kpi7.metric("Average Cycle Inventory", f"{result.average_cycle_inventory:,.1f} units")
            if result.reorder_point is not None:
                kpi8.metric("Reorder Point (ROP)", f"{result.reorder_point:,.1f} units", "Place order at this level")
            elif result.total_acquisition_cost is not None:
                kpi8.metric("Total Acquisition Cost", f"${result.total_acquisition_cost:,.2f}")

            # Cost Sensitivity Chart
            st.subheader("📊 Inventory Cost Trade-Off Curve")
            q_vals = np.linspace(max(10, result.optimal_order_quantity * 0.2), result.optimal_order_quantity * 2.5, 60)
            h_eff = result.holding_cost_per_unit
            df_costs = pd.DataFrame({
                "Order Quantity (Q)": q_vals,
                "Annual Ordering Cost ($)": (annual_demand / q_vals) * ordering_cost,
                "Annual Holding Cost ($)": (q_vals / 2.0) * h_eff,
                "Total Annual Cost ($)": ((annual_demand / q_vals) * ordering_cost) + ((q_vals / 2.0) * h_eff)
            }).set_index("Order Quantity (Q)")
            st.line_chart(df_costs)

            # Technical Details & Verification
            tab1, tab2, tab3 = st.tabs(["📐 Mathematical Derivation", "✅ Verification Audit", "📋 Model Assumptions"])
            with tab1:
                st.code(
                    f"Optimal Order Quantity (EOQ):\n"
                    f"  Q* = sqrt((2 * D * S) / H)\n"
                    f"     = sqrt((2 * {result.annual_demand:,.0f} * ${result.ordering_cost:,.2f}) / ${result.holding_cost_per_unit:,.2f})\n"
                    f"     = {result.optimal_order_quantity:,.2f} units\n\n"
                    f"Annual Ordering Cost = (D / Q*) * S = ({result.annual_demand:,.0f} / {result.optimal_order_quantity:,.2f}) * ${result.ordering_cost:,.2f} = ${result.annual_ordering_cost:,.2f}\n"
                    f"Annual Holding Cost  = (Q* / 2) * H = ({result.optimal_order_quantity:,.2f} / 2) * ${result.holding_cost_per_unit:,.2f} = ${result.annual_holding_cost:,.2f}\n"
                    f"Total Inventory Cost = Annual Ordering Cost + Annual Holding Cost = ${result.total_inventory_cost:,.2f}",
                    language="text"
                )
            with tab2:
                check_data = [
                    {"Check": c.name, "Status": "PASS" if c.passed else "FAIL", "Details": c.message}
                    for c in result.checks
                ]
                st.table(pd.DataFrame(check_data))
            with tab3:
                for a in result.assumptions:
                    st.markdown(f"- {a}")

        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")


# =====================================================================
# Model 2: Quantity Discount Model
# =====================================================================

elif model == "Quantity Discount Model":
    st.header("🏷️ Quantity Discount Model Calculator")
    st.write(
        "Evaluate all-units quantity discount schedules across price tiers. "
        "Compares feasible unconstrained EOQs and price breakpoints to identify the global cost-minimizing order lot size."
    )

    col1, col2 = st.columns([1.1, 1.1])

    with col1:
        st.subheader("1. Demand & Ordering")
        annual_demand = st.number_input(
            "Annual Demand (D, units/year)",
            min_value=1.0,
            value=5000.0,
            step=500.0,
            help="Total annual requirements."
        )
        ordering_cost = st.number_input(
            "Ordering / Setup Cost (S, $ per order)",
            min_value=0.01,
            value=50.0,
            step=5.0,
            help="Fixed cost per order."
        )

    with col2:
        st.subheader("2. Holding Cost Specification")
        qd_holding_type = st.radio(
            "Holding Cost Rule",
            ["Percentage of Unit Price (%)", "Fixed Cost ($/unit/year)"],
            help="In all-units discount models, holding cost is typically a percentage of each tier's unit purchase price."
        )
        holding_rate = None
        fixed_holding = None
        if qd_holding_type == "Percentage of Unit Price (%)":
            hr_pct = st.number_input("Annual Holding Rate (h, %)", min_value=0.1, max_value=100.0, value=20.0, step=1.0)
            holding_rate = hr_pct / 100.0
        else:
            fixed_holding = st.number_input("Fixed Holding Cost (H, $/unit/year)", min_value=0.01, value=4.0, step=0.5)

    st.subheader("3. Quantity Discount Price Schedule")
    st.write("Configure the discount tiers and price breaks (Min Qty, Max Qty, Unit Price):")

    default_tiers = [
        {"Tier": 1, "Min Qty": 1, "Max Qty": 499, "Unit Price ($)": 20.0},
        {"Tier": 2, "Min Qty": 500, "Max Qty": 999, "Unit Price ($)": 18.0},
        {"Tier": 3, "Min Qty": 1000, "Max Qty": 999999, "Unit Price ($)": 16.0},
    ]
    edited_tiers = st.data_editor(pd.DataFrame(default_tiers), num_rows="dynamic", width="stretch")

    if st.button("Calculate Quantity Discount", type="primary", width="stretch"):
        try:
            # Parse schedule from edited table
            schedule = []
            for _, row in edited_tiers.iterrows():
                min_q = float(row["Min Qty"])
                max_q_raw = row["Max Qty"]
                max_q = None if pd.isna(max_q_raw) or float(max_q_raw) >= 999999 else float(max_q_raw)
                p_val = float(row["Unit Price ($)"])
                schedule.append({"min_qty": min_q, "max_qty": max_q, "unit_price": p_val})

            if not schedule:
                st.error("Please provide at least one discount tier.")
            else:
                result = calculate_quantity_discount(
                    annual_demand=annual_demand,
                    ordering_cost=ordering_cost,
                    discount_schedule=schedule,
                    holding_cost_rate=holding_rate,
                    fixed_holding_cost=fixed_holding
                )

                st.markdown(
                    f'<div class="rec-box">'
                    f'<h4>🎯 Executive Recommendation</h4>'
                    f'Order <b>{result.optimal_order_quantity:,.0f} units</b> at <b>${result.optimal_unit_price:.2f}/unit</b> '
                    f'(<b>Tier {result.optimal_tier_id}</b>). '
                    f'Generates <b>${result.annual_savings:,.2f} ({result.percentage_savings:.1f}%)</b> in annual net savings '
                    f'compared to the baseline price tier.'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recommended Tier", f"Tier {result.optimal_tier_id}", f"${result.optimal_unit_price:.2f}/unit")
                m2.metric("Optimal Order Quantity", f"{result.optimal_order_quantity:,.0f} units")
                m3.metric("Total Annual Cost", f"${result.optimal_total_cost:,.2f}")
                m4.metric("Annual Net Savings", f"${result.annual_savings:,.2f}", f"{result.percentage_savings:.1f}%")

                # Detailed Evaluation Table
                st.subheader("📋 Comprehensive Tier Evaluation")
                tier_rows = []
                for t in result.tiers:
                    max_label = f"{t.max_qty:,.0f}" if t.max_qty is not None else "Above"
                    tier_rows.append({
                        "Tier": t.tier_id,
                        "Quantity Range": f"{t.min_qty:,.0f} – {max_label}",
                        "Unit Price": f"${t.unit_price:.2f}",
                        "Holding Cost (H)": f"${t.holding_cost:.2f}",
                        "Unconstrained EOQ": f"{t.eoq_unconstrained:,.1f}",
                        "Candidate Lot Size": f"{t.candidate_qty:,.0f}",
                        "Feasible?": "Yes" if t.is_feasible else "No",
                        "Purchase Cost ($)": f"${t.annual_purchase_cost:,.2f}",
                        "Ordering Cost ($)": f"${t.annual_ordering_cost:,.2f}",
                        "Holding Cost ($)": f"${t.annual_holding_cost:,.2f}",
                        "Total Annual Cost ($)": f"${t.total_annual_cost:,.2f}",
                        "Optimal?": "⭐ BEST" if t.tier_id == result.optimal_tier_id else ""
                    })
                st.dataframe(pd.DataFrame(tier_rows), width="stretch")

                # Tier Cost Comparison Bar Chart
                st.subheader("📊 Total Annual Cost Comparison by Tier")
                df_bar = pd.DataFrame({
                    "Tier": [f"Tier {t.tier_id} (${t.unit_price:.2f})" for t in result.tiers],
                    "Total Annual Cost ($)": [t.total_annual_cost for t in result.tiers]
                }).set_index("Tier")
                st.bar_chart(df_bar)

                # Derivation & Verification
                tab1, tab2 = st.tabs(["✅ Verification Checks", "📋 Assumptions"])
                with tab1:
                    check_data = [
                        {"Check": c.name, "Status": "PASS" if c.passed else "FAIL", "Details": c.message}
                        for c in result.checks
                    ]
                    st.table(pd.DataFrame(check_data))
                with tab2:
                    for a in result.assumptions:
                        st.markdown(f"- {a}")

        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")


# =====================================================================
# Model 3: Safety Stock and Reorder Point
# =====================================================================

elif model == "Safety Stock and Reorder Point":
    st.header("🛡️ Safety Stock & Reorder Point (Continuous Review)")
    st.write(
        "Determine the Reorder Point ($ROP = d \\cdot L + SS$) and buffer safety stock to protect against stockouts "
        "across all **4 demand and lead-time uncertainty regimes**."
    )

    col1, col2, col3 = st.columns([1.1, 1.1, 1.0])

    with col1:
        st.subheader("1. Demand Dynamics")
        time_unit = st.selectbox("Planning Time Unit", ["days", "weeks", "months"], index=0)
        unit_singular = time_unit[:-1]
        mean_demand = st.number_input(
            f"Average Demand per {unit_singular} (d)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Expected mean demand per period."
        )
        std_demand = st.number_input(
            f"Demand Std Dev per {unit_singular} (σ_d)",
            min_value=0.0,
            value=20.0,
            step=1.0,
            help="Set to 0 if demand is constant/deterministic."
        )

    with col2:
        st.subheader("2. Lead-Time Dynamics")
        lead_time = st.number_input(
            f"Average Lead Time ({time_unit}) (L)",
            min_value=0.0,
            value=4.0,
            step=0.5,
            help="Replenishment lead time duration."
        )
        std_lead_time = st.number_input(
            f"Lead-Time Std Dev ({time_unit}) (σ_L)",
            min_value=0.0,
            value=1.0,
            step=0.2,
            help="Set to 0 if lead time is constant/deterministic."
        )

    with col3:
        st.subheader("3. Service Level Target")
        service_level = st.slider(
            "Cycle Service Level (CSL)",
            min_value=0.50,
            max_value=0.999,
            value=0.95,
            step=0.005,
            format="%.3f",
            help="Probability of zero stockouts during a replenishment cycle."
        )
        order_quantity = st.number_input(
            "Order Quantity Q (optional, for Fill Rate)",
            min_value=0.0,
            value=500.0,
            step=50.0,
            help="Batch size Q to evaluate Type 2 Service Level (Fill Rate)."
        )

    # Dynamic Uncertainty Regime Preview Badge
    if std_demand == 0.0 and std_lead_time == 0.0:
        regime_label = "Regime 1: Constant Demand & Constant Lead Time (Zero Safety Stock)"
    elif std_demand > 0.0 and std_lead_time == 0.0:
        regime_label = "Regime 2: Variable Demand & Constant Lead Time (σ_LT = σ_d * √L)"
    elif std_demand == 0.0 and std_lead_time > 0.0:
        regime_label = "Regime 3: Constant Demand & Variable Lead Time (σ_LT = d * σ_L)"
    else:
        regime_label = "Regime 4: Variable Demand & Variable Lead Time (σ_LT = √(L*σ_d² + d²*σ_L²))"

    st.info(f"**Detected Uncertainty Scenario:** {regime_label}")

    if st.button("Calculate Safety Stock & ROP", type="primary", width="stretch"):
        try:
            result = calculate_rop(
                mean_demand=mean_demand,
                lead_time=lead_time,
                service_level=service_level,
                std_demand=std_demand,
                std_lead_time=std_lead_time,
                time_unit=time_unit,
                order_quantity=order_quantity if order_quantity > 0 else None
            )

            st.markdown(
                f'<div class="rec-box">'
                f'<h4>🎯 Executive Recommendation</h4>'
                f'Set the <b>Reorder Point (ROP) to {result.reorder_point:,.1f} units</b> '
                f'(incorporating <b>{result.safety_stock:,.1f} units of Safety Stock</b>) '
                f'to achieve a <b>{result.service_level*100:.1f}% Cycle Service Level</b>. '
                f'Reorder immediately when inventory drops to this threshold.'
                f'</div>',
                unsafe_allow_html=True
            )

            # Key Metrics
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Reorder Point (ROP)", f"{result.reorder_point:,.1f} units", f"{round(result.reorder_point)} rounded")
            r2.metric("Safety Stock Buffer", f"{result.safety_stock:,.1f} units")
            r3.metric("Expected Lead-Time Demand", f"{result.lead_time_demand_mean:,.1f} units", f"d * L")
            r4.metric("Lead-Time Demand Std Dev", f"{result.lead_time_demand_std:,.2f} units", f"σ_LT")

            r5, r6, r7, r8 = st.columns(4)
            r5.metric("Target Service Level (CSL)", f"{result.service_level*100:.1f}%", f"Z = {result.z_score:.4f}")
            r6.metric("Stockout Risk per Cycle", f"{result.stockout_risk:.1f}%")
            if result.fill_rate is not None:
                r7.metric("Item Fill Rate (Type 2 SL)", f"{result.fill_rate*100:.2f}%")
                r8.metric("Exp. Shortage per Cycle", f"{result.expected_shortage_per_cycle:,.2f} units")
            else:
                r7.metric("Uncertainty Type", result.uncertainty_type)
                r8.metric("Safety Stock Coverage", f"{(result.safety_stock / (result.mean_demand_per_period or 1)):.1f} {time_unit}")

            # Normal Distribution Density Chart
            if result.lead_time_demand_std > 0:
                st.subheader("📊 Lead-Time Demand Distribution & Safety Stock Buffer")
                mu_ltd = result.lead_time_demand_mean
                sig_ltd = result.lead_time_demand_std
                x_vals = np.linspace(max(0, mu_ltd - 3.5 * sig_ltd), mu_ltd + 3.5 * sig_ltd, 100)
                pdf_vals = [normal_pdf((x - mu_ltd) / sig_ltd) / sig_ltd for x in x_vals]
                df_dist = pd.DataFrame({
                    "Lead Time Demand": x_vals,
                    "Probability Density": pdf_vals
                }).set_index("Lead Time Demand")
                st.line_chart(df_dist)
                st.caption(f"Mean Demand during Lead Time = {mu_ltd:,.1f} | Reorder Point = {result.reorder_point:,.1f} (Z = {result.z_score:.2f})")

            # Mathematical Details
            tab1, tab2, tab3 = st.tabs(["📐 Mathematical Derivation", "✅ Verification Audit", "📋 Model Assumptions"])
            with tab1:
                st.code(
                    f"1. Expected Lead-Time Demand:\n"
                    f"   mu_LTD = d * L = {result.mean_demand_per_period:,.1f} * {result.mean_lead_time:.1f} = {result.lead_time_demand_mean:,.1f} units\n\n"
                    f"2. Lead-Time Demand Uncertainty ({result.uncertainty_type}):\n"
                    f"   sigma_LTD = {result.lead_time_demand_std:,.4f} units\n\n"
                    f"3. Safety Stock (SS):\n"
                    f"   Z for {result.service_level*100:.1f}% Service Level = {result.z_score:.4f}\n"
                    f"   SS = Z * sigma_LTD = {result.z_score:.4f} * {result.lead_time_demand_std:,.4f} = {result.safety_stock:,.2f} units\n\n"
                    f"4. Reorder Point (ROP):\n"
                    f"   ROP = mu_LTD + SS = {result.lead_time_demand_mean:,.1f} + {result.safety_stock:,.2f} = {result.reorder_point:,.2f} units",
                    language="text"
                )
            with tab2:
                check_data = [
                    {"Check": c.name, "Status": "PASS" if c.passed else "FAIL", "Details": c.message}
                    for c in result.checks
                ]
                st.table(pd.DataFrame(check_data))
            with tab3:
                for a in result.assumptions:
                    st.markdown(f"- {a}")

        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")


# =====================================================================
# Model 4: Fixed-Period Review
# =====================================================================

elif model == "Fixed-Period Review":
    st.header("📅 Fixed-Period Review Policy (Order-Up-To Level / (R, S))")
    st.write(
        "For periodic inventory reviews where stock is counted every $R$ periods. "
        "The system protects across the **protection horizon ($R + L$)** and sets a target base-stock level ($S$). "
        "The recommended order quantity is $Q = \\max(0, S - IP)$."
    )

    col1, col2, col3 = st.columns([1.1, 1.1, 1.1])

    with col1:
        st.subheader("1. Review & Lead Time")
        time_unit = st.selectbox("Review Time Unit", ["weeks", "days", "months"], index=0)
        review_period = st.number_input(
            f"Review Period Interval (R, {time_unit})",
            min_value=0.1,
            value=2.0,
            step=0.5,
            help="Interval between scheduled periodic reviews (e.g. reviewed every 2 weeks)."
        )
        lead_time = st.number_input(
            f"Delivery Lead Time (L, {time_unit})",
            min_value=0.0,
            value=1.0,
            step=0.5,
            help="Supplier replenishment lead time."
        )

    with col2:
        st.subheader("2. Demand Dynamics")
        unit_sing = time_unit[:-1]
        mean_demand = st.number_input(
            f"Mean Demand per {unit_sing} (d)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Expected average demand per periodic time unit."
        )
        std_demand = st.number_input(
            f"Demand Std Dev per {unit_sing} (σ_d)",
            min_value=0.0,
            value=20.0,
            step=1.0,
            help="Standard deviation of demand per period."
        )
        ss_mode = st.radio("Safety Stock Calculation", ["From Service Level (CSL %)", "Direct Safety Stock Entry"])
        service_level = None
        direct_ss = None
        if ss_mode == "From Service Level (CSL %)":
            service_level = st.slider("Target Service Level (CSL)", 0.50, 0.999, 0.95, 0.005)
        else:
            direct_ss = st.number_input("Direct Safety Stock Units (SS)", min_value=0.0, value=40.0, step=5.0)

    with col3:
        st.subheader("3. Current Inventory Position (IP)")
        ip_mode = st.radio("Inventory Position Entry", ["Component Breakdown (OH + OO - BO)", "Direct Inventory Position (IP)"])
        on_hand = 0.0
        on_order = 0.0
        backorders = 0.0
        inventory_position = None

        if ip_mode == "Component Breakdown (OH + OO - BO)":
            on_hand = st.number_input("On-Hand Physical Stock (OH)", min_value=0.0, value=80.0, step=10.0)
            on_order = st.number_input("On-Order / Pipeline Stock (OO)", min_value=0.0, value=50.0, step=10.0)
            backorders = st.number_input("Backorders / Unfilled Demand (BO)", min_value=0.0, value=10.0, step=5.0)
        else:
            inventory_position = st.number_input("Net Inventory Position (IP)", min_value=-500.0, value=120.0, step=10.0)

    if st.button("Calculate Periodic Review Order", type="primary", width="stretch"):
        try:
            result = calculate_periodic_review(
                review_period=review_period,
                lead_time=lead_time,
                mean_demand_per_period=mean_demand,
                std_demand_per_period=std_demand,
                service_level=service_level,
                safety_stock=direct_ss,
                time_unit=time_unit,
                on_hand=on_hand,
                on_order=on_order,
                backorders=backorders,
                inventory_position=inventory_position
            )

            st.markdown(
                f'<div class="rec-box">'
                f'<h4>🎯 Executive Recommendation</h4>'
                f'Set Target Order-Up-To Level (S) to <b>{result.target_inventory_level:,.1f} units</b>. '
                f'<b>Order {result.recommended_order_quantity:,.0f} units today</b> '
                f'(Current IP = {result.inventory_position:,.0f} units).'
                f'</div>',
                unsafe_allow_html=True
            )

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Order-Up-To Target (S)", f"{result.target_inventory_level:,.1f} units", "Base-stock level")
            p2.metric("Recommended Order (Q)", f"{result.recommended_order_quantity:,.0f} units", f"max(0, S - IP)")
            p3.metric("Current Inventory Position", f"{result.inventory_position:,.1f} units", "OH + OO - BO")
            p4.metric("Safety Stock Buffer", f"{result.safety_stock:,.1f} units")

            p5, p6, p7, p8 = st.columns(4)
            p5.metric("Protection Horizon", f"{result.protection_interval:.1f} {time_unit}", f"R ({review_period}) + L ({lead_time})")
            p6.metric("Protection Demand Mean", f"{result.protection_demand_mean:,.1f} units")
            p7.metric("Protection Demand Std Dev", f"{result.protection_demand_std:,.2f} units")
            p8.metric("Periodic SS Multiplier", f"{result.periodic_ss_multiplier:.2f}x", f"vs Continuous Review SS ({result.continuous_ss_equivalent:,.1f})")

            # Inventory Breakdown Chart
            st.subheader("📊 Inventory Position vs. Target Level S")
            df_ip = pd.DataFrame({
                "Component": ["On-Hand", "On-Order", "Backorders", "Net IP", "Order Quantity (Q)", "Target Level (S)"],
                "Units": [result.on_hand, result.on_order, -result.backorders, result.inventory_position, result.recommended_order_quantity, result.target_inventory_level]
            }).set_index("Component")
            st.bar_chart(df_ip)

            # Technical Details
            tab1, tab2, tab3 = st.tabs(["📐 Mathematical Derivation", "✅ Verification Audit", "📋 Model Assumptions"])
            with tab1:
                st.code(
                    f"Periodic Review (R, S) Formulation:\n"
                    f"  Protection Interval = Review Period ({result.review_period}) + Lead Time ({result.lead_time}) = {result.protection_interval:.1f} {result.time_unit}\n"
                    f"  Expected Demand during Protection Interval = d * (R + L) = {result.mean_demand_per_period:,.1f} * {result.protection_interval:.1f} = {result.protection_demand_mean:,.1f} units\n"
                    f"  Protection Std Dev = sigma_d * sqrt(R + L) = {result.std_demand_per_period:,.2f} * sqrt({result.protection_interval:.1f}) = {result.protection_demand_std:,.2f} units\n"
                    f"  Safety Stock = Z * sigma_prot = {result.safety_stock:,.2f} units\n"
                    f"  Target Base-Stock Level S = mu_prot + SS = {result.protection_demand_mean:,.1f} + {result.safety_stock:,.2f} = {result.target_inventory_level:,.2f} units\n"
                    f"  Current Inventory Position IP = OH + OO - BO = {result.inventory_position:,.1f} units\n"
                    f"  Order Quantity Q = max(0, S - IP) = max(0, {result.target_inventory_level:,.1f} - {result.inventory_position:,.1f}) = {result.recommended_order_quantity:,.2f} units",
                    language="text"
                )
            with tab2:
                check_data = [
                    {"Check": c.name, "Status": "PASS" if c.passed else "FAIL", "Details": c.message}
                    for c in result.checks
                ]
                st.table(pd.DataFrame(check_data))
            with tab3:
                for a in result.assumptions:
                    st.markdown(f"- {a}")

        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")


# =====================================================================
# Model 5: Newsvendor Model
# =====================================================================

elif model == "Newsvendor Model":
    st.header("📰 Newsvendor Model (Single-Period Inventory Decision)")
    st.write(
        "Optimize single-period commitments for perishable, seasonal, or fashion items. "
        "Balances the cost of underage ($C_u$, lost margin) against the cost of overage ($C_o$, disposal loss) "
        "via the **Critical Fractile (CR)**: $P(D \\le Q^*) = \\frac{C_u}{C_u + C_o}$."
    )

    col1, col2 = st.columns([1.1, 1.1])

    with col1:
        st.subheader("1. Cost Economics")
        nv_cost_mode = st.radio(
            "Cost Input Method",
            ["Retail & Procurement Economics (P, C, V, g)", "Direct Marginal Costs (Cu, Co)"]
        )

        selling_price = None
        purchase_cost = None
        salvage_value = 0.0
        shortage_penalty = 0.0
        cost_of_underage = None
        cost_of_overage = None

        if nv_cost_mode == "Retail & Procurement Economics (P, C, V, g)":
            selling_price = st.number_input("Selling / Retail Price (P, $)", min_value=0.01, value=150.0, step=10.0)
            purchase_cost = st.number_input("Procurement / Purchase Cost (C, $)", min_value=0.01, value=60.0, step=5.0)
            salvage_value = st.number_input("Salvage / Clearance Value (V, $)", min_value=0.0, value=20.0, step=5.0)
            shortage_penalty = st.number_input("Shortage / Goodwill Penalty (g, $)", min_value=0.0, value=0.0, step=5.0)
        else:
            cost_of_underage = st.number_input("Cost of Underage (Cu, $)", min_value=0.01, value=90.0, step=5.0, help="Unit profit margin lost per unmet demand.")
            cost_of_overage = st.number_input("Cost of Overage (Co, $)", min_value=0.01, value=40.0, step=5.0, help="Loss incurred on each leftover unsold unit.")

    with col2:
        st.subheader("2. Demand Distribution")
        dist_choice = st.selectbox("Distribution Model", ["Normal Distribution", "Uniform Distribution"], index=0)

        mean_demand = None
        std_demand = None
        uniform_a = None
        uniform_b = None
        dist_key = "normal"

        if dist_choice == "Normal Distribution":
            dist_key = "normal"
            mean_demand = st.number_input("Expected Mean Demand (μ)", min_value=1.0, value=500.0, step=25.0)
            std_demand = st.number_input("Demand Standard Deviation (σ)", min_value=0.1, value=100.0, step=10.0)
        else:
            dist_key = "uniform"
            uniform_a = st.number_input("Minimum Demand (a)", min_value=0.0, value=200.0, step=25.0)
            uniform_b = st.number_input("Maximum Demand (b)", min_value=1.0, value=800.0, step=25.0)

    if st.button("Calculate Newsvendor Order Quantity", type="primary", width="stretch"):
        try:
            result = calculate_newsvendor(
                selling_price=selling_price,
                purchase_cost=purchase_cost,
                salvage_value=salvage_value,
                shortage_penalty=shortage_penalty,
                cost_of_underage=cost_of_underage,
                cost_of_overage=cost_of_overage,
                mean_demand=mean_demand,
                std_demand=std_demand,
                demand_distribution=dist_key,
                uniform_a=uniform_a,
                uniform_b=uniform_b
            )

            st.markdown(
                f'<div class="rec-box">'
                f'<h4>🎯 Executive Recommendation</h4>'
                f'<b>Order {result.optimal_order_quantity:,.0f} units</b> '
                f'(Critical Fractile: <b>{result.critical_fractile*100:.1f}%</b>). '
                f'Generates an expected single-period profit of <b>${result.expected_profit:,.2f}</b>.'
                f'</div>',
                unsafe_allow_html=True
            )

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Optimal Order Quantity (Q*)", f"{result.optimal_order_quantity:,.0f} units", f"{result.optimal_order_quantity:,.1f}")
            n2.metric("Critical Fractile (CR)", f"{result.critical_fractile*100:.2f}%", "Target in-stock rate")
            n3.metric("Cost of Underage (Cu)", f"${result.cost_of_underage:.2f}", "Lost margin")
            n4.metric("Cost of Overage (Co)", f"${result.cost_of_overage:.2f}", "Disposal loss")

            n5, n6, n7, n8 = st.columns(4)
            n5.metric("Expected Total Profit", f"${result.expected_profit:,.2f}")
            n6.metric("Expected Units Sold", f"{result.expected_sales:,.1f} units")
            n7.metric("Expected Leftover Units", f"{result.expected_leftover:,.1f} units")
            n8.metric("Expected EVPI", f"${result.evpi:,.2f}", "Value of perfect info")

            # Distribution Curve
            if dist_key == "normal" and result.std_demand is not None:
                st.subheader("📊 Demand Distribution & Critical Fractile Cutoff")
                mu = result.mean_demand
                sig = result.std_demand
                q_opt = result.optimal_order_quantity
                x_vals = np.linspace(max(0, mu - 3.5 * sig), mu + 3.5 * sig, 100)
                pdf_vals = [normal_pdf((x - mu) / sig) / sig for x in x_vals]
                cdf_vals = [normal_cdf((x - mu) / sig) for x in x_vals]
                df_nv = pd.DataFrame({
                    "Demand": x_vals,
                    "Demand PDF": pdf_vals,
                    "Cumulative CDF": cdf_vals
                }).set_index("Demand")
                st.line_chart(df_nv)
                st.caption(f"Mean Demand = {mu:,.0f} | Q* = {q_opt:,.0f} (CDF = {result.critical_fractile*100:.1f}%)")

            # Technical Details
            tab1, tab2, tab3 = st.tabs(["📐 Mathematical Derivation", "✅ Verification Audit", "📋 Model Assumptions"])
            with tab1:
                st.code(
                    f"Newsvendor Formulation:\n"
                    f"  Cost of Underage (Cu) = ${result.cost_of_underage:.2f}\n"
                    f"  Cost of Overage  (Co) = ${result.cost_of_overage:.2f}\n"
                    f"  Critical Ratio   (CR) = Cu / (Cu + Co) = ${result.cost_of_underage:.2f} / (${result.cost_of_underage:.2f} + ${result.cost_of_overage:.2f}) = {result.critical_fractile:.4f}\n"
                    f"  Z-Score (Norm Inv)    = {result.z_score:.4f} (if normal distribution)\n"
                    f"  Optimal Quantity (Q*) = Mean + Z * StdDev = {result.optimal_order_quantity:,.2f} units\n\n"
                    f"Expected Profit Performance:\n"
                    f"  Expected Revenue     = ${result.expected_revenue:,.2f}\n"
                    f"  Expected Cost        = ${result.expected_purchase_cost:,.2f}\n"
                    f"  Expected Net Profit  = ${result.expected_profit:,.2f}\n"
                    f"  Profit w/ Perfect Info = ${result.profit_under_perfect_information:,.2f}\n"
                    f"  EVPI                 = ${result.evpi:,.2f}",
                    language="text"
                )
            with tab2:
                check_data = [
                    {"Check": c.name, "Status": "PASS" if c.passed else "FAIL", "Details": c.message}
                    for c in result.checks
                ]
                st.table(pd.DataFrame(check_data))
            with tab3:
                for a in result.assumptions:
                    st.markdown(f"- {a}")

        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")


# =====================================================================
# Model 6: 🤖 AI Natural Language Decision Agent
# =====================================================================

elif model == "🤖 AI Natural Language Decision Agent":
    st.header("🤖 AI Natural Language Inventory Decision Agent")
    st.write(
        "Ask inventory questions in plain English. The agent automatically executes the 7-stage pipeline: "
        "**Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain**."
    )

    # Preset Sample Queries
    st.markdown("##### 💡 Or select a pre-loaded industry query:")
    preset_cols = st.columns(5)
    sample_q = ""

    if preset_cols[0].button("EOQ Inquiry", width="stretch"):
        st.session_state["query_input"] = (
            "We sell 12,000 units of widgets per year. Ordering cost is $50, unit purchase price is $10, "
            "and annual holding rate is 20%. What is our EOQ and total cost?"
        )
    if preset_cols[1].button("Quantity Discount", width="stretch"):
        st.session_state["query_input"] = (
            "Annual demand of 5,000 units with cost per order of $50. Percentage holding cost is 20%. "
            "Price schedule: 1-499 units at unit price $20, 500-999 units at $18, 1000+ units at $16."
        )
    if preset_cols[2].button("Safety Stock & ROP", width="stretch"):
        st.session_state["query_input"] = (
            "Average demand is 100 with a standard deviation of 20. Lead time averages 4 days with a standard deviation of 1 day. "
            "Target service level is 95%."
        )
    if preset_cols[3].button("Periodic Review", width="stretch"):
        st.session_state["query_input"] = (
            "Review period is 2 weeks. Supplier lead time is 1 week. Weekly demand averages 100 units with std dev 20. "
            "On-hand inventory is 80 units, on-order inventory is 50 units, and backorders are 10 units. Service level is 95%."
        )
    if preset_cols[4].button("Newsvendor", width="stretch"):
        st.session_state["query_input"] = (
            "A fashion retailer sells ski jackets for selling price $150, purchase cost $60, and salvage value $20. "
            "Forecast demand is normal with mean 500 jackets and demand standard deviation of 100."
        )

    user_query = st.text_area(
        "Customer / Manager Question:",
        value=st.session_state.get("query_input", "We sell 12,000 units of widgets per year. Ordering cost is $50, unit cost is $10, and annual holding rate is 20%. What is our EOQ?"),
        height=100
    )

    if st.button("Submit to Decision Agent", type="primary", width="stretch"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing natural language through 7-stage supply chain pipeline..."):
                agent = InventoryDecisionAgent()
                response: AgentResponse = agent.process_question(user_query)

                # Pipeline Stage Trace
                st.subheader("1. Pipeline Trace & Model Identification")
                trace_col1, trace_col2 = st.columns([1.5, 1.0])
                with trace_col1:
                    st.success(f"**Identified Model:** {response.model_identification.selected_model.value}")
                    st.caption(f"Confidence: {response.model_identification.confidence*100:.0f}% | {response.model_identification.reasoning}")
                with trace_col2:
                    st.info(f"**Selected Tool:** `{response.tool_execution.tool_name}`")

                with st.expander("Detailed Pipeline Execution Log", expanded=False):
                    for step in response.execution_pipeline_trace:
                        st.text(step)

                # Extracted Inputs
                st.subheader("2. Extracted Numerical Inputs")
                ext_col1, ext_col2 = st.columns(2)
                with ext_col1:
                    st.write("**Extracted Parameters:**")
                    st.json(response.input_extraction.extracted_params)
                with ext_col2:
                    if response.input_extraction.missing_required_params:
                        st.error(f"**Missing Parameters:** {response.input_extraction.missing_required_params}")
                    else:
                        st.success("All required parameters successfully extracted!")
                    if response.input_extraction.assumptions_applied:
                        st.write("**Applied Assumptions:**")
                        for a in response.input_extraction.assumptions_applied:
                            st.caption(f"• {a}")

                # Executive Recommendation
                st.subheader("3. Executive Recommendation")
                st.markdown(
                    f'<div class="rec-box">'
                    f'<h4>🎯 {response.explanation.headline_recommendation}</h4>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # KPIs
                st.subheader("4. Key Performance Indicators")
                kpi_cols = st.columns(len(response.explanation.numeric_breakdown) if len(response.explanation.numeric_breakdown) <= 4 else 4)
                for idx, (k, v) in enumerate(response.explanation.numeric_breakdown.items()):
                    col_idx = idx % len(kpi_cols)
                    kpi_cols[col_idx].metric(k, str(v))

                # Mathematical Derivation
                st.subheader("5. Mathematical Formulation & Derivation")
                st.code(response.explanation.mathematical_derivation, language="text")

                # Verification & Insights
                tab_v, tab_i, tab_r = st.tabs(["✅ Verification Audit", "💡 Managerial Insights", "📄 Complete Decision Report"])
                with tab_v:
                    for v_line in response.explanation.verification_audit:
                        st.markdown(f"- {v_line}")
                with tab_i:
                    for ins in response.explanation.managerial_insights:
                        st.markdown(f"- {ins}")
                with tab_r:
                    st.markdown(response.explanation.formatted_report)
