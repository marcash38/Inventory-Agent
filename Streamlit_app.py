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

    if st.button("Calculate EOQ", type="primary", use_container_width=True):
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
    edited_tiers = st.data_editor(pd.DataFrame(default_tiers), num_rows="dynamic", use_container_width=True)

    if st.button("Calculate Quantity Discount", type="primary", use_container_width=True):
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
                st.dataframe(pd.DataFrame(tier_rows), use_container_width=True)

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

    if st.button("Calculate Safety Stock & ROP", type="primary", use_container_width=True):
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

    if st.button("Calculate Periodic Review Order", type="primary", use_container_width=True):
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

    if st.button("Calculate Newsvendor Order Quantity", type="primary", use_container_width=True):
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

    if preset_cols[0].button("EOQ Inquiry", use_container_width=True):
        st.session_state["query_input"] = (
            "We sell 12,000 units of widgets per year. Ordering cost is $50, unit purchase price is $10, "
            "and annual holding rate is 20%. What is our EOQ and total cost?"
        )
    if preset_cols[1].button("Quantity Discount", use_container_width=True):
        st.session_state["query_input"] = (
            "Annual demand of 5,000 units with cost per order of $50. Percentage holding cost is 20%. "
            "Price schedule: 1-499 units at unit price $20, 500-999 units at $18, 1000+ units at $16."
        )
    if preset_cols[2].button("Safety Stock & ROP", use_container_width=True):
        st.session_state["query_input"] = (
            "Average demand is 100 with a standard deviation of 20. Lead time averages 4 days with a standard deviation of 1 day. "
            "Target service level is 95%."
        )
    if preset_cols[3].button("Periodic Review", use_container_width=True):
        st.session_state["query_input"] = (
            "Review period is 2 weeks. Supplier lead time is 1 week. Weekly demand averages 100 units with std dev 20. "
            "On-hand inventory is 80 units, on-order inventory is 50 units, and backorders are 10 units. Service level is 95%."
        )
    if preset_cols[4].button("Newsvendor", use_container_width=True):
        st.session_state["query_input"] = (
            "A fashion retailer sells ski jackets for selling price $150, purchase cost $60, and salvage value $20. "
            "Forecast demand is normal with mean 500 jackets and demand standard deviation of 100."
        )

    user_query = st.text_area(
        "Customer / Manager Question:",
        value=st.session_state.get("query_input", "We sell 12,000 units of widgets per year. Ordering cost is $50, unit cost is $10, and annual holding rate is 20%. What is our EOQ?"),
        height=100
    )

    if st.button("Submit to Decision Agent", type="primary", use_container_width=True):
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
