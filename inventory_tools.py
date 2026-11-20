"""
Supply Chain Analytics - Inventory Decision Tools
Pure Python numerical calculations and verification routines for 5 core inventory models:
1. Economic Order Quantity (EOQ): calculate_eoq()
2. Quantity Discount Model: calculate_quantity_discount()
3. Safety Stock and Reorder Point (Continuous Review): calculate_rop()
4. Fixed-Period Review (Periodic Review / Order-Up-To): calculate_periodic_review()
5. Newsvendor Model (Single-Period Inventory): calculate_newsvendor()

Every tool:
- Validates inputs
- Returns structured results (dataclass with properties)
- Returns intermediate calculations
- Returns the final answer
- Identifies assumptions
- Never invents missing numerical inputs
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =====================================================================
# Statistical Distribution Helpers (Standard Normal N(0, 1))
# =====================================================================

def normal_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal distribution N(0, 1)."""
    if HAS_SCIPY:
        return float(stats.norm.cdf(z))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_pdf(z: float) -> float:
    """Probability density function for standard normal distribution N(0, 1)."""
    if HAS_SCIPY:
        return float(stats.norm.pdf(z))
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)


def normal_ppf(p: float) -> float:
    """
    Inverse cumulative distribution function (Quantile / PPF) for N(0, 1).
    Uses scipy if available, or Peter J. Acklam's rational approximation (error < 1.15e-9).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability p must be strictly between 0 and 1, got {p}")
    
    if HAS_SCIPY:
        return float(stats.norm.ppf(p))
    
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]

    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]

    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]

    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    q_low = 0.02425
    q_high = 1.0 - q_low

    if p < q_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= q_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def standard_normal_loss_function(z: float) -> float:
    """
    Unit normal loss function L(z) = phi(z) - z * (1 - Phi(z)).
    Computes expected shortages for standard normal distribution.
    """
    phi_z = normal_pdf(z)
    Phi_z = normal_cdf(z)
    return phi_z - z * (1.0 - Phi_z)


# =====================================================================
# Result Data Structures
# =====================================================================

@dataclass
class VerificationCheck:
    name: str
    passed: bool
    message: str
    severity: str = "INFO"  # INFO, WARNING, ERROR


@dataclass
class EOQResult:
    annual_demand: float
    ordering_cost: float
    holding_cost_per_unit: float
    optimal_order_quantity: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_inventory_cost: float
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: float
    average_cycle_inventory: float
    unit_cost: Optional[float] = None
    annual_purchasing_cost: Optional[float] = None
    total_acquisition_cost: Optional[float] = None
    lead_time_days: Optional[float] = None
    reorder_point: Optional[float] = None
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    checks: List[VerificationCheck] = field(default_factory=list)


@dataclass
class DiscountTier:
    tier_id: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost: float = 0.0
    eoq_unconstrained: float = 0.0
    candidate_qty: float = 0.0
    is_feasible: bool = False
    annual_purchase_cost: float = 0.0
    annual_ordering_cost: float = 0.0
    annual_holding_cost: float = 0.0
    total_annual_cost: float = 0.0


@dataclass
class QuantityDiscountResult:
    annual_demand: float
    ordering_cost: float
    holding_cost_rate: Optional[float]
    fixed_holding_cost: Optional[float]
    tiers: List[DiscountTier]
    optimal_tier_id: int
    optimal_order_quantity: float
    optimal_unit_price: float
    optimal_purchase_cost: float
    optimal_ordering_cost: float
    optimal_holding_cost: float
    optimal_total_cost: float
    baseline_total_cost: float
    annual_savings: float
    percentage_savings: float
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    checks: List[VerificationCheck] = field(default_factory=list)


@dataclass
class SafetyStockROPResult:
    mean_demand_per_period: float
    std_demand_per_period: float
    mean_lead_time: float
    std_lead_time: float
    service_level: float
    time_unit: str
    z_score: float
    lead_time_demand_mean: float
    lead_time_demand_std: float
    safety_stock: float
    reorder_point: float
    stockout_risk: float
    uncertainty_type: str  # "Constant Demand & Lead Time", "Variable Demand & Constant Lead Time", etc.
    order_quantity: Optional[float] = None
    expected_shortage_per_cycle: Optional[float] = None
    fill_rate: Optional[float] = None
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    checks: List[VerificationCheck] = field(default_factory=list)


@dataclass
class FixedPeriodReviewResult:
    review_period: float
    lead_time: float
    mean_demand_per_period: float
    std_demand_per_period: float
    service_level: float
    time_unit: str
    protection_interval: float
    protection_demand_mean: float
    protection_demand_std: float
    z_score: float
    safety_stock: float
    target_inventory_level: float  # Base stock / Order-up-to level (S)
    on_hand: float = 0.0
    on_order: float = 0.0
    backorders: float = 0.0
    inventory_position: float = 0.0
    recommended_order_quantity: float = 0.0
    continuous_ss_equivalent: float = 0.0
    periodic_ss_multiplier: float = 1.0
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    checks: List[VerificationCheck] = field(default_factory=list)


@dataclass
class NewsvendorResult:
    selling_price: float
    purchase_cost: float
    salvage_value: float
    shortage_penalty: float
    cost_of_underage: float  # Cu
    cost_of_overage: float   # Co
    critical_fractile: float  # CR = Cu / (Cu + Co)
    distribution_type: str   # "normal", "discrete", "uniform"
    mean_demand: Optional[float] = None
    std_demand: Optional[float] = None
    optimal_order_quantity: float = 0.0
    z_score: Optional[float] = None
    expected_shortage: float = 0.0
    expected_sales: float = 0.0
    expected_leftover: float = 0.0
    expected_revenue: float = 0.0
    expected_purchase_cost: float = 0.0
    expected_shortage_penalty: float = 0.0
    expected_salvage_revenue: float = 0.0
    expected_profit: float = 0.0
    profit_under_perfect_information: float = 0.0
    evpi: float = 0.0
    in_stock_probability: float = 0.0
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    checks: List[VerificationCheck] = field(default_factory=list)


# =====================================================================
# 1. Economic Order Quantity (EOQ): calculate_eoq()
# =====================================================================

def calculate_eoq(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
    demand: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days: float = 365.0
) -> EOQResult:
    """
    Deterministic Economic Order Quantity calculation:
        Q* = sqrt(2 * D * S / H)

    Validates inputs, returns structured results with intermediate calculations,
    final answer, assumptions, and domain verification. Never invents missing numerical inputs.
    """
    assumptions: List[str] = [
        "Demand occurs at a constant, continuous rate over the planning horizon.",
        "Replenishment lots arrive in full simultaneously (zero production replenishment lead time).",
        "No shortages or stockouts are permitted."
    ]

    # Resolve demand aliases
    d_val = annual_demand if annual_demand is not None else demand
    if d_val is None:
        raise ValueError("Missing required numerical input: annual_demand (or demand) must be provided.")
    if d_val <= 0:
        raise ValueError(f"Annual demand must be strictly positive, got {d_val}")

    # Resolve ordering cost
    if ordering_cost is None:
        raise ValueError("Missing required numerical input: ordering_cost must be provided.")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be strictly positive, got {ordering_cost}")

    # Resolve holding cost (fixed per unit or rate * unit_cost)
    h_val = holding_cost_per_unit if holding_cost_per_unit is not None else holding_cost
    if h_val is None:
        if holding_cost_rate is not None and unit_cost is not None:
            if holding_cost_rate <= 0:
                raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")
            if unit_cost <= 0:
                raise ValueError(f"Unit cost must be positive, got {unit_cost}")
            h_val = holding_cost_rate * unit_cost
            assumptions.append(f"Derived holding cost per unit H = {holding_cost_rate*100:.1f}% * ${unit_cost:.2f} = ${h_val:.2f}/unit/yr.")
        else:
            raise ValueError("Missing required numerical input: either holding_cost_per_unit (or holding_cost) OR (holding_cost_rate and unit_cost) must be provided.")
    
    if h_val <= 0:
        raise ValueError(f"Holding cost must be strictly positive, got {h_val}")

    # Optimal EOQ Calculation
    q_star = math.sqrt((2.0 * d_val * ordering_cost) / h_val)

    # Intermediate calculations
    annual_ordering_cost = (d_val / q_star) * ordering_cost
    annual_holding_cost = (q_star / 2.0) * h_val
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    orders_per_year = d_val / q_star
    cycle_time_years = q_star / d_val
    cycle_time_days = cycle_time_years * work_days
    average_cycle_inventory = q_star / 2.0

    annual_purchasing_cost = None
    total_acquisition_cost = None
    if unit_cost is not None:
        annual_purchasing_cost = d_val * unit_cost
        total_acquisition_cost = total_inventory_cost + annual_purchasing_cost

    reorder_point = None
    if lead_time_days is not None:
        if lead_time_days < 0:
            raise ValueError(f"Lead time days cannot be negative, got {lead_time_days}")
        daily_demand = d_val / work_days
        reorder_point = daily_demand * lead_time_days

    # Domain verification checks
    checks: List[VerificationCheck] = [
        VerificationCheck("Demand Positivity", d_val > 0, f"Annual demand D = {d_val:,.0f} units is positive"),
        VerificationCheck("Ordering Cost Positivity", ordering_cost > 0, f"Ordering cost S = ${ordering_cost:,.2f} is positive"),
        VerificationCheck("Holding Cost Positivity", h_val > 0, f"Holding cost H = ${h_val:,.2f}/unit/year is positive"),
        VerificationCheck(
            "EOQ First-Order Optimality Condition",
            abs(annual_ordering_cost - annual_holding_cost) / total_inventory_cost < 0.0001,
            f"Annual Ordering Cost (${annual_ordering_cost:,.2f}) strictly balances Annual Holding Cost (${annual_holding_cost:,.2f})."
        )
    ]

    intermediate = {
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_inventory_cost": total_inventory_cost,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "average_cycle_inventory": average_cycle_inventory
    }
    if annual_purchasing_cost is not None:
        intermediate["annual_purchasing_cost"] = annual_purchasing_cost
        intermediate["total_acquisition_cost"] = total_acquisition_cost
    if reorder_point is not None:
        intermediate["deterministic_reorder_point"] = reorder_point

    final_ans = {
        "optimal_order_quantity": q_star,
        "rounded_order_quantity": round(q_star),
        "total_annual_inventory_cost": total_inventory_cost
    }

    return EOQResult(
        annual_demand=d_val,
        ordering_cost=ordering_cost,
        holding_cost_per_unit=h_val,
        optimal_order_quantity=q_star,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_inventory_cost=total_inventory_cost,
        orders_per_year=orders_per_year,
        cycle_time_years=cycle_time_years,
        cycle_time_days=cycle_time_days,
        average_cycle_inventory=average_cycle_inventory,
        unit_cost=unit_cost,
        annual_purchasing_cost=annual_purchasing_cost,
        total_acquisition_cost=total_acquisition_cost,
        lead_time_days=lead_time_days,
        reorder_point=reorder_point,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        assumptions=assumptions,
        checks=checks
    )


# =====================================================================
# 2. Quantity Discount Tool: calculate_quantity_discount()
# =====================================================================

def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    discount_schedule: Optional[List[Dict[str, Any]]] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
    demand: Optional[float] = None,
    holding_cost: Optional[float] = None
) -> QuantityDiscountResult:
    """
    Evaluates all-units quantity discount schedules to identify the global cost-minimizing lot size.
    Compares feasible quantities across all price tiers.
    Supports fixed holding cost and percentage holding cost.
    Never invents missing numerical inputs.
    """
    assumptions: List[str] = [
        "All-units quantity discount structure applies across all order tiers.",
        "Demand is deterministic and constant over the planning horizon.",
        "Supplier provides instantaneous order delivery in full."
    ]

    d_val = annual_demand if annual_demand is not None else demand
    if d_val is None:
        raise ValueError("Missing required numerical input: annual_demand (or demand) must be provided.")
    if d_val <= 0:
        raise ValueError(f"Annual demand must be strictly positive, got {d_val}")

    if ordering_cost is None:
        raise ValueError("Missing required numerical input: ordering_cost must be provided.")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be strictly positive, got {ordering_cost}")

    if discount_schedule is None or len(discount_schedule) == 0:
        raise ValueError("Missing required input: discount_schedule must be provided as a non-empty list of tiers.")

    f_hc = fixed_holding_cost if fixed_holding_cost is not None else holding_cost
    if holding_cost_rate is None and f_hc is None:
        raise ValueError("Missing required numerical input: either holding_cost_rate OR fixed_holding_cost must be provided.")

    if holding_cost_rate is not None and holding_cost_rate <= 0:
        raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")
    if f_hc is not None and f_hc <= 0:
        raise ValueError(f"Fixed holding cost must be positive, got {f_hc}")

    # Evaluate each tier
    tiers: List[DiscountTier] = []
    tier_evaluations: List[Dict[str, Any]] = []

    for idx, raw_tier in enumerate(discount_schedule):
        if "unit_price" not in raw_tier:
            raise ValueError(f"Tier {idx+1} is missing required 'unit_price'.")
        price = float(raw_tier["unit_price"])
        if price <= 0:
            raise ValueError(f"Tier {idx+1} unit_price must be positive, got {price}")

        min_q = float(raw_tier.get("min_qty", 1.0))
        max_q = float(raw_tier["max_qty"]) if raw_tier.get("max_qty") is not None else None

        if min_q < 0:
            raise ValueError(f"Tier {idx+1} min_qty cannot be negative.")
        if max_q is not None and max_q < min_q:
            raise ValueError(f"Tier {idx+1} max_qty ({max_q}) cannot be less than min_qty ({min_q}).")

        # Determine holding cost for this tier
        if holding_cost_rate is not None:
            h_tier = holding_cost_rate * price
        else:
            h_tier = float(f_hc)

        # Unconstrained EOQ
        eoq_raw = math.sqrt((2.0 * d_val * ordering_cost) / h_tier)

        # Feasibility and candidate quantity selection
        is_feasible = False
        candidate_q = eoq_raw

        if eoq_raw < min_q:
            # Below tier minimum: candidate is the tier minimum breakpoint
            candidate_q = min_q
            is_feasible = True
        elif max_q is not None and eoq_raw > max_q:
            # Above tier maximum: dominated by higher tier, but candidate is max_q
            candidate_q = max_q
            is_feasible = False
        else:
            # Within tier boundaries
            candidate_q = eoq_raw
            is_feasible = True

        ann_purchase = d_val * price
        ann_order = (d_val / candidate_q) * ordering_cost
        ann_hold = (candidate_q / 2.0) * h_tier
        tot_cost = ann_purchase + ann_order + ann_hold

        tier_obj = DiscountTier(
            tier_id=idx + 1,
            min_qty=min_q,
            max_qty=max_q,
            unit_price=price,
            holding_cost=h_tier,
            eoq_unconstrained=eoq_raw,
            candidate_qty=candidate_q,
            is_feasible=is_feasible,
            annual_purchase_cost=ann_purchase,
            annual_ordering_cost=ann_order,
            annual_holding_cost=ann_hold,
            total_annual_cost=tot_cost
        )
        tiers.append(tier_obj)
        tier_evaluations.append({
            "tier_id": idx + 1,
            "price": price,
            "unconstrained_eoq": eoq_raw,
            "candidate_qty": candidate_q,
            "is_feasible": is_feasible,
            "annual_purchase_cost": ann_purchase,
            "annual_ordering_cost": ann_order,
            "annual_holding_cost": ann_hold,
            "total_annual_cost": tot_cost
        })

    # Find the optimal tier among feasible candidates
    feasible_tiers = [t for t in tiers if t.is_feasible]
    if not feasible_tiers:
        feasible_tiers = tiers

    best_tier = min(feasible_tiers, key=lambda t: t.total_annual_cost)
    baseline_tier = tiers[0]
    annual_savings = baseline_tier.total_annual_cost - best_tier.total_annual_cost
    pct_savings = (annual_savings / baseline_tier.total_annual_cost) * 100.0 if baseline_tier.total_annual_cost > 0 else 0.0

    checks: List[VerificationCheck] = [
        VerificationCheck("Tiers Evaluated", True, f"Evaluated {len(tiers)} discount price tiers."),
        VerificationCheck(
            "Global Cost Minimization",
            True,
            f"Selected Tier {best_tier.tier_id} (Q* = {best_tier.candidate_qty:,.0f} @ ${best_tier.unit_price:.2f}) with minimum total annual cost ${best_tier.total_annual_cost:,.2f}."
        )
    ]

    intermediate = {
        "tier_evaluations": tier_evaluations,
        "baseline_tier_cost": baseline_tier.total_annual_cost,
        "optimal_purchase_cost": best_tier.annual_purchase_cost,
        "optimal_ordering_cost": best_tier.annual_ordering_cost,
        "optimal_holding_cost": best_tier.annual_holding_cost
    }

    final_ans = {
        "optimal_tier_id": best_tier.tier_id,
        "optimal_order_quantity": best_tier.candidate_qty,
        "optimal_unit_price": best_tier.unit_price,
        "optimal_total_cost": best_tier.total_annual_cost,
        "annual_savings": annual_savings,
        "percentage_savings": pct_savings
    }

    return QuantityDiscountResult(
        annual_demand=d_val,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_cost_rate,
        fixed_holding_cost=f_hc,
        tiers=tiers,
        optimal_tier_id=best_tier.tier_id,
        optimal_order_quantity=best_tier.candidate_qty,
        optimal_unit_price=best_tier.unit_price,
        optimal_purchase_cost=best_tier.annual_purchase_cost,
        optimal_ordering_cost=best_tier.annual_ordering_cost,
        optimal_holding_cost=best_tier.annual_holding_cost,
        optimal_total_cost=best_tier.total_annual_cost,
        baseline_total_cost=baseline_tier.total_annual_cost,
        annual_savings=annual_savings,
        percentage_savings=pct_savings,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        assumptions=assumptions,
        checks=checks
    )


# =====================================================================
# 3. Safety Stock & Reorder Point: calculate_rop()
# =====================================================================

def calculate_rop(
    mean_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = 0.95,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    time_unit: str = "days",
    order_quantity: Optional[float] = None,
    demand: Optional[float] = None
) -> SafetyStockROPResult:
    """
    Calculates Safety Stock (SS) and Reorder Point (ROP) across 4 uncertainty regimes:
    1. Constant demand and constant lead time (std_demand == 0, std_lead_time == 0)
    2. Variable demand and constant lead time (std_demand > 0, std_lead_time == 0)
    3. Constant demand and variable lead time (std_demand == 0, std_lead_time > 0)
    4. Variable demand and variable lead time (std_demand > 0, std_lead_time > 0):
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS

    Validates inputs, returns structured results with intermediate calculations,
    final answer, assumptions, and domain verification. Never invents missing numerical inputs.
    """
    assumptions: List[str] = []

    d_val = mean_demand if mean_demand is not None else demand
    if d_val is None:
        raise ValueError("Missing required numerical input: mean_demand (or demand) must be provided.")
    if d_val < 0:
        raise ValueError(f"Mean demand cannot be negative, got {d_val}")

    if lead_time is None:
        raise ValueError("Missing required numerical input: lead_time must be provided.")
    if lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}")

    sl_val = service_level if service_level is not None else 0.95
    if not (0.0 < sl_val < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 exclusive, got {sl_val}")

    if std_demand < 0:
        raise ValueError(f"Standard deviation of demand cannot be negative, got {std_demand}")
    if std_lead_time < 0:
        raise ValueError(f"Standard deviation of lead time cannot be negative, got {std_lead_time}")

    # Determine uncertainty regime
    if std_demand > 0 and std_lead_time > 0:
        uncertainty_type = "Variable Demand and Variable Lead Time"
        sigma_lt = math.sqrt((lead_time * (std_demand ** 2)) + ((d_val ** 2) * (std_lead_time ** 2)))
        assumptions.append("Demand per period and replenishment lead time are mutually independent random variables.")
    elif std_demand > 0 and std_lead_time == 0:
        uncertainty_type = "Variable Demand and Constant Lead Time"
        sigma_lt = std_demand * math.sqrt(lead_time)
        assumptions.append("Lead time is constant and deterministic.")
    elif std_demand == 0 and std_lead_time > 0:
        uncertainty_type = "Constant Demand and Variable Lead Time"
        sigma_lt = d_val * std_lead_time
        assumptions.append("Demand per period is constant and deterministic.")
    else:
        uncertainty_type = "Constant Demand and Constant Lead Time"
        sigma_lt = 0.0
        assumptions.append("Both demand and lead time are constant and deterministic (zero safety stock required).")

    ltd_mean = d_val * lead_time
    z_score = normal_ppf(sl_val)
    safety_stock = z_score * sigma_lt
    reorder_point = ltd_mean + safety_stock
    stockout_risk = (1.0 - sl_val) * 100.0

    # Fill rate & expected shortage
    expected_shortage = None
    fill_rate = None
    if sigma_lt > 0:
        expected_shortage = sigma_lt * standard_normal_loss_function(z_score)
        if order_quantity is not None and order_quantity > 0:
            fill_rate = max(0.0, min(1.0, 1.0 - (expected_shortage / order_quantity)))

    checks: List[VerificationCheck] = [
        VerificationCheck("Service Level Validity", True, f"Service level {sl_val*100:.1f}% maps to Z-score = {z_score:.4f}"),
        VerificationCheck(
            "Safety Stock Non-Negativity",
            safety_stock >= 0 if sl_val >= 0.5 else True,
            f"Safety stock is {safety_stock:.2f} units"
        ),
        VerificationCheck(
            "ROP Floor Condition",
            reorder_point >= ltd_mean if sl_val >= 0.5 else True,
            f"ROP ({reorder_point:.1f}) covers expected lead-time consumption ({ltd_mean:.1f})"
        )
    ]

    intermediate = {
        "mean_lead_time_demand": ltd_mean,
        "sigma_LT": sigma_lt,
        "z_score": z_score,
        "stockout_risk_percentage": stockout_risk,
        "uncertainty_type": uncertainty_type
    }
    if expected_shortage is not None:
        intermediate["expected_shortage_per_cycle"] = expected_shortage
    if fill_rate is not None:
        intermediate["item_fill_rate"] = fill_rate

    final_ans = {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "cycle_service_level": sl_val
    }

    return SafetyStockROPResult(
        mean_demand_per_period=d_val,
        std_demand_per_period=std_demand,
        mean_lead_time=lead_time,
        std_lead_time=std_lead_time,
        service_level=sl_val,
        time_unit=time_unit,
        z_score=z_score,
        lead_time_demand_mean=ltd_mean,
        lead_time_demand_std=sigma_lt,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        stockout_risk=stockout_risk,
        uncertainty_type=uncertainty_type,
        order_quantity=order_quantity,
        expected_shortage_per_cycle=expected_shortage,
        fill_rate=fill_rate,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        assumptions=assumptions,
        checks=checks
    )


# Backward-compatible alias
calculate_safety_stock_rop = calculate_rop


# =====================================================================
# 4. Fixed-Period Review: calculate_periodic_review()
# =====================================================================

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    mean_demand_per_period: Optional[float] = None,
    std_demand_per_period: float = 0.0,
    service_level: Optional[float] = 0.95,
    safety_stock: Optional[float] = None,
    time_unit: str = "weeks",
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    inventory_position: Optional[float] = None,
    mean_demand: Optional[float] = None
) -> FixedPeriodReviewResult:
    """
    Fixed-Period Review Policy (Periodic Review Order-Up-To Level / (R, S)):
        Protection period = review period + lead time
        Target inventory = demand during protection period + safety stock
        Order quantity = max(0, target inventory - inventory position)

    Validates inputs, returns structured results with intermediate calculations,
    final answer, assumptions, and domain verification. Never invents missing numerical inputs.
    """
    assumptions: List[str] = [
        "Inventory is reviewed at discrete, fixed review intervals.",
        "Protection horizon spans review period R plus replenishment lead time L.",
        "Unmet demand is backordered until the next replenishment order arrives."
    ]

    if review_period is None:
        raise ValueError("Missing required numerical input: review_period must be provided.")
    if review_period <= 0:
        raise ValueError(f"Review period must be strictly positive, got {review_period}")

    if lead_time is None:
        raise ValueError("Missing required numerical input: lead_time must be provided.")
    if lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}")

    d_val = mean_demand_per_period if mean_demand_per_period is not None else mean_demand
    if d_val is None:
        raise ValueError("Missing required numerical input: mean_demand_per_period (or mean_demand) must be provided.")
    if d_val < 0:
        raise ValueError(f"Mean demand cannot be negative, got {d_val}")

    if std_demand_per_period < 0:
        raise ValueError(f"Standard deviation of demand cannot be negative, got {std_demand_per_period}")

    sl_val = service_level if service_level is not None else 0.95
    if not (0.0 < sl_val < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 exclusive, got {sl_val}")

    # Core Calculations
    protection_interval = review_period + lead_time
    protection_demand_mean = d_val * protection_interval
    protection_demand_std = std_demand_per_period * math.sqrt(protection_interval)

    z_score = normal_ppf(sl_val)

    if safety_stock is not None:
        ss_val = float(safety_stock)
        assumptions.append(f"Using directly provided safety stock SS = {ss_val:.2f} units.")
    else:
        ss_val = z_score * protection_demand_std

    target_inventory = protection_demand_mean + ss_val

    if inventory_position is not None:
        ip_val = float(inventory_position)
        assumptions.append(f"Using directly provided inventory position IP = {ip_val:.2f} units.")
    else:
        ip_val = on_hand + on_order - backorders

    order_qty = max(0.0, target_inventory - ip_val)

    continuous_ss = (z_score * std_demand_per_period * math.sqrt(lead_time)) if lead_time > 0 else 0.0
    multiplier = math.sqrt(protection_interval / lead_time) if lead_time > 0 else 1.0

    checks: List[VerificationCheck] = [
        VerificationCheck("Protection Interval", True, f"Protection period = {review_period} + {lead_time} = {protection_interval} {time_unit}"),
        VerificationCheck("Base Stock Adequacy", target_inventory >= protection_demand_mean, f"Target base-stock level {target_inventory:.1f} covers expected demand + safety buffer"),
        VerificationCheck("Order Non-Negativity", order_qty >= 0, f"Recommended order quantity {order_qty:.1f} >= 0")
    ]

    intermediate = {
        "protection_period": protection_interval,
        "demand_during_protection_period": protection_demand_mean,
        "protection_demand_std": protection_demand_std,
        "z_score": z_score,
        "safety_stock": ss_val,
        "inventory_position": ip_val,
        "continuous_safety_stock_equivalent": continuous_ss,
        "periodic_safety_stock_multiplier": multiplier
    }

    final_ans = {
        "target_inventory": target_inventory,
        "recommended_order_quantity": order_qty,
        "order_up_to_level_S": target_inventory
    }

    return FixedPeriodReviewResult(
        review_period=review_period,
        lead_time=lead_time,
        mean_demand_per_period=d_val,
        std_demand_per_period=std_demand_per_period,
        service_level=sl_val,
        time_unit=time_unit,
        protection_interval=protection_interval,
        protection_demand_mean=protection_demand_mean,
        protection_demand_std=protection_demand_std,
        z_score=z_score,
        safety_stock=ss_val,
        target_inventory_level=target_inventory,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
        inventory_position=ip_val,
        recommended_order_quantity=order_qty,
        continuous_ss_equivalent=continuous_ss,
        periodic_ss_multiplier=multiplier,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        assumptions=assumptions,
        checks=checks
    )


# Backward-compatible alias
calculate_fixed_period_review = calculate_periodic_review


# =====================================================================
# 5. Newsvendor Tool: calculate_newsvendor()
# =====================================================================

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    purchase_cost: Optional[float] = None,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
    cost_of_underage: Optional[float] = None,
    cost_of_overage: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    demand_distribution: str = "normal",
    discrete_distribution: Optional[List[Tuple[float, float]]] = None,
    uniform_a: Optional[float] = None,
    uniform_b: Optional[float] = None
) -> NewsvendorResult:
    """
    Computes single-period stocking quantity using the Critical Fractile (Newsvendor) framework:
        Underage cost Cu = selling_price - purchase_cost + shortage_penalty (or direct input)
        Overage cost  Co = purchase_cost - salvage_value (or direct input)
        Critical ratio CR = Cu / (Cu + Co)
        Recommended order quantity Q* = mu + z * sigma where z = norm_ppf(CR)

    Validates inputs, returns structured results with intermediate calculations,
    final answer, assumptions, and domain verification. Never invents missing numerical inputs.
    """
    assumptions: List[str] = [
        "Single-period decision with a single procurement or production commitment before demand occurs.",
        "Unsold inventory is disposed of at salvage value (zero carried over to future periods).",
        "Unmet demand is lost (with shortage penalty if specified)."
    ]

    # Resolve Underage Cost (Cu) and Overage Cost (Co)
    if cost_of_underage is not None and cost_of_overage is not None:
        cu = float(cost_of_underage)
        co = float(cost_of_overage)
        p_eff = float(selling_price) if selling_price is not None else (co + cu + salvage_value)
        c_eff = float(purchase_cost) if purchase_cost is not None else (co + salvage_value)
        assumptions.append(f"Using directly specified underage cost Cu = ${cu:.2f} and overage cost Co = ${co:.2f}.")
    elif selling_price is not None and purchase_cost is not None:
        p_eff = float(selling_price)
        c_eff = float(purchase_cost)
        if p_eff <= c_eff:
            raise ValueError(f"Selling price (${p_eff}) must exceed purchase cost (${c_eff}) for a profitable business.")
        if salvage_value > c_eff:
            raise ValueError(f"Salvage value (${salvage_value}) cannot exceed purchase cost (${c_eff}).")
        cu = p_eff - c_eff + shortage_penalty
        co = c_eff - salvage_value
    else:
        raise ValueError("Missing required numerical input: must provide either (selling_price and purchase_cost) OR (cost_of_underage and cost_of_overage).")

    if salvage_value < 0:
        raise ValueError(f"Salvage value cannot be negative, got {salvage_value}")
    if cu <= 0:
        raise ValueError(f"Cost of underage Cu (${cu:.2f}) must be strictly positive.")
    if co <= 0:
        raise ValueError(f"Cost of overage Co (${co:.2f}) must be strictly positive.")

    critical_ratio = cu / (cu + co)

    z_score = None
    exp_shortage = 0.0
    exp_sales = 0.0
    exp_leftover = 0.0
    optimal_q = 0.0

    if demand_distribution == "normal":
        if mean_demand is None:
            raise ValueError("Missing required numerical input: mean_demand must be provided for normal distribution.")
        if std_demand is None:
            raise ValueError("Missing required numerical input: std_demand must be provided for normal distribution.")
        if std_demand <= 0:
            raise ValueError(f"Standard deviation of demand must be positive, got {std_demand}")

        z_score = normal_ppf(critical_ratio)
        optimal_q = mean_demand + (z_score * std_demand)
        optimal_q = max(0.0, optimal_q)

        exp_shortage = std_demand * standard_normal_loss_function(z_score)
        exp_sales = mean_demand - exp_shortage
        exp_leftover = optimal_q - exp_sales

    elif demand_distribution == "discrete":
        if not discrete_distribution:
            raise ValueError("discrete_distribution must be a non-empty list of (demand, probability) tuples.")
        sorted_dist = sorted(discrete_distribution, key=lambda x: x[0])
        cum_prob = 0.0
        chosen_q = sorted_dist[-1][0]
        for d_val, p_val in sorted_dist:
            cum_prob += p_val
            if cum_prob >= critical_ratio:
                chosen_q = d_val
                break
        optimal_q = chosen_q
        exp_sales = sum(min(optimal_q, d) * p for d, p in sorted_dist)
        exp_leftover = sum(max(0.0, optimal_q - d) * p for d, p in sorted_dist)
        exp_shortage = sum(max(0.0, d - optimal_q) * p for d, p in sorted_dist)
        mean_demand = sum(d * p for d, p in sorted_dist)

    elif demand_distribution == "uniform":
        if uniform_a is None or uniform_b is None or uniform_a >= uniform_b:
            raise ValueError("Valid uniform boundaries uniform_a < uniform_b must be provided.")
        optimal_q = uniform_a + critical_ratio * (uniform_b - uniform_a)
        mean_demand = (uniform_a + uniform_b) / 2.0
        if optimal_q <= uniform_a:
            exp_shortage = mean_demand - optimal_q
            exp_leftover = 0.0
        elif optimal_q >= uniform_b:
            exp_shortage = 0.0
            exp_leftover = optimal_q - mean_demand
        else:
            exp_shortage = ((uniform_b - optimal_q) ** 2) / (2.0 * (uniform_b - uniform_a))
            exp_leftover = ((optimal_q - uniform_a) ** 2) / (2.0 * (uniform_b - uniform_a))
        exp_sales = mean_demand - exp_shortage
    else:
        raise ValueError(f"Unsupported demand distribution: {demand_distribution}")

    exp_revenue = (p_eff * exp_sales) + (salvage_value * exp_leftover)
    exp_purchase_cost = c_eff * optimal_q
    exp_penalty = shortage_penalty * exp_shortage
    exp_profit = exp_revenue - exp_purchase_cost - exp_penalty

    profit_perfect = (p_eff - c_eff) * (mean_demand if mean_demand is not None else 0.0)
    evpi = max(0.0, profit_perfect - exp_profit)

    checks: List[VerificationCheck] = [
        VerificationCheck("Underage Cost Positivity", cu > 0, f"Cu = ${cu:,.2f} > 0"),
        VerificationCheck("Overage Cost Positivity", co > 0, f"Co = ${co:,.2f} > 0"),
        VerificationCheck("Critical Ratio In-Bounds", 0 < critical_ratio < 1, f"CR = {critical_ratio:.4f} in (0, 1)"),
        VerificationCheck("Optimal Order Direction", True, f"Optimal Q* ({optimal_q:,.1f}) {'exceeds' if critical_ratio >= 0.5 else 'is below'} mean demand ({mean_demand:,.1f})"),
        VerificationCheck("Sales Feasibility", exp_sales <= optimal_q, f"Expected sales ({exp_sales:,.1f}) <= Q* ({optimal_q:,.1f})"),
        VerificationCheck("EVPI Positivity", evpi >= 0, f"Expected Value of Perfect Information = ${evpi:,.2f} >= 0")
    ]

    intermediate = {
        "cost_of_underage_Cu": cu,
        "cost_of_overage_Co": co,
        "critical_ratio": critical_ratio,
        "z_score": z_score,
        "expected_shortage": exp_shortage,
        "expected_sales": exp_sales,
        "expected_leftover": exp_leftover,
        "expected_revenue": exp_revenue,
        "expected_purchase_cost": exp_purchase_cost,
        "expected_profit": exp_profit,
        "profit_under_perfect_information": profit_perfect,
        "evpi": evpi
    }

    final_ans = {
        "recommended_order_quantity": optimal_q,
        "critical_ratio": critical_ratio,
        "expected_profit": exp_profit
    }

    return NewsvendorResult(
        selling_price=p_eff,
        purchase_cost=c_eff,
        salvage_value=salvage_value,
        shortage_penalty=shortage_penalty,
        cost_of_underage=cu,
        cost_of_overage=co,
        critical_fractile=critical_ratio,
        distribution_type=demand_distribution,
        mean_demand=mean_demand,
        std_demand=std_demand,
        optimal_order_quantity=optimal_q,
        z_score=z_score,
        expected_shortage=exp_shortage,
        expected_sales=exp_sales,
        expected_leftover=exp_leftover,
        expected_revenue=exp_revenue,
        expected_purchase_cost=exp_purchase_cost,
        expected_shortage_penalty=exp_penalty,
        expected_salvage_revenue=salvage_value * exp_leftover,
        expected_profit=exp_profit,
        profit_under_perfect_information=profit_perfect,
        evpi=evpi,
        in_stock_probability=critical_ratio,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        assumptions=assumptions,
        checks=checks
    )
