"""
Supply Chain Analytics - Inventory Decision Agent
Architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Recognizes extensive inventory terminology, synonyms, and domain specifications.
Examines the entire question thoroughly before declaring any information missing.
"""

import re
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_safety_stock_rop,
    calculate_fixed_period_review,
    calculate_newsvendor,
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
    VerificationCheck
)


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point (Continuous Review)"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review (Periodic Review Order-Up-To)"
    NEWSVENDOR = "Newsvendor Model (Single-Period Decision)"


@dataclass
class ModelIdentificationResult:
    selected_model: InventoryModelType
    confidence: float
    reasoning: str
    scores: Dict[str, float]


@dataclass
class InputExtractionResult:
    model_type: InventoryModelType
    extracted_params: Dict[str, Any]
    missing_required_params: List[str]
    assumptions_applied: List[str]
    raw_matches: Dict[str, Any]


@dataclass
class ToolExecutionResult:
    model_type: InventoryModelType
    tool_name: str
    raw_result: Any
    is_success: bool
    error_message: Optional[str] = None


@dataclass
class VerificationSummary:
    model_type: InventoryModelType
    all_passed: bool
    checks: List[VerificationCheck]
    audit_notes: List[str]


@dataclass
class AgentExplanation:
    headline_recommendation: str
    mathematical_derivation: str
    numeric_breakdown: Dict[str, Any]
    managerial_insights: List[str]
    verification_audit: List[str]
    formatted_report: str


@dataclass
class AgentResponse:
    customer_question: str
    model_identification: ModelIdentificationResult
    input_extraction: InputExtractionResult
    tool_execution: ToolExecutionResult
    verification: VerificationSummary
    explanation: AgentExplanation
    execution_pipeline_trace: List[str]


class InventoryDecisionAgent:
    """
    Intelligent Supply Chain Inventory Decision Agent.
    Orchestrates the end-to-end pipeline:
    Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
    """

    def __init__(self):
        # Keyword and heuristic profiles for model classification
        self.model_keywords = {
            InventoryModelType.QUANTITY_DISCOUNT: [
                r"discount", r"price\s*tier", r"volume\s*discount", r"bulk", r"price\s*break",
                r"quantity\s*break", r"break\s*point", r"tier", r"schedule", r"all-units", r"bracket"
            ],
            InventoryModelType.FIXED_PERIOD_REVIEW: [
                r"fixed[\s-]period", r"periodic\s*review", r"review\s*period", r"order-up-to",
                r"base[\s-]stock", r"every\s*\d+\s*(?:day|week|month)", r"protection\s*interval",
                r"periodic", r"on[\s-]order", r"on[\s-]hand", r"inventory\s*position"
            ],
            InventoryModelType.NEWSVENDOR: [
                r"newsvendor", r"newsboy", r"single[\s-]period", r"salvage", r"leftover",
                r"perishable", r"seasonal", r"critical\s*fractile", r"critical\s*ratio",
                r"cost\s*of\s*underage", r"cost\s*of\s*overage", r"underage\s*cost", r"overage\s*cost",
                r"excess\s*cost", r"clearance", r"scrap\s*value", r"unmet\s*demand\s*penalty", r"holiday"
            ],
            InventoryModelType.SAFETY_STOCK_ROP: [
                r"safety\s*stock", r"reorder\s*point", r"rop", r"service\s*level", r"lead\s*time\s*demand",
                r"std\s*dev", r"standard\s*deviation", r"stockout", r"continuous\s*review", r"csl",
                r"lead\s*time\s*uncertainty", r"cycle\s*service\s*level"
            ],
            InventoryModelType.EOQ: [
                r"\beoq\b", r"economic\s*order\s*quantity", r"holding\s*cost", r"ordering\s*cost",
                r"order\s*cost", r"setup\s*cost", r"setup\s*expense", r"cost\s*per\s*order",
                r"cost\s*of\s*placing\s*an\s*order", r"ordering/setup\s*cost", r"batch\s*size",
                r"optimal\s*order\s*(?:quantity|size)", r"annual\s*demand"
            ]
        }

    # =================================================================
    # Step 1: Identify Model
    # =================================================================

    def identify_model(self, question: str) -> ModelIdentificationResult:
        """
        Classifies the user question into one of the 5 inventory models.
        Uses multi-pattern semantic scoring and context disambiguation.
        """
        q_lower = question.lower()
        scores: Dict[InventoryModelType, float] = {m: 0.0 for m in InventoryModelType}

        for model, patterns in self.model_keywords.items():
            for pat in patterns:
                matches = re.findall(pat, q_lower)
                if matches:
                    scores[model] += len(matches) * 2.0

        # Disambiguation logic:
        # If quantity discount phrases appear, prioritize Quantity Discount
        if re.search(r"discount|price\s*tier|price\s*break|quantity\s*break|\btiers?\b|\bbreak\s*point\b|price\s*schedule|pricing\s*schedule|\bschedule\b|\bbrackets?\b", q_lower):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 7.0

        # If periodic review / review period appears, prioritize Fixed Period Review
        if re.search(r"review\s*period|review\s*interval|every\s*\d+\s*(?:weeks?|days?|months?)|order-up-to|base[\s-]stock", q_lower):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 6.0

        # If salvage / critical fractile / single period / leftover / underage / overage appears, prioritize Newsvendor
        if re.search(r"salvage|leftover|newsvendor|single[\s-]period|critical\s*fractile|perishable|underage|overage|excess\s*cost|shortage\s*cost", q_lower):
            scores[InventoryModelType.NEWSVENDOR] += 6.0

        # Safety stock vs EOQ
        if re.search(r"safety\s*stock|reorder\s*point|\brop\b", q_lower) and not re.search(r"review\s*period|review\s*interval|every\s*\d+\s*week", q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 6.0
        elif re.search(r"lead[\s-]time", q_lower) and not re.search(r"ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|holding\s*cost|carrying\s*cost", q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 4.0

        # Select highest scoring model
        best_model = max(scores, key=scores.get)
        highest_score = scores[best_model]

        # Calculate confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = min(0.99, round(highest_score / (total_score + 1.0) + 0.35, 2))
        else:
            best_model = InventoryModelType.EOQ
            confidence = 0.50

        reasoning = (
            f"Question matched primary indicators for {best_model.value}. "
            f"(Score: {highest_score:.1f}, Relative weight: {confidence*100:.0f}%)"
        )

        return ModelIdentificationResult(
            selected_model=best_model,
            confidence=confidence,
            reasoning=reasoning,
            scores={k.value: round(v, 2) for k, v in scores.items()}
        )

    # =================================================================
    # Text Extraction Helpers
    # =================================================================

    @staticmethod
    def _clean_text(text: str) -> str:
        """Standardizes text formatting, commas, currency markers, and common contractions."""
        # Normalize commas inside numbers (e.g., 10,000 -> 10000)
        t = re.sub(r"(\d+),(\d+)", r"\1\2", text)
        # Normalize slashed terminology so regex can match cleanly
        t = re.sub(r"ordering\s*/\s*setup\s*cost", "ordering/setup cost", t, flags=re.I)
        # Normalize dollar spacing: "$ 50" -> "$50"
        t = re.sub(r"\$\s+(\d+)", r"$\1", t)
        return t

    @classmethod
    def _extract_entity(
        cls,
        text: str,
        synonyms: List[str],
        is_percentage: bool = False,
        negative_lookahead: Optional[str] = None,
        forbidden_prefixes: Optional[List[str]] = None
    ) -> Optional[Tuple[float, str]]:
        """
        Searches the entire text in both forward and backward directions for any synonym pattern.
        Supports flexible linking words (averages, mean, stands at, etc.) and forbidden prefix checks.
        Returns (numeric_value, matched_synonym_text) or None.
        """
        t = text
        linking_words = r"(?:\b(?:is|are|of|averages?|amounting\s*to|totaling|total|equals?|at|average|mean|to|about|approx(?:imately)?|currently|stands\s*at)\b|[:=]|\s+)*"

        for syn in synonyms:
            # Pattern with negative lookahead if specified
            if negative_lookahead:
                guard = rf"(?!\s*{negative_lookahead})"
            else:
                guard = ""

            if is_percentage:
                # 1. Forward percentage: syn is 20%, syn of 0.20, syn are 20%
                fp = rf"(?:{syn}){guard}\s*{linking_words}\s*(\d+(?:\.\d+)?)\s*%"
                m = re.search(fp, t, re.I)
                if m:
                    if forbidden_prefixes:
                        prefix = t[max(0, m.start() - 25):m.start()].lower()
                        if any(bad in prefix for bad in forbidden_prefixes):
                            continue
                    return (float(m.group(1)) / 100.0, syn)

                fp_dec = rf"(?:{syn}){guard}\s*{linking_words}\s*(0\.\d+)"
                m_dec = re.search(fp_dec, t, re.I)
                if m_dec:
                    if forbidden_prefixes:
                        prefix = t[max(0, m_dec.start() - 25):m_dec.start()].lower()
                        if any(bad in prefix for bad in forbidden_prefixes):
                            continue
                    return (float(m_dec.group(1)), syn)

                # 2. Backward percentage: 20% syn
                bp = rf"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?(?:{syn}){guard}"
                m_b = re.search(bp, t, re.I)
                if m_b:
                    if forbidden_prefixes:
                        prefix = t[max(0, m_b.start() - 25):m_b.start()].lower()
                        if any(bad in prefix for bad in forbidden_prefixes):
                            continue
                    return (float(m_b.group(1)) / 100.0, syn)

            else:
                # 1. Forward: syn is $50, syn are 5 units, syn of 50 units, syn = 50, syn: average 50
                fp = rf"(?:{syn}){guard}\s*{linking_words}\s*\$?\s*(\d+(?:\.\d+)?)"
                m = re.search(fp, t, re.I)
                if m:
                    if forbidden_prefixes:
                        prefix = t[max(0, m.start() - 25):m.start()].lower()
                        if any(bad in prefix for bad in forbidden_prefixes):
                            continue
                    return (float(m.group(1)), syn)

                # 2. Backward: $50 syn, 50 units syn, 5 days syn, 5 units on backorder
                bp = rf"\$?\s*(\d+(?:\.\d+)?)\s*(?:units?|items?|orders?|days?|weeks?|months?|dollars?|USD)?\s*(?:of|per|for|in|on)?\s*(?:{syn}){guard}"
                m_b = re.search(bp, t, re.I)
                if m_b:
                    if forbidden_prefixes:
                        prefix = t[max(0, m_b.start() - 25):m_b.start()].lower()
                        if any(bad in prefix for bad in forbidden_prefixes):
                            continue
                    return (float(m_b.group(1)), syn)

        return None

    # =================================================================
    # Step 2: Extract Inputs
    # =================================================================

    def extract_inputs(self, question: str, model_type: InventoryModelType) -> InputExtractionResult:
        """
        Examines the entire question before declaring any information missing.
        Recognizes common supply chain terminology, synonyms, and domain specifications.
        """
        q = self._clean_text(question)
        extracted: Dict[str, Any] = {}
        missing: List[str] = []
        assumptions: List[str] = []
        raw_matches: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # 1. Economic Order Quantity (EOQ)
        # -------------------------------------------------------------
        if model_type == InventoryModelType.EOQ:
            # 1.1 Ordering Cost (treat all 7 user-specified synonyms + standard abbreviations as ordering cost)
            ordering_cost_synonyms = [
                r"ordering/setup\s*cost",
                r"ordering\s*cost",
                r"order\s*cost",
                r"setup\s*cost",
                r"setup\s*expense",
                r"cost\s*per\s*order",
                r"cost\s*of\s*placing\s*an\s*order",
                r"cost\s*of\s*placing\s*order",
                r"cost\s*to\s*place\s*an\s*order",
                r"cost\s*to\s*place\s*order",
                r"cost\s*of\s*an\s*order",
                r"order\s*placement\s*cost",
                r"setup\s*charge",
                r"order\s*charge",
                r"ordering\s*charge",
                r"setup\s*fee",
                r"order\s*fee",
                r"order\s*expense",
                r"\bS\b"
            ]
            ord_res = self._extract_entity(q, ordering_cost_synonyms)
            if ord_res:
                extracted["ordering_cost"] = ord_res[0]
                raw_matches["ordering_cost"] = ord_res[1]

            # 1.2 Annual Demand
            demand_synonyms = [
                r"annual\s*demand",
                r"yearly\s*demand",
                r"annual\s*usage",
                r"annual\s*requirement",
                r"annual\s*sales",
                r"demand\s*per\s*year",
                r"demand\s*(?:is|of|averages|amounts\s*to)?",
                r"sells?\s*\d+\s*units",
                r"uses?\s*\d+\s*units",
                r"\bD\b"
            ]
            dem_res = self._extract_entity(q, demand_synonyms)
            if dem_res:
                extracted["annual_demand"] = dem_res[0]
                raw_matches["annual_demand"] = dem_res[1]
            else:
                # Direct regex for "sells 12000 units"
                m_sell = re.search(r"(?:sells?|uses?)\s*(\d+(?:\.\d+)?)\s*units", q, re.I)
                if m_sell:
                    extracted["annual_demand"] = float(m_sell.group(1))
                    raw_matches["annual_demand"] = "sells X units"

            # 1.3 Percentage Holding Cost / Rate
            holding_rate_synonyms = [
                r"percentage\s*holding\s*cost",
                r"holding\s*cost\s*rate",
                r"holding\s*rate",
                r"carrying\s*cost\s*rate",
                r"carrying\s*rate",
                r"annual\s*holding\s*rate",
                r"cost\s*of\s*capital\s*rate",
                r"cost\s*of\s*capital",
                r"holding\s*percentage",
                r"carrying\s*percentage",
                r"\bh\b"
            ]
            hr_res = self._extract_entity(q, holding_rate_synonyms, is_percentage=True)
            if hr_res:
                extracted["holding_cost_rate"] = hr_res[0]
                raw_matches["holding_cost_rate"] = hr_res[1]
            else:
                # Generic pattern: "holding cost is 20%" or "20% holding cost"
                m_generic_pct = re.search(r"(?:holding|carrying)[^\d\n]*?(\d+(?:\.\d+)?)\s*%", q, re.I)
                if not m_generic_pct:
                    m_generic_pct = re.search(r"(\d+(?:\.\d+)?)\s*%[^\d\n]*?(?:holding|carrying)", q, re.I)
                if m_generic_pct:
                    extracted["holding_cost_rate"] = float(m_generic_pct.group(1)) / 100.0
                    raw_matches["holding_cost_rate"] = "holding cost percentage"

            # 1.4 Fixed Holding Cost ($/unit/year)
            holding_cost_synonyms = [
                r"fixed\s*holding\s*cost",
                r"holding\s*cost\s*per\s*unit",
                r"carrying\s*cost\s*per\s*unit",
                r"holding\s*cost",
                r"carrying\s*cost",
                r"storage\s*cost",
                r"holding\s*expense",
                r"carrying\s*expense",
                r"cost\s*of\s*holding",
                r"cost\s*to\s*carry",
                r"holding\s*charge",
                r"inventory\s*carrying\s*cost",
                r"\bH\b"
            ]
            # Guard against percentage
            hc_res = self._extract_entity(q, holding_cost_synonyms, is_percentage=False, negative_lookahead=r"[^\$0-9\n]*?\d+\s*%")
            if hc_res:
                # Check if it was actually a percentage mistakenly parsed
                m_chk_pct = re.search(rf"{hc_res[1]}[^\d\n]*?(\d+(?:\.\d+)?)\s*%", q, re.I)
                if not m_chk_pct:
                    extracted["holding_cost_per_unit"] = hc_res[0]
                    raw_matches["holding_cost_per_unit"] = hc_res[1]

            # 1.5 Unit Cost / Unit Purchase Price
            unit_cost_synonyms = [
                r"unit\s*purchase\s*price",
                r"unit\s*cost",
                r"purchase\s*price",
                r"purchase\s*cost",
                r"item\s*cost",
                r"item\s*price",
                r"cost\s*per\s*unit",
                r"price\s*per\s*unit",
                r"wholesale\s*price",
                r"acquisition\s*cost",
                r"\bC\b"
            ]
            uc_res = self._extract_entity(q, unit_cost_synonyms)
            if uc_res:
                extracted["unit_cost"] = uc_res[0]
                raw_matches["unit_cost"] = uc_res[1]

            # 1.6 Lead Time & Working Days
            lt_synonyms = [r"lead\s*time", r"delivery\s*lead\s*time", r"replenishment\s*lead\s*time", r"delivery\s*time", r"\bL\b"]
            lt_res = self._extract_entity(q, lt_synonyms)
            if lt_res:
                lt_val = lt_res[0]
                # Check if weeks mentioned
                if re.search(rf"{lt_res[1]}[^\d\n]*?{lt_val}\s*weeks?", q, re.I):
                    lt_val *= 7.0
                extracted["lead_time_days"] = lt_val

            m_wd = re.search(r"(?:working\s*days|days\s*per\s*year|work\s*days)\s*(?:is|of|:|=)?\s*(\d+)", q, re.I)
            extracted["work_days"] = float(m_wd.group(1)) if m_wd else 365.0

            # Resolution of holding cost (fixed vs percentage * unit_cost)
            if "holding_cost_per_unit" not in extracted:
                if "holding_cost_rate" in extracted and "unit_cost" in extracted:
                    extracted["holding_cost_per_unit"] = extracted["holding_cost_rate"] * extracted["unit_cost"]
                    assumptions.append(
                        f"Calculated holding cost H = {extracted['holding_cost_rate']*100:.1f}% * ${extracted['unit_cost']:.2f} = ${extracted['holding_cost_per_unit']:.2f}/unit/yr"
                    )
                elif "holding_cost_rate" in extracted:
                    missing.append("unit_cost (required to convert percentage holding cost into $/unit/year)")
                else:
                    missing.append("holding_cost_per_unit (or percentage holding cost + unit purchase price)")

            if "annual_demand" not in extracted:
                missing.append("annual_demand")
            if "ordering_cost" not in extracted:
                missing.append("ordering_cost (or setup cost/expense, cost per order, cost of placing an order)")

        # -------------------------------------------------------------
        # 2. Quantity Discount Model
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            # 2.1 Demand
            dem_res = self._extract_entity(q, [r"annual\s*demand", r"demand\s*(?:is|of|averages)?", r"\bD\b"])
            if dem_res:
                extracted["annual_demand"] = dem_res[0]
            else:
                m_sell = re.search(r"(?:sells?|uses?)\s*(\d+(?:\.\d+)?)\s*units", q, re.I)
                if m_sell:
                    extracted["annual_demand"] = float(m_sell.group(1))
                else:
                    missing.append("annual_demand")

            # 2.2 Ordering Cost (all synonyms)
            ord_res = self._extract_entity(q, [
                r"ordering/setup\s*cost", r"ordering\s*cost", r"order\s*cost", r"setup\s*cost",
                r"setup\s*expense", r"cost\s*per\s*order", r"cost\s*of\s*placing\s*an\s*order",
                r"cost\s*of\s*placing\s*order", r"order\s*fee", r"\bS\b"
            ])
            if ord_res:
                extracted["ordering_cost"] = ord_res[0]
            else:
                missing.append("ordering_cost (or setup cost/expense, cost per order)")

            # 2.3 Percentage Holding Cost vs Fixed Holding Cost
            hr_res = self._extract_entity(q, [
                r"percentage\s*holding\s*cost", r"holding\s*cost\s*rate", r"holding\s*rate",
                r"carrying\s*cost\s*rate", r"carrying\s*rate", r"annual\s*holding\s*rate"
            ], is_percentage=True)

            hc_res = self._extract_entity(q, [
                r"fixed\s*holding\s*cost", r"holding\s*cost\s*per\s*unit", r"fixed\s*carrying\s*cost"
            ], is_percentage=False, negative_lookahead=r"[^\$0-9\n]*?\d+\s*%")

            if hr_res:
                extracted["holding_cost_rate"] = hr_res[0]
            elif hc_res:
                extracted["fixed_holding_cost"] = hc_res[0]
            else:
                extracted["holding_cost_rate"] = 0.20
                assumptions.append("Percentage holding cost not explicitly specified; assumed 20% (0.20) of unit price.")

            # 2.4 Parse Discount Schedule / Quantity Breaks
            # Support ranges: 1-499 at $20, 500-999 at $18, 1000+ at $16
            schedule: List[Dict[str, Any]] = []

            tier_range_matches = re.findall(
                r"(?:quantity\s*break|tier\s*\d+|bracket)?\s*(\d+)\s*(?:-|to)\s*(\d+)[^\$\d\n]*?\$?\s*(\d+(?:\.\d+)?)",
                q, re.I
            )
            for tm in tier_range_matches:
                schedule.append({
                    "min_qty": float(tm[0]),
                    "max_qty": float(tm[1]),
                    "unit_price": float(tm[2])
                })

            top_tier = re.search(
                r"(\d+)\s*(?:\+|or\s*more|and\s*above|and\s*over|>=)[^\$\d\n]*?\$?\s*(\d+(?:\.\d+)?)",
                q, re.I
            )
            if top_tier:
                schedule.append({
                    "min_qty": float(top_tier.group(1)),
                    "max_qty": None,
                    "unit_price": float(top_tier.group(2))
                })

            # Check for "price breaks at 500 ($18) and 1000 ($16) with base unit purchase price of $20"
            if not schedule:
                m_base = re.search(r"(?:base\s*price|unit\s*purchase\s*price|regular\s*price|standard\s*price)[^\$\d\n]*?\$?\s*(\d+(?:\.\d+)?)", q, re.I)
                breaks = re.findall(r"(?:break|tier|threshold|at\s*least)?\s*(\d+)\s*(?:units)?[^\$\d\n]*?\$?\s*(\d+(?:\.\d+)?)", q, re.I)
                if m_base and breaks:
                    base_p = float(m_base.group(1))
                    schedule.append({"min_qty": 1.0, "max_qty": float(breaks[0][0]) - 1.0, "unit_price": base_p})
                    for i in range(len(breaks)):
                        min_q = float(breaks[i][0])
                        max_q = float(breaks[i+1][0]) - 1.0 if i + 1 < len(breaks) else None
                        schedule.append({"min_qty": min_q, "max_qty": max_q, "unit_price": float(breaks[i][1])})

            if schedule:
                extracted["discount_schedule"] = schedule
            else:
                missing.append("discount_schedule (quantity breaks and unit prices for each tier)")

        # -------------------------------------------------------------
        # 3. Safety Stock & Reorder Point (Continuous Review)
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            # 3.1 Global direct search for explicit composite terms first
            lts_synonyms = [
                r"lead[\s-]time\s*standard\s*deviation",
                r"lead[\s-]time\s*std\s*dev",
                r"lead[\s-]time\s*sigma",
                r"standard\s*deviation\s*(?:of|in)\s*lead[\s-]time",
                r"std\s*dev\s*(?:of|in)\s*lead[\s-]time",
                r"variability\s*in\s*lead[\s-]time",
                r"lead[\s-]time\s*variability",
                r"sigma_L"
            ]
            lts_res = self._extract_entity(q, lts_synonyms)
            if lts_res:
                extracted["std_lead_time"] = lts_res[0]

            ds_synonyms = [
                r"demand\s*standard\s*deviation",
                r"demand\s*std\s*dev",
                r"demand\s*sigma",
                r"standard\s*deviation\s*(?:of|in)\s*demand",
                r"std\s*dev\s*(?:of|in)\s*demand",
                r"variability\s*in\s*demand",
                r"demand\s*variability",
                r"sigma_d"
            ]
            ds_res = self._extract_entity(q, ds_synonyms)
            if ds_res:
                extracted["std_demand"] = ds_res[0]

            # 3.2 Service Level
            sl_synonyms = [
                r"cycle\s*service\s*level",
                r"service\s*level",
                r"csl",
                r"target\s*service\s*level",
                r"desired\s*service\s*level",
                r"in[\s-]stock\s*probability",
                r"non[\s-]stockout\s*probability",
                r"type\s*1\s*service\s*level"
            ]
            sl_res = self._extract_entity(q, sl_synonyms, is_percentage=True)
            if sl_res:
                extracted["service_level"] = sl_res[0]
            else:
                extracted["service_level"] = 0.95
                assumptions.append("Target service level not specified; assumed standard 95% (0.95).")

            # 3.3 Clause-aware chunking for proximity and disambiguation
            pattern = r"\.(?!\d)|;|\n|,\s*(?=(?:average\s+|mean\s+)?(?:lead[\s-]time|delivery|supplier|demand|daily|weekly|annual|service))\b|\band\s+(?=(?:average\s+|mean\s+)?(?:lead[\s-]time|delivery|supplier|demand|daily|weekly|annual))\b"
            chunks = [c.strip() for c in re.split(pattern, q, flags=re.I) if c and c.strip()]

            for c in chunks:
                has_lead = bool(re.search(r"lead[\s-]time|delivery|supplier", c, re.I))
                has_demand = bool(re.search(r"demand|sales|usage", c, re.I))

                # Lead-time specific clause
                if has_lead and not has_demand:
                    if "lead_time" not in extracted:
                        m_lt = re.search(r"(?:average\s+|mean\s+)?(?:lead[\s-]time|delivery\s*time|supplier)(?!\s*(?:standard\s*deviation|std\s*dev|sigma|variance|variability))[^\d\n]*?(?:average|mean|averages|is|are|of|to|equals?|at|:|=)?\s*(\d+(?:\.\d+)?)", c, re.I)
                        if not m_lt:
                            m_lt = re.search(r"(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)\s*(?:lead[\s-]time|delivery)(?!\s*(?:standard\s*deviation|std\s*dev|sigma))", c, re.I)
                        if m_lt:
                            extracted["lead_time"] = float(m_lt.group(1))

                    if "std_lead_time" not in extracted:
                        m_sd = re.search(r"(?:standard\s*deviation|std\s*dev|sigma)[^\d\n]*?(\d+(?:\.\d+)?)", c, re.I)
                        if m_sd:
                            extracted["std_lead_time"] = float(m_sd.group(1))

                # Demand specific clause
                elif has_demand and not has_lead:
                    if "mean_demand" not in extracted:
                        m_dm = re.search(r"(?:average|mean|expected|daily|weekly|annual)?\s*demand(?!\s*(?:standard\s*deviation|std\s*dev|sigma|variance|variability))[^\d\n]*?(?:average|mean|averages|is|are|of|to|equals?|at|:|=)?\s*(\d+(?:\.\d+)?)", c, re.I)
                        if not m_dm:
                            m_dm = re.search(r"(\d+(?:\.\d+)?)\s*(?:units?|items?)[^\d\n]*?(?:demand|sales)(?!\s*(?:standard\s*deviation|std\s*dev|sigma))", c, re.I)
                        if m_dm:
                            extracted["mean_demand"] = float(m_dm.group(1))

                    if "std_demand" not in extracted:
                        m_sd = re.search(r"(?:standard\s*deviation|std\s*dev|sigma)[^\d\n]*?(\d+(?:\.\d+)?)", c, re.I)
                        if m_sd:
                            extracted["std_demand"] = float(m_sd.group(1))

            # 3.4 Fallbacks across full question text
            if "lead_time" not in extracted:
                lt_synonyms = [
                    r"average\s*lead[\s-]time",
                    r"mean\s*lead[\s-]time",
                    r"expected\s*lead[\s-]time",
                    r"average\s*delivery\s*time",
                    r"lead[\s-]time\s*averages",
                    r"lead[\s-]time(?!\s*(?:standard\s*deviation|std\s*dev|sigma|variance|variability))",
                    r"delivery\s*time(?!\s*(?:standard\s*deviation|std\s*dev|sigma|variance|variability))",
                    r"supplier\s*lead[\s-]time",
                    r"replenishment\s*lead[\s-]time",
                    r"\bmu_L\b",
                    r"\bL\b"
                ]
                lt_res = self._extract_entity(q, lt_synonyms, negative_lookahead=r"(?:standard\s*deviation|std\s*dev|sigma|variance|variability)")
                if lt_res:
                    extracted["lead_time"] = lt_res[0]
                else:
                    missing.append("average lead time (replenishment delivery time)")

            if "mean_demand" not in extracted:
                dm_synonyms = [
                    r"average\s*demand",
                    r"mean\s*demand",
                    r"expected\s*demand",
                    r"average\s*daily\s*demand",
                    r"average\s*weekly\s*demand",
                    r"mean\s*daily\s*demand",
                    r"mean\s*weekly\s*demand",
                    r"daily\s*demand",
                    r"weekly\s*demand",
                    r"demand\s*averages",
                    r"with\s*mean",
                    r"mean\s*of",
                    r"\bmean\b",
                    r"demand(?!\s*(?:standard\s*deviation|std\s*dev|sigma|variance|variability))",
                    r"\bmu_d\b",
                    r"\bd\b"
                ]
                dm_res = self._extract_entity(q, dm_synonyms, negative_lookahead=r"(?:standard\s*deviation|std\s*dev|sigma|variance|variability)")
                if dm_res:
                    extracted["mean_demand"] = dm_res[0]
                else:
                    missing.append("average demand (mean demand per period)")

            # 3.5 Default standard deviations to 0.0 if deterministic / constant
            if "std_lead_time" not in extracted:
                extracted["std_lead_time"] = 0.0
            if "std_demand" not in extracted:
                extracted["std_demand"] = 0.0

            # 3.6 Time unit & order quantity (optional for fill rate)
            time_unit = "days"
            if re.search(r"\bweeks?\b", q, re.I):
                time_unit = "weeks"
            elif re.search(r"\bmonths?\b", q, re.I):
                time_unit = "months"
            extracted["time_unit"] = time_unit

            oq_res = self._extract_entity(q, [r"order\s*quantity", r"batch\s*size", r"lot\s*size", r"\bQ\b"])
            if oq_res:
                extracted["order_quantity"] = oq_res[0]

        # -------------------------------------------------------------
        # 4. Fixed-Period Review Model
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            # 4.1 Review Period (R or P)
            rp_synonyms = [
                r"review\s*period",
                r"review\s*interval",
                r"review\s*cycle",
                r"periodic\s*review\s*interval",
                r"reviewed\s*every",
                r"order\s*cycle",
                r"order\s*interval",
                r"inventory\s*review\s*period",
                r"re[\s-]order\s*interval",
                r"\bP\b",
                r"\bR\b"
            ]
            rp_res = self._extract_entity(q, rp_synonyms)
            if rp_res:
                extracted["review_period"] = rp_res[0]
            else:
                missing.append("review_period (interval between periodic reviews, e.g. every 2 weeks)")

            # 4.2 Lead Time (L)
            lt_res = self._extract_entity(q, [
                r"supplier\s*lead[\s-]time", r"delivery\s*lead[\s-]time", r"replenishment\s*lead[\s-]time",
                r"lead[\s-]time", r"delivery\s*time", r"\bL\b"
            ])
            extracted["lead_time"] = lt_res[0] if lt_res else 0.0

            # 4.3 Demand Mean & Std Dev
            dm_res = self._extract_entity(q, [
                r"average\s*demand", r"mean\s*demand", r"expected\s*demand",
                r"weekly\s*demand", r"daily\s*demand", r"periodic\s*demand",
                r"with\s*mean", r"mean\s*of", r"\bmean\b",
                r"demand(?!\s*(?:standard\s*deviation|std\s*dev|sigma))"
            ])
            if dm_res:
                extracted["mean_demand_per_period"] = dm_res[0]
            else:
                missing.append("mean_demand_per_period (average demand per period)")

            ds_res = self._extract_entity(q, [
                r"demand\s*standard\s*deviation", r"demand\s*std\s*dev",
                r"standard\s*deviation\s*of\s*demand", r"std\s*dev\s*of\s*demand",
                r"standard\s*deviation", r"std\s*dev", r"\bsigma\b"
            ])
            extracted["std_demand_per_period"] = ds_res[0] if ds_res else 0.0

            # 4.4 Service Level or Directly Given Safety Stock
            ss_res = self._extract_entity(q, [r"safety\s*stock", r"safety\s*buffer", r"buffer\s*stock", r"\bSS\b"])
            if ss_res:
                extracted["safety_stock"] = ss_res[0]
                assumptions.append(f"Using directly specified safety stock SS = {ss_res[0]:,.1f} units.")

            sl_res = self._extract_entity(q, [
                r"service\s*level", r"cycle\s*service\s*level", r"csl", r"target\s*service\s*level"
            ], is_percentage=True)
            if sl_res:
                extracted["service_level"] = sl_res[0]
            elif "safety_stock" not in extracted:
                extracted["service_level"] = 0.95
                assumptions.append("Target service level not specified; assumed 95% (0.95).")

            # 4.5 Inventory Position vs Component Stock (On-Hand, On-Order, Backorders)
            ip_res = self._extract_entity(q, [
                r"current\s*inventory\s*position", r"inventory\s*position",
                r"net\s*inventory\s*position", r"\bIP\b"
            ])
            if ip_res:
                extracted["inventory_position"] = ip_res[0]
                assumptions.append(f"Using directly specified Inventory Position IP = {ip_res[0]:,.0f} units.")
            else:
                oh_res = self._extract_entity(q, [
                    r"on[\s-]hand\s*inventory", r"on[\s-]hand\s*stock", r"inventory\s*on[\s-]hand",
                    r"stock\s*on[\s-]hand", r"current\s*on[\s-]hand", r"on[\s-]hand", r"physical\s*stock"
                ])
                oo_res = self._extract_entity(q, [
                    r"on[\s-]order\s*inventory", r"on[\s-]order\s*stock", r"open\s*orders?",
                    r"pipeline\s*inventory", r"pipeline\s*stock", r"inbound\s*orders?",
                    r"units\s*on[\s-]order", r"on[\s-]order"
                ])
                bo_res = self._extract_entity(q, [
                    r"backorders?", r"backordered\s*units", r"unfilled\s*orders?", r"backlog"
                ])
                extracted["on_hand"] = oh_res[0] if oh_res else 0.0
                extracted["on_order"] = oo_res[0] if oo_res else 0.0
                extracted["backorders"] = bo_res[0] if bo_res else 0.0

            time_unit = "weeks"
            if re.search(r"\bdays?\b", q, re.I):
                time_unit = "days"
            elif re.search(r"\bmonths?\b", q, re.I):
                time_unit = "months"
            extracted["time_unit"] = time_unit

        # -------------------------------------------------------------
        # 5. Newsvendor Model (Single-Period Decision)
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.NEWSVENDOR:
            # 5.1 Selling Price (P)
            sp_synonyms = [
                r"selling\s*price", r"retail\s*price", r"sales\s*price", r"regular\s*price",
                r"price\s*per\s*unit", r"revenue\s*per\s*unit", r"sells?\s*for", r"sold\s*for", r"\bP\b"
            ]
            sp_res = self._extract_entity(q, sp_synonyms)
            if sp_res:
                extracted["selling_price"] = sp_res[0]

            # 5.2 Purchase Cost (C) (guarded against 'cost of underage' / 'cost of overage' / 'shortage cost')
            pc_synonyms = [
                r"purchase\s*cost", r"wholesale\s*cost", r"acquisition\s*cost", r"production\s*cost",
                r"procurement\s*cost", r"unit\s*purchase\s*price", r"unit\s*cost", r"cost\s*to\s*buy",
                r"cost\s*per\s*unit", r"cost\s*per\s*item", r"item\s*cost", r"product\s*cost",
                r"costs?", r"\bC\b"
            ]
            pc_res = self._extract_entity(
                q,
                pc_synonyms,
                negative_lookahead=r"of\s*(?:underage|overage|shortage|excess|ordering|holding)",
                forbidden_prefixes=["shortage", "excess", "underage", "overage", "ordering", "setup", "holding", "carrying"]
            )
            if pc_res:
                extracted["purchase_cost"] = pc_res[0]

            # 5.3 Salvage Value (V)
            sv_synonyms = [
                r"salvage\s*value", r"salvage\s*price", r"clearance\s*price", r"scrap\s*value",
                r"liquidation\s*value", r"residual\s*value", r"leftover\s*value", r"salvages?\s*for",
                r"clearance", r"\bV\b"
            ]
            sv_res = self._extract_entity(q, sv_synonyms)
            extracted["salvage_value"] = sv_res[0] if sv_res else 0.0

            # 5.4 Underage / Shortage Cost (Cu or penalty g)
            cu_synonyms = [
                r"cost\s*of\s*underage", r"underage\s*cost", r"cost\s*of\s*shortage", r"shortage\s*cost",
                r"lost\s*profit\s*per\s*unit", r"lost\s*profit", r"shortage\s*penalty",
                r"unmet\s*demand\s*penalty", r"penalty\s*per\s*unmet\s*demand", r"lost\s*goodwill",
                r"stockout\s*penalty", r"stockout\s*cost", r"\bCu\b"
            ]
            cu_res = self._extract_entity(q, cu_synonyms)
            if cu_res:
                extracted["cost_of_underage"] = cu_res[0]

            # 5.5 Overage / Excess Cost (Co)
            co_synonyms = [
                r"cost\s*of\s*overage", r"overage\s*cost", r"cost\s*of\s*excess", r"excess\s*cost",
                r"loss\s*per\s*unsold\s*unit", r"loss\s*on\s*leftover\s*unit", r"overstock\s*cost",
                r"overstock\s*penalty", r"\bCo\b"
            ]
            co_res = self._extract_entity(q, co_synonyms)
            if co_res:
                extracted["cost_of_overage"] = co_res[0]

            # Shortage penalty (if separate from Cu)
            pen_res = self._extract_entity(q, [r"shortage\s*penalty", r"unmet\s*demand\s*penalty", r"lost\s*goodwill"])
            extracted["shortage_penalty"] = pen_res[0] if pen_res else 0.0

            # Check if we have either (P and C) OR (Cu and Co)
            has_price_cost = ("selling_price" in extracted and "purchase_cost" in extracted)
            has_underage_overage = ("cost_of_underage" in extracted and "cost_of_overage" in extracted)

            if not has_price_cost and not has_underage_overage:
                if "selling_price" not in extracted and "cost_of_underage" not in extracted:
                    missing.append("selling_price (or cost_of_underage)")
                if "purchase_cost" not in extracted and "cost_of_overage" not in extracted:
                    missing.append("purchase_cost (or cost_of_overage)")

            # 5.6 Demand Mean & Standard Deviation
            dm_res = self._extract_entity(q, [
                r"mean\s*demand", r"average\s*demand", r"expected\s*demand",
                r"forecast\s*demand", r"demand\s*mean", r"mean\s*of\s*demand",
                r"with\s*mean", r"mean\s*of", r"\bmean\b",
                r"demand(?!\s*(?:standard\s*deviation|std\s*dev|sigma))", r"\bmu\b"
            ], negative_lookahead=r"(?:standard\s*deviation|std\s*dev|sigma)")
            if dm_res:
                extracted["mean_demand"] = dm_res[0]
            else:
                missing.append("mean_demand (expected average single-period demand)")

            ds_res = self._extract_entity(q, [
                r"demand\s*standard\s*deviation", r"standard\s*deviation\s*of\s*demand",
                r"standard\s*deviation\s*in\s*demand", r"demand\s*std\s*dev",
                r"std\s*dev\s*of\s*demand", r"demand\s*sigma", r"standard\s*deviation",
                r"std\s*dev", r"\bsigma\b"
            ])
            if ds_res:
                extracted["std_demand"] = ds_res[0]
            else:
                missing.append("std_demand (standard deviation of demand)")

            extracted["demand_distribution"] = "normal"

        return InputExtractionResult(
            model_type=model_type,
            extracted_params=extracted,
            missing_required_params=missing,
            assumptions_applied=assumptions,
            raw_matches=raw_matches
        )

    # =================================================================
    # Step 3: Select Python Tool
    # =================================================================

    def select_tool(self, model_type: InventoryModelType) -> Tuple[Callable, str]:
        """
        Maps the identified model to the appropriate numerical calculation tool in inventory_tools.
        """
        mapping = {
            InventoryModelType.EOQ: (calculate_eoq, "calculate_eoq"),
            InventoryModelType.QUANTITY_DISCOUNT: (calculate_quantity_discount, "calculate_quantity_discount"),
            InventoryModelType.SAFETY_STOCK_ROP: (calculate_rop, "calculate_rop"),
            InventoryModelType.FIXED_PERIOD_REVIEW: (calculate_periodic_review, "calculate_periodic_review"),
            InventoryModelType.NEWSVENDOR: (calculate_newsvendor, "calculate_newsvendor"),
        }
        return mapping[model_type]

    # =================================================================
    # Step 4: Calculate
    # =================================================================

    def calculate(self, tool_fn: Callable, tool_name: str, model_type: InventoryModelType, params: Dict[str, Any]) -> ToolExecutionResult:
        """
        Executes the pure Python numerical calculation with extracted inputs.
        """
        try:
            result = tool_fn(**params)
            return ToolExecutionResult(
                model_type=model_type,
                tool_name=tool_name,
                raw_result=result,
                is_success=True
            )
        except Exception as e:
            return ToolExecutionResult(
                model_type=model_type,
                tool_name=tool_name,
                raw_result=None,
                is_success=False,
                error_message=str(e)
            )

    # =================================================================
    # Step 5: Verify
    # =================================================================

    def verify(self, model_type: InventoryModelType, inputs: Dict[str, Any], tool_result: ToolExecutionResult) -> VerificationSummary:
        """
        Audits mathematical integrity, non-negativity, and supply chain domain constraints.
        """
        if not tool_result.is_success:
            return VerificationSummary(
                model_type=model_type,
                all_passed=False,
                checks=[VerificationCheck("Execution Success", False, f"Calculation failed: {tool_result.error_message}", "ERROR")],
                audit_notes=["Calculation encountered a fatal numerical or parameter error."]
            )

        res = tool_result.raw_result
        checks = list(res.checks) if hasattr(res, "checks") else []
        notes = []

        if model_type == InventoryModelType.EOQ:
            assert isinstance(res, EOQResult)
            if res.optimal_order_quantity <= 0:
                checks.append(VerificationCheck("Optimal Q Validity", False, "EOQ must be strictly positive", "ERROR"))
            if res.cycle_time_days > 365:
                notes.append(f"Cycle time is {res.cycle_time_days:.1f} days (> 1 year). Consider smaller batch sizes or checking demand units.")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            assert isinstance(res, QuantityDiscountResult)
            if res.optimal_order_quantity <= 0:
                checks.append(VerificationCheck("Optimal Q Validity", False, "Discount Q* must be strictly positive", "ERROR"))
            if res.annual_savings < 0:
                checks.append(VerificationCheck("Economic Justification", False, "Discounted total cost exceeds baseline cost", "WARNING"))

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            assert isinstance(res, SafetyStockROPResult)
            if res.service_level >= 0.5 and res.safety_stock < 0:
                checks.append(VerificationCheck("Safety Stock Sign", False, "Safety stock cannot be negative for CSL >= 50%", "ERROR"))

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            assert isinstance(res, FixedPeriodReviewResult)
            if res.target_inventory_level < res.protection_demand_mean:
                checks.append(VerificationCheck("Order-up-to Adequacy", False, "Target inventory is below expected protection demand", "ERROR"))

        elif model_type == InventoryModelType.NEWSVENDOR:
            assert isinstance(res, NewsvendorResult)
            if res.critical_fractile <= 0 or res.critical_fractile >= 1:
                checks.append(VerificationCheck("Critical Ratio Bounds", False, "Critical fractile must be strictly in (0, 1)", "ERROR"))

        all_passed = all(c.passed for c in checks if c.severity == "ERROR")
        return VerificationSummary(
            model_type=model_type,
            all_passed=all_passed,
            checks=checks,
            audit_notes=notes
        )

    # =================================================================
    # Step 6: Explain
    # =================================================================

    def explain(
        self,
        question: str,
        model_id: ModelIdentificationResult,
        input_ext: InputExtractionResult,
        tool_res: ToolExecutionResult,
        verification: VerificationSummary
    ) -> AgentExplanation:
        """
        Synthesizes a structured, executive Supply Chain Decision Report.
        """
        if not tool_res.is_success:
            err_msg = tool_res.error_message or "Unknown calculation error."
            return AgentExplanation(
                headline_recommendation="ERROR: Calculation could not be completed.",
                mathematical_derivation="N/A",
                numeric_breakdown={"error": err_msg},
                managerial_insights=["Please provide the missing inputs or check numerical values."],
                verification_audit=["Calculation failed."],
                formatted_report=f"### Supply Chain Decision Agent Error\n\nCould not process the query due to: {err_msg}"
            )

        res = tool_res.raw_result
        model_type = model_id.selected_model

        headline = ""
        derivation = ""
        numeric_dict: Dict[str, Any] = {}
        insights: List[str] = []

        if model_type == InventoryModelType.EOQ:
            assert isinstance(res, EOQResult)
            headline = f"Order {res.optimal_order_quantity:,.0f} units per replenishment order ({res.orders_per_year:.1f} orders/year)."
            derivation = (
                f"EOQ Equation:\n"
                f"  Q* = sqrt((2 * D * S) / H)\n"
                f"     = sqrt((2 * {res.annual_demand:,.0f} * ${res.ordering_cost:,.2f}) / ${res.holding_cost_per_unit:,.2f})\n"
                f"     = {res.optimal_order_quantity:,.2f} units (approx. {round(res.optimal_order_quantity)} units)\n"
                f"Annual Ordering Cost = (D / Q*) * S = ({res.annual_demand:,.0f} / {res.optimal_order_quantity:,.2f}) * ${res.ordering_cost:,.2f} = ${res.annual_ordering_cost:,.2f}\n"
                f"Annual Holding Cost  = (Q* / 2) * H = ({res.optimal_order_quantity:,.2f} / 2) * ${res.holding_cost_per_unit:,.2f} = ${res.annual_holding_cost:,.2f}"
            )
            numeric_dict = {
                "Optimal Order Quantity (EOQ)": f"{res.optimal_order_quantity:,.2f} units",
                "Annual Ordering Cost": f"${res.annual_ordering_cost:,.2f}",
                "Annual Holding Cost": f"${res.annual_holding_cost:,.2f}",
                "Total Annual Inventory Cost": f"${res.total_inventory_cost:,.2f}",
                "Order Frequency": f"{res.orders_per_year:.2f} orders/year",
                "Cycle Time Between Orders": f"{res.cycle_time_days:.1f} days ({res.cycle_time_years*52:.1f} weeks)",
                "Average Cycle Inventory": f"{res.average_cycle_inventory:,.1f} units"
            }
            if res.reorder_point is not None:
                numeric_dict["Deterministic Reorder Point (ROP)"] = f"{res.reorder_point:,.1f} units"
            if res.total_acquisition_cost is not None:
                numeric_dict["Total Annual Acquisition Cost"] = f"${res.total_acquisition_cost:,.2f}"

            insights = [
                f"Optimality Balance: At EOQ, annual setup cost (${res.annual_ordering_cost:,.2f}) perfectly balances carrying cost (${res.annual_holding_cost:,.2f}).",
                f"Cadence: Place an order approximately every {res.cycle_time_days:.0f} calendar days.",
                "Sensitivity: EOQ exhibits high cost resilience (a 20% deviation in batch size only increases total inventory cost by ~1.6%)."
            ]

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            assert isinstance(res, QuantityDiscountResult)
            headline = f"Order {res.optimal_order_quantity:,.0f} units at ${res.optimal_unit_price:.2f}/unit (Tier {res.optimal_tier_id}) to save ${res.annual_savings:,.2f}/yr ({res.percentage_savings:.1f}%)."
            derivation = (
                f"All-Units Discount Optimization:\n"
                f"  Evaluated each price tier for unconstrained EOQ vs tier breakpoint.\n"
                f"  Winning Tier {res.optimal_tier_id}: Price = ${res.optimal_unit_price:.2f}, Batch Size Q* = {res.optimal_order_quantity:,.0f} units.\n"
                f"  Total Cost = Annual Purchase (${res.optimal_purchase_cost:,.2f}) + Setup (${res.optimal_ordering_cost:,.2f}) + Holding (${res.optimal_holding_cost:,.2f}) = ${res.optimal_total_cost:,.2f}."
            )
            numeric_dict = {
                "Recommended Tier": f"Tier {res.optimal_tier_id}",
                "Optimal Order Quantity": f"{res.optimal_order_quantity:,.0f} units",
                "Unit Purchase Price": f"${res.optimal_unit_price:.2f}",
                "Total Annual Cost": f"${res.optimal_total_cost:,.2f}",
                "Baseline Total Cost (No Discount)": f"${res.baseline_total_cost:,.2f}",
                "Annual Net Savings": f"${res.annual_savings:,.2f} ({res.percentage_savings:.1f}%)"
            }
            insights = [
                f"Discount Trade-off: Increasing order batch from baseline yields substantial purchase cost discounts that outweigh increased carrying costs.",
                f"Cash & Working Capital: Storing {res.optimal_order_quantity:,.0f} units will tie up ${res.optimal_order_quantity * res.optimal_unit_price / 2:,.2f} in average cycle inventory. Ensure warehouse capacity is sufficient."
            ]

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            assert isinstance(res, SafetyStockROPResult)
            headline = f"Set Reorder Point (ROP) to {res.reorder_point:,.1f} units (incorporating {res.safety_stock:,.1f} units of Safety Stock) for a {res.service_level*100:.1f}% Service Level."
            derivation = (
                f"Safety Stock & ROP Formulas:\n"
                f"  Expected Lead Time Demand (mu_LTD) = d * L = {res.mean_demand_per_period:,.1f} * {res.mean_lead_time:.1f} = {res.lead_time_demand_mean:,.1f} units\n"
                f"  Lead Time Demand Std Dev (sigma_LTD) = {res.lead_time_demand_std:,.2f} units ({res.uncertainty_type})\n"
                f"  Z-Score for {res.service_level*100:.1f}% Cycle Service Level = {res.z_score:.4f}\n"
                f"  Safety Stock (SS) = Z * sigma_LTD = {res.z_score:.4f} * {res.lead_time_demand_std:,.2f} = {res.safety_stock:,.2f} units\n"
                f"  Reorder Point (ROP) = mu_LTD + SS = {res.lead_time_demand_mean:,.1f} + {res.safety_stock:,.2f} = {res.reorder_point:,.2f} units"
            )
            numeric_dict = {
                "Reorder Point (ROP)": f"{res.reorder_point:,.1f} units (Order when inventory drops to this level)",
                "Safety Stock Buffer": f"{res.safety_stock:,.1f} units",
                "Expected Lead Time Demand": f"{res.lead_time_demand_mean:,.1f} units",
                "Cycle Service Level (CSL)": f"{res.service_level*100:.1f}%",
                "Stockout Probability per Replenishment Cycle": f"{res.stockout_risk:.1f}%",
                "Uncertainty Scenario": res.uncertainty_type
            }
            if res.fill_rate is not None:
                numeric_dict["Item Fill Rate (Type 2 Service Level)"] = f"{res.fill_rate*100:.2f}%"

            insights = [
                f"Service Protection: A safety stock of {res.safety_stock:.0f} units guarantees that only {res.stockout_risk:.1f}% of replenishment cycles risk a stockout.",
                f"Non-Linear Safety Cost: Increasing service level from 95% (Z=1.645) to 99% (Z=2.326) requires a 41% increase in safety stock."
            ]

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            assert isinstance(res, FixedPeriodReviewResult)
            headline = f"Set Target Order-Up-To Level (S) to {res.target_inventory_level:,.1f} units. Order {res.recommended_order_quantity:,.0f} units today."
            derivation = (
                f"Periodic Review (R, S) Formulation:\n"
                f"  Protection Interval = Review Period ({res.review_period}) + Lead Time ({res.lead_time}) = {res.protection_interval:.1f} {res.time_unit}\n"
                f"  Protection Demand Mean = {res.protection_demand_mean:,.1f} units\n"
                f"  Protection Demand Std Dev = {res.protection_demand_std:,.2f} units\n"
                f"  Safety Stock = {res.safety_stock:,.2f} units\n"
                f"  Target Level S = {res.protection_demand_mean:,.1f} + {res.safety_stock:,.2f} = {res.target_inventory_level:,.2f} units\n"
                f"  Inventory Position = {res.inventory_position:,.1f} units\n"
                f"  Order Quantity Q = max(0, S - IP) = {res.recommended_order_quantity:,.2f} units"
            )
            numeric_dict = {
                "Target Order-Up-To Level (S)": f"{res.target_inventory_level:,.1f} units",
                "Recommended Order Quantity Today": f"{res.recommended_order_quantity:,.0f} units",
                "Safety Stock Buffer": f"{res.safety_stock:,.1f} units",
                "Protection Interval Horizon": f"{res.protection_interval:.1f} {res.time_unit}",
                "Current Inventory Position": f"{res.inventory_position:,.1f} units"
            }
            insights = [
                f"Periodic Review vs Continuous Review: Periodic systems require {res.periodic_ss_multiplier:.2f}x higher safety stock ({res.safety_stock:.0f} vs {res.continuous_ss_equivalent:.0f} units) because uncertainty spans both the review interval and delivery lead time.",
                "Review Cadence: Schedule re-orders strictly every " + f"{res.review_period} {res.time_unit}."
            ]

        elif model_type == InventoryModelType.NEWSVENDOR:
            assert isinstance(res, NewsvendorResult)
            headline = f"Order {res.optimal_order_quantity:,.0f} units for an expected profit of ${res.expected_profit:,.2f} (Critical Fractile: {res.critical_fractile*100:.1f}%)."
            derivation = (
                f"Newsvendor Critical Fractile Formulation:\n"
                f"  Cost of Underage (Cu) = ${res.cost_of_underage:.2f}\n"
                f"  Cost of Overage  (Co) = ${res.cost_of_overage:.2f}\n"
                f"  Critical Fractile (CR) = Cu / (Cu + Co) = ${res.cost_of_underage:.2f} / (${res.cost_of_underage:.2f} + ${res.cost_of_overage:.2f}) = {res.critical_fractile:.4f}\n"
                f"  Z* = norm_ppf({res.critical_fractile:.4f}) = {res.z_score:.4f}\n"
                f"  Optimal Quantity Q* = Mean ({res.mean_demand:,.1f}) + Z* * StdDev ({res.std_demand:,.1f}) = {res.optimal_order_quantity:,.2f} units"
            )
            numeric_dict = {
                "Optimal Order Quantity (Q*)": f"{res.optimal_order_quantity:,.0f} units",
                "Critical Fractile (In-Stock Target)": f"{res.critical_fractile*100:.2f}%",
                "Cost of Underage (Lost Margin Cu)": f"${res.cost_of_underage:,.2f}",
                "Cost of Overage (Excess Loss Co)": f"${res.cost_of_overage:,.2f}",
                "Expected Total Profit": f"${res.expected_profit:,.2f}",
                "Expected Units Sold": f"{res.expected_sales:,.1f} units",
                "Expected Leftover Inventory": f"{res.expected_leftover:,.1f} units",
                "Expected Unmet Demand (Shortage)": f"{res.expected_shortage:,.1f} units",
                "Expected Value of Perfect Information (EVPI)": f"${res.evpi:,.2f}"
            }
            insights = [
                f"Risk Profile: Because Cu (${res.cost_of_underage:.2f}) {'exceeds' if res.cost_of_underage > res.cost_of_overage else 'is less than'} Co (${res.cost_of_overage:.2f}), optimal policy orders {'ABOVE' if res.optimal_order_quantity > (res.mean_demand or 0) else 'BELOW'} average demand to hedge against lost sales margin.",
                f"Value of Better Forecasting: Perfect demand information would yield up to an additional ${res.evpi:,.2f} in profit."
            ]

        # Audit formatting
        audit_lines = [f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.message}" for c in verification.checks]

        # Assemble markdown report
        report_sections = [
            f"# Supply Chain Inventory Decision Report",
            f"**Customer Query:** \"{question}\"\n",
            f"## 1. Executive Recommendation",
            f"> **{headline}**\n",
            f"## 2. Decision Model & Pipeline",
            f"- **Model Selected:** {model_type.value}",
            f"- **Confidence:** {model_id.confidence*100:.0f}%",
            f"- **Identification Logic:** {model_id.reasoning}\n",
            f"## 3. Mathematical Formulation & Calculation",
            f"```text\n{derivation}\n```\n",
            f"## 4. Key Supply Chain Performance Indicators",
            "\n".join([f"- **{k}:** {v}" for k, v in numeric_dict.items()]),
            f"\n## 5. Verification & Mathematical Sanity Audit",
            "\n".join([f"- {line}" for line in audit_lines]),
            f"\n## 6. Managerial & Operational Insights",
            "\n".join([f"- {insight}" for insight in insights])
        ]
        formatted_report = "\n\n".join(report_sections)

        return AgentExplanation(
            headline_recommendation=headline,
            mathematical_derivation=derivation,
            numeric_breakdown=numeric_dict,
            managerial_insights=insights,
            verification_audit=audit_lines,
            formatted_report=formatted_report
        )

    # =================================================================
    # Complete Pipeline Runner
    # =================================================================

    def process_question(self, question: str) -> AgentResponse:
        """
        Executes the entire 7-stage architecture:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        pipeline_trace = []
        pipeline_trace.append(f"[Stage 1 - Input]: Received question: '{question}'")

        # 1. Identify Model
        model_id = self.identify_model(question)
        pipeline_trace.append(f"[Stage 2 - Identify Model]: Classified as '{model_id.selected_model.value}' (Confidence: {model_id.confidence*100:.0f}%)")

        # 2. Extract Inputs
        input_ext = self.extract_inputs(question, model_id.selected_model)
        pipeline_trace.append(f"[Stage 3 - Extract Inputs]: Extracted {len(input_ext.extracted_params)} parameters. Missing: {input_ext.missing_required_params or 'None'}")

        # If critical required parameters are missing, handle gracefully
        if input_ext.missing_required_params:
            missing_str = ", ".join(input_ext.missing_required_params)
            err_msg = f"Cannot calculate {model_id.selected_model.value} because required input(s) are missing: {missing_str}."
            tool_res = ToolExecutionResult(
                model_type=model_id.selected_model,
                tool_name="N/A",
                raw_result=None,
                is_success=False,
                error_message=err_msg
            )
            verification = VerificationSummary(
                model_type=model_id.selected_model,
                all_passed=False,
                checks=[VerificationCheck("Required Inputs", False, f"Missing: {missing_str}", "ERROR")],
                audit_notes=["Query lacked essential numerical parameters."]
            )
            explanation = self.explain(question, model_id, input_ext, tool_res, verification)
            return AgentResponse(
                customer_question=question,
                model_identification=model_id,
                input_extraction=input_ext,
                tool_execution=tool_res,
                verification=verification,
                explanation=explanation,
                execution_pipeline_trace=pipeline_trace
            )

        # 3. Select Tool
        tool_fn, tool_name = self.select_tool(model_id.selected_model)
        pipeline_trace.append(f"[Stage 4 - Select Tool]: Selected Python tool '{tool_name}' in inventory_tools")

        # 4. Calculate
        tool_res = self.calculate(tool_fn, tool_name, model_id.selected_model, input_ext.extracted_params)
        pipeline_trace.append(f"[Stage 5 - Calculate]: Calculation executed ({'Success' if tool_res.is_success else 'Failed'})")

        # 5. Verify
        verification = self.verify(model_id.selected_model, input_ext.extracted_params, tool_res)
        pipeline_trace.append(f"[Stage 6 - Verify]: Audit complete ({len(verification.checks)} checks, All Passed: {verification.all_passed})")

        # 6. Explain
        explanation = self.explain(question, model_id, input_ext, tool_res, verification)
        pipeline_trace.append(f"[Stage 7 - Explain]: Decision report synthesized.")

        return AgentResponse(
            customer_question=question,
            model_identification=model_id,
            input_extraction=input_ext,
            tool_execution=tool_res,
            verification=verification,
            explanation=explanation,
            execution_pipeline_trace=pipeline_trace
        )
