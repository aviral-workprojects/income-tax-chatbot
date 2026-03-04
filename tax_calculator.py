"""
tax_calculator.py
-----------------
Implements Indian income tax calculation for both the Old and New tax regimes
as per the Finance Act 2024 / Assessment Year 2024-25 and 2025-26.

This module is PURELY computational – it has no UI code.
app.py calls these functions and renders the results.

Key assumptions
---------------
- Individual taxpayer (not HUF / firm / company).
- Age below 60 (Senior / super-senior citizen brackets are noted in comments).
- Standard deduction and basic exemption are applied automatically.
- Education cess (4%) is applied on the base tax.
- Surcharge is NOT calculated here (applies above ₹50 lakh; add if needed).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaxSlab:
    """Represents a single tax slab."""
    lower: float        # lower bound (inclusive), ₹
    upper: float        # upper bound (inclusive), ₹  (use float('inf') for topmost)
    rate: float         # tax rate as a fraction, e.g. 0.05 for 5%


@dataclass
class SlabBreakdown:
    """Tax applied within a single slab."""
    slab_label: str     # human-readable range, e.g. "₹3,00,001 – ₹6,00,000"
    taxable_in_slab: float
    rate_pct: float     # percentage, e.g. 5.0
    tax_in_slab: float


@dataclass
class TaxResult:
    """Complete tax computation result."""
    regime: str
    gross_income: float
    deductions: float
    taxable_income: float
    slab_breakdown: List[SlabBreakdown] = field(default_factory=list)
    base_tax: float = 0.0
    rebate_87a: float = 0.0
    tax_after_rebate: float = 0.0
    cess: float = 0.0
    total_tax: float = 0.0
    effective_rate_pct: float = 0.0


# ---------------------------------------------------------------------------
# Tax slabs (AY 2025-26, i.e. FY 2024-25)
# ---------------------------------------------------------------------------

# Old regime slabs – individual below 60 years
OLD_REGIME_SLABS: List[TaxSlab] = [
    TaxSlab(lower=0,         upper=250_000,      rate=0.00),
    TaxSlab(lower=250_001,   upper=500_000,      rate=0.05),
    TaxSlab(lower=500_001,   upper=1_000_000,    rate=0.20),
    TaxSlab(lower=1_000_001, upper=float("inf"), rate=0.30),
]

# New regime slabs (post-Budget 2023, applicable from AY 2024-25 onwards)
NEW_REGIME_SLABS: List[TaxSlab] = [
    TaxSlab(lower=0,         upper=300_000,      rate=0.00),
    TaxSlab(lower=300_001,   upper=600_000,      rate=0.05),
    TaxSlab(lower=600_001,   upper=900_000,      rate=0.10),
    TaxSlab(lower=900_001,   upper=1_200_000,    rate=0.15),
    TaxSlab(lower=1_200_001, upper=1_500_000,    rate=0.20),
    TaxSlab(lower=1_500_001, upper=float("inf"), rate=0.30),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard deduction (applicable to salaried individuals under both regimes)
STANDARD_DEDUCTION_OLD = 50_000   # ₹50,000 under old regime
STANDARD_DEDUCTION_NEW = 75_000   # ₹75,000 under new regime (Budget 2024)

# Rebate under Section 87A
REBATE_87A_OLD_LIMIT = 500_000    # taxable income up to ₹5 lakh
REBATE_87A_OLD_MAX   = 12_500     # maximum rebate

REBATE_87A_NEW_LIMIT = 700_000    # taxable income up to ₹7 lakh (new regime)
REBATE_87A_NEW_MAX   = 25_000     # maximum rebate

# Education and health cess on total tax
CESS_RATE = 0.04   # 4%


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_old_regime(gross_income: float, other_deductions: float = 0.0) -> TaxResult:
    """
    Compute income tax under the Old Tax Regime.

    Parameters
    ----------
    gross_income       : float – total annual income (₹)
    other_deductions   : float – deductions such as 80C, 80D, HRA, etc. (₹)
                                 The user can enter this in the UI.

    Returns
    -------
    TaxResult instance with a full slab-wise breakdown.
    """
    total_deductions = STANDARD_DEDUCTION_OLD + other_deductions
    taxable_income = max(0.0, gross_income - total_deductions)

    slab_breakdown, base_tax = _apply_slabs(taxable_income, OLD_REGIME_SLABS)

    # Section 87A rebate
    rebate = 0.0
    if taxable_income <= REBATE_87A_OLD_LIMIT:
        rebate = min(base_tax, REBATE_87A_OLD_MAX)

    tax_after_rebate = max(0.0, base_tax - rebate)
    cess = round(tax_after_rebate * CESS_RATE, 2)
    total_tax = round(tax_after_rebate + cess, 2)
    effective_rate = round((total_tax / gross_income) * 100, 2) if gross_income > 0 else 0.0

    return TaxResult(
        regime="Old Regime",
        gross_income=gross_income,
        deductions=total_deductions,
        taxable_income=taxable_income,
        slab_breakdown=slab_breakdown,
        base_tax=round(base_tax, 2),
        rebate_87a=round(rebate, 2),
        tax_after_rebate=round(tax_after_rebate, 2),
        cess=cess,
        total_tax=total_tax,
        effective_rate_pct=effective_rate,
    )


def compute_new_regime(gross_income: float) -> TaxResult:
    """
    Compute income tax under the New Tax Regime.

    Under the new regime, most deductions are not available (except standard
    deduction).  No 'other_deductions' parameter is accepted here.

    Parameters
    ----------
    gross_income : float – total annual income (₹)

    Returns
    -------
    TaxResult instance with a full slab-wise breakdown.
    """
    deductions = STANDARD_DEDUCTION_NEW
    taxable_income = max(0.0, gross_income - deductions)

    slab_breakdown, base_tax = _apply_slabs(taxable_income, NEW_REGIME_SLABS)

    # Section 87A rebate (new regime: up to ₹7 lakh taxable income)
    rebate = 0.0
    if taxable_income <= REBATE_87A_NEW_LIMIT:
        rebate = min(base_tax, REBATE_87A_NEW_MAX)

    tax_after_rebate = max(0.0, base_tax - rebate)
    cess = round(tax_after_rebate * CESS_RATE, 2)
    total_tax = round(tax_after_rebate + cess, 2)
    effective_rate = round((total_tax / gross_income) * 100, 2) if gross_income > 0 else 0.0

    return TaxResult(
        regime="New Regime",
        gross_income=gross_income,
        deductions=deductions,
        taxable_income=taxable_income,
        slab_breakdown=slab_breakdown,
        base_tax=round(base_tax, 2),
        rebate_87a=round(rebate, 2),
        tax_after_rebate=round(tax_after_rebate, 2),
        cess=cess,
        total_tax=total_tax,
        effective_rate_pct=effective_rate,
    )


def compare_regimes(
    gross_income: float, old_regime_deductions: float = 0.0
) -> Tuple[TaxResult, TaxResult]:
    """
    Compute tax under both regimes and return a tuple (old_result, new_result).

    Convenience function used by the Regime Comparison page.
    """
    old = compute_old_regime(gross_income, other_deductions=old_regime_deductions)
    new = compute_new_regime(gross_income)
    return old, new


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_slabs(
    taxable_income: float, slabs: List[TaxSlab]
) -> Tuple[List[SlabBreakdown], float]:
    """
    Apply progressive tax slabs to the taxable income.

    Returns (list_of_SlabBreakdown, total_base_tax).
    """
    breakdown: List[SlabBreakdown] = []
    total_tax = 0.0
    remaining = taxable_income

    for slab in slabs:
        if remaining <= 0:
            break

        # Width of this slab
        slab_width = (
            slab.upper - slab.lower + 1
            if slab.upper != float("inf")
            else remaining
        )

        # Income that falls within this slab
        amount_in_slab = min(remaining, slab_width)
        tax_in_slab = amount_in_slab * slab.rate

        # Always include the slab in the breakdown, even if tax is 0,
        # so the student can see the full slab table.
        upper_label = (
            f"₹{int(slab.upper):,}"
            if slab.upper != float("inf")
            else "and above"
        )
        breakdown.append(
            SlabBreakdown(
                slab_label=f"₹{int(slab.lower):,} – {upper_label}",
                taxable_in_slab=round(amount_in_slab, 2),
                rate_pct=slab.rate * 100,
                tax_in_slab=round(tax_in_slab, 2),
            )
        )

        total_tax += tax_in_slab
        remaining -= amount_in_slab

    return breakdown, total_tax


# ---------------------------------------------------------------------------
# Utility – format currency for display
# ---------------------------------------------------------------------------

def format_inr(amount: float) -> str:
    """Return a ₹ formatted string with Indian number formatting."""
    # Python's built-in formatting does not support Indian grouping,
    # so we implement a simple version here.
    amount = round(amount, 2)
    integer_part = int(amount)
    decimal_part = f"{amount - integer_part:.2f}"[1:]  # ".xx"

    s = str(integer_part)
    if len(s) <= 3:
        return f"₹{s}{decimal_part}"

    # Last three digits, then groups of two
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]

    # Remove a leading comma if the last group had only one digit
    result = result.lstrip(",")
    return f"₹{result}{decimal_part}"
