from __future__ import annotations

import random
from dataclasses import dataclass


PROVIDERS = [
    "Nationwide", "HSBC UK", "Barclays", "Halifax", "NatWest",
    "Santander UK", "First Direct", "Virgin Money", "Lloyds Bank",
    "TSB", "Yorkshire BS", "Coventry BS", "Skipton BS", "Leeds BS",
    "Accord Mortgages", "Platform", "BM Solutions", "Clydesdale Bank",
]

TERM_CONFIGS = [
    {"type": "Fixed",   "length": "2 years",  "label": "2 Year Fixed"},
    {"type": "Fixed",   "length": "5 years",  "label": "5 Year Fixed"},
    {"type": "Fixed",   "length": "10 years", "label": "10 Year Fixed"},
    {"type": "Tracker", "length": "2 years",  "label": "2 Year Tracker"},
    {"type": "Tracker", "length": "5 years",  "label": "5 Year Tracker"},
    {"type": "Discount","length": "2 years",  "label": "2 Year Discount"},
]

# Weights: Fixed products are most common in UK market
TERM_WEIGHTS = [30, 30, 10, 12, 8, 10]

LTV_TIERS = [60, 70, 75, 80, 85, 90, 95]
LTV_WEIGHTS = [15, 10, 25, 15, 15, 15, 5]

PRODUCT_SUFFIXES = [
    "Standard", "Premium", "Fee Saver", "New Home", "Remortgage",
    "First-Time Buyer", "Green Mortgage", "Rate Switch",
    "Buy to Let", "Help to Buy", "Shared Ownership", "Exclusive",
    "Flexi", "Offset", "Self-Build",
]

BOOKING_FEES = [0, 0, 499, 995, 999, 999, 1499]  # 0 duplicated to raise frequency

NOTES_POOL = [
    "Available for properties with an EPC rating of A or B.",
    "Includes £250 cashback for first-time buyers.",
    "Fee-free remortgage product for high-scoring applicants.",
    "Available to existing customers only.",
    "Requires a minimum loan of £150,000.",
    "Maximum loan size of £500,000.",
    "Not available for new build properties.",
    "Includes free standard valuation.",
    "Early repayment charge of 2% applies in year 1, 1% in year 2.",
    "Available for purchase and remortgage.",
    "Offers overpayment facility of up to 10% per year without penalty.",
    "Portable to a new property subject to affordability checks.",
    "Rate guaranteed for 90 days from application.",
    "Part of the April 2026 rate reduction programme.",
    "Requires a current account with the lender.",
    "Available for interest-only with minimum 50% equity.",
    "Subject to a completion deadline of 30 June 2026.",
    "Market-leading rate for low LTV home movers.",
    "Higher LTV option with no upfront booking fee.",
    "Rate reduced by 0.15% today as part of spring pricing update.",
    "Cashback of £500 on completion.",
    "Free legal fees for remortgage customers.",
    "Available for shared ownership purchases from 25% share.",
    "Self-employed applicants require two years of accounts.",
    "Minimum property value of £75,000.",
    "Capital repayment only; interest-only not available.",
    "Rate stepped down after year 2 on tracker.",
    "No early repayment charges after the fixed period ends.",
    "New build only; standard properties not eligible.",
    "Includes building insurance for first year.",
]

# Rate spread above base rate: (min_spread, max_spread) indexed by LTV tier
# Lower LTV = tighter spread; higher LTV = wider spread
_LTV_SPREAD = {
    60: (0.25, 0.75),
    70: (0.50, 1.00),
    75: (0.75, 1.30),
    80: (1.00, 1.60),
    85: (1.25, 1.90),
    90: (1.60, 2.40),
    95: (2.00, 3.00),
}

# 5yr fixed typically prices slightly differently vs 2yr in current market
_TERM_ADJUSTMENT = {
    "2 years":  0.00,
    "5 years":  0.10,   # modest term premium in current inverted environment
    "10 years": 0.25,
    # Trackers and Discounts use different logic below
}


@dataclass
class MortgageProduct:
    report_date: str
    base_rate: float
    provider: str
    mortgage_name: str
    interest_rate: float
    maximum_ltv: int
    term_type: str
    length: str
    booking_fee: int
    aprc: float
    notes: str

    def to_input_text(self) -> str:
        fee_str = f"£{self.booking_fee:,}" if self.booking_fee > 0 else "£0"
        return (
            f"Report date: {self.report_date}\n"
            f"Base rate: {self.base_rate}%\n"
            f"Provider: {self.provider}\n"
            f"Mortgage name: {self.mortgage_name}\n"
            f"Interest rate: {self.interest_rate}%\n"
            f"Maximum LTV: {self.maximum_ltv}%\n"
            f"Term type: {self.term_type}\n"
            f"Length: {self.length}\n"
            f"Booking fee: {fee_str}\n"
            f"APRC: {self.aprc}%\n"
            f"Notes: {self.notes}"
        )


def _sample_rate(base_rate: float, ltv: int, term_config: dict, rng: random.Random) -> float:
    lo, hi = _LTV_SPREAD[ltv]
    if term_config["type"] == "Fixed":
        adj = _TERM_ADJUSTMENT.get(term_config["length"], 0.0)
        spread = rng.uniform(lo + adj, hi + adj)
    elif term_config["type"] == "Tracker":
        spread = rng.uniform(0.40, 1.20)
    else:  # Discount
        spread = rng.uniform(lo - 0.10, hi - 0.10)
    return round(base_rate + spread, 2)


def _sample_aprc(interest_rate: float, term_type: str, rng: random.Random) -> float:
    if term_type == "Fixed":
        aprc = interest_rate + rng.uniform(0.80, 1.60)
    else:
        aprc = interest_rate + rng.uniform(0.50, 1.20)
    return round(aprc, 2)


def _build_name(provider: str, term_config: dict, suffix: str) -> str:
    return f"{term_config['label']} {suffix}"


def sample_products(
    num_products: int,
    base_rate: float,
    report_date: str,
    seed: int = 42,
) -> list[MortgageProduct]:
    rng = random.Random(seed)
    seen_names: set[str] = set()
    products: list[MortgageProduct] = []

    attempts = 0
    while len(products) < num_products and attempts < num_products * 10:
        attempts += 1
        provider = rng.choice(PROVIDERS)
        term_config = rng.choices(TERM_CONFIGS, weights=TERM_WEIGHTS, k=1)[0]
        ltv = rng.choices(LTV_TIERS, weights=LTV_WEIGHTS, k=1)[0]
        suffix = rng.choice(PRODUCT_SUFFIXES)
        booking_fee = rng.choice(BOOKING_FEES)
        notes = rng.choice(NOTES_POOL)

        name = _build_name(provider, term_config, suffix)
        dedup_key = (provider, name, ltv)
        if dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)

        interest_rate = _sample_rate(base_rate, ltv, term_config, rng)
        aprc = _sample_aprc(interest_rate, term_config["type"], rng)

        products.append(MortgageProduct(
            report_date=report_date,
            base_rate=base_rate,
            provider=provider,
            mortgage_name=name,
            interest_rate=interest_rate,
            maximum_ltv=ltv,
            term_type=term_config["type"],
            length=term_config["length"],
            booking_fee=booking_fee,
            aprc=aprc,
            notes=notes,
        ))

    return products
