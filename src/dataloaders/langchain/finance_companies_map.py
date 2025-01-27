from enum import Enum

# Mapping for company names to ticker symbols
COMPANY_MAP = {
    "3m": "MMM",
    "activisionblizzard": "ATVI",
    "adobe": "ADBE",
    "aes": "AES",
    "amazon": "AMZN",
    "amcor": "AMCR",
    "amd": "AMD",
    "americanexpress": "AXP",
    "americanwaterworks": "AWK",
    "bestbuy": "BBY",
    "block": "SQ",
    "boeing": "BA",
    "cocacola": "KO",
    "corning": "GLW",
    "costco": "COST",
    "cvshealth": "CVS",
    "footlocker": "FL",
    "generalmills": "GIS",
    "johnson_johnson": "JNJ",
    "jpmorgan": "JPM",
    "kraftheinz": "KHC",
    "lockheedmartin": "LMT",
    "mgmresorts": "MGM",
    "microsoft": "MSFT",
    "netflix": "NFLX",
    "nike": "NKE",
    "paypal": "PYPL",
    "pepsico": "PEP",
    "pfizer": "PFE",
    "ultabeauty": "ULTA",
    "verizon": "VZ",
    "walmart": "WMT",
}


class FilingItems10k(Enum):
    """Filing items for SEC 10K filings."""

    ITEM_1 = "Item 1"
    ITEM_1A = "Item 1A"
    ITEM_1B = "Item 1B"
    ITEM_2 = "Item 2"
    ITEM_3 = "Item 3"
    ITEM_4 = "Item 4"
    ITEM_5 = "Item 5"
    ITEM_6 = "Item 6"
    ITEM_7 = "Item 7"
    ITEM_7A = "Item 7A"
    ITEM_8 = "Item 8"
    ITEM_9 = "Item 9"
    ITEM_9A = "Item 9A"
    ITEM_9B = "Item 9B"
    ITEM_10 = "Item 10"
    ITEM_11 = "Item 11"
    ITEM_12 = "Item 12"
    ITEM_13 = "Item 13"
    ITEM_14 = "Item 14"
    ITEM_15 = "Item 15"
    ITEM_16 = "Item 16"


class FilingItems10q(Enum):
    """Filing items for SEC 10Q filings."""

    ITEM_1 = "Item 1"
    ITEM_1A = "Item 1A"
    ITEM_2 = "Item 2"
    ITEM_3 = "Item 3"
    ITEM_4 = "Item 4"
    ITEM_5 = "Item 5"
    ITEM_6 = "Item 6"
