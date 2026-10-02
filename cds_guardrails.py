"""
=============================================================================
Module: cds_guardrails.py
Description: Production-ready SAP Clean Core Guardrails, Syntax Linter &
             DDIC Synonym Mapper.
             Validates ABAP Core Data Services (CDS) source definitions against
             SAP Clean Core extensibility principles:
               - Gate 1: Intercepts raw DDIC database table access (VBAK, VBAP, etc.)
               - Gate 2: Enforces modern CDS 'define view entity' syntax and
                         rejects legacy 'define view' & obsolete SQL view names.
               - DDIC Synonym & Field Mapper: Resolves legacy ECC technical fields
                 (e.g., KUNNR, NAME1, VBELN, NETWR) to modern S/4HANA VDM elements.
Author: SAP S/4HANA Cloud & Clean Core Architecture Engineering
=============================================================================
"""

import re
from typing import Dict, List, Any, Optional, Set


# =============================================================================
# DDIC SYNONYM DICTIONARY (ECC -> S/4HANA VDM FIELD MAPPINGS)
# =============================================================================

DDIC_SYNONYM_MAP: Dict[str, str] = {
    # Customer Master / KNA1
    "kunnr": "customer",
    "name1": "customername",
    "name2": "customername2",
    "ort01": "cityname",
    "land1": "country",
    "pstlz": "postalcode",
    "stras": "streetname",
    "regio": "region",
    "telf1": "phonenumber",
    "telfx": "faxnumber",
    "ktokd": "customeraccountgroup",
    "brsch": "industry",
    "anred": "title",

    # Sales Order / VBAK & VBAP
    "vbeln": "salesorder",
    "posnr": "salesorderitem",
    "audat": "documentdate",
    "erdat": "creationdate",
    "ernam": "createdbyuser",
    "netwr": "totalnetamount",
    "waerk": "transactioncurrency",
    "vkorg": "salesorganization",
    "vtweg": "distributionchannel",
    "spart": "organizationdivision",
    "kunnr_ana": "soldtoparty",
    "matnr": "product",
    "maktx": "productdescription",
    "kwmeng": "orderquantity",
    "vrkme": "orderquantityunit",
    "netpr": "netpriceamount",
    "werks": "plant",
    "lgort": "storagelocation",
    "kpein": "conditionpricingunit",
    "kmein": "conditionunit",
    "charg": "batch",

    # Billing / VBRK & VBRP
    "fkart": "billingdocumenttype",
    "fkdat": "billingdocumentdate",
    "kunrg": "payerparty",
    "aubel": "salesdocument",
    "aupos": "salesdocumentitem",

    # Purchasing / EKKO & EKPO
    "ebeln": "purchaseorder",
    "ebelp": "purchaseorderitem",
    "lifnr": "supplier",
    "bsart": "purchaseordertype",
    "bukrs": "companycode",
    "menge": "orderquantity",
    "meins": "purchaseorderquantityunit",
    "peinh": "priceunit",
    "infnr": "purchasinginforecord",

    # Vendor Master / LFA1
    "sortl": "searchterm1",

    # Finance / BKPF & BSEG / ACDOCA
    "belnr": "accountingdocument",
    "gjahr": "fiscalyear",
    "blart": "documenttype",
    "bldat": "documentdate",
    "budat": "postingdate",
    "hkont": "glaccount",
    "wrbtr": "amountintransactioncurrency",
    "dmbtr": "amountincompanycodecurrency",
    "koart": "accounttype",
    "shkzg": "debitcreditcode",
    "kostl": "costcenter",
    "prctr": "profitcenter",
    "gsber": "businessarea",
    "segment": "segment"
}


def normalize_field_name(field_str: str) -> str:
    """
    Normalizes a DDIC field name or technical abbreviation to standard VDM element name.
    Strips table prefixes (e.g. KNA1-KUNNR, kna1.kunnr, vbak~vbeln) and looks up
    synonyms in DDIC_SYNONYM_MAP.

    Examples:
        "KNA1-KUNNR"   -> "customer"
        "VBAK.VBELN"   -> "salesorder"
        "NAME1"        -> "customername"
        "CustomerName" -> "customername"
    """
    if not field_str:
        return ""

    raw = field_str.strip()
    if "-" in raw:
        raw = raw.split("-")[-1]
    elif "." in raw:
        raw = raw.split(".")[-1]
    elif "~" in raw:
        raw = raw.split("~")[-1]

    raw_clean = re.sub(r"[^A-Za-z0-9_]", "", raw).strip().lower()
    return DDIC_SYNONYM_MAP.get(raw_clean, raw_clean)


def parse_abap_snippet(snippet: str) -> Dict[str, Any]:
    """
    Parses pasted raw ABAP code or SQL queries to extract:
    - Raw DDIC database tables queried (FROM / JOIN)
    - Raw field names selected
    - Normalized S/4HANA VDM field names
    - Recommended released VDM replacement views
    """
    if not snippet or not snippet.strip():
        return {
            "raw_tables": [],
            "raw_fields": [],
            "normalized_fields": [],
            "vdm_replacements": []
        }

    # Clean comments
    clean = re.sub(r"//.*$", "", snippet, flags=re.MULTILINE)
    clean = re.sub(r"/\*[\s\S]*?\*/", "", clean)
    clean = re.sub(r"--.*$", "", clean, flags=re.MULTILINE)

    # 1. Extract tables
    table_matches = re.findall(r"\b(?:from|join)\s+([A-Za-z0-9_/\.]+)", clean, re.IGNORECASE)
    raw_tables = []
    vdm_replacements = []
    reserved_kw = {"SELECT", "WHERE", "GROUP", "ORDER", "HAVING", "AS", "ON", "INNER", "LEFT", "RIGHT", "OUTER", "INTO", "TABLE"}
    for t in table_matches:
        t_clean = t.strip().upper()
        if t_clean not in reserved_kw and t_clean not in raw_tables:
            raw_tables.append(t_clean)
            if t_clean in CleanCoreGuardrails.RAW_TABLE_REPLACEMENTS:
                vdm_replacements.append(CleanCoreGuardrails.RAW_TABLE_REPLACEMENTS[t_clean])

    # 2. Extract fields
    raw_fields = []
    select_match = re.search(r"\bselect\s+(?:distinct\s+)?([\s\S]+?)\s+\bfrom\b", clean, re.IGNORECASE)
    if select_match:
        fields_part = select_match.group(1)
        field_tokens = [f.strip() for f in re.split(r"[, \n\r\t]+", fields_part) if f.strip()]
        for tok in field_tokens:
            tok_clean = tok.strip(";,()")
            if tok_clean.upper() not in ["SELECT", "DISTINCT", "AS", "FROM", "*", "COUNT", "SUM", "AVG", "MIN", "MAX"] and tok_clean:
                raw_fields.append(tok_clean)
    else:
        # Fallback: parse comma or whitespace separated tokens
        tokens = [t.strip() for t in re.split(r"[, \n\r\t]+", clean) if t.strip()]
        for tok in tokens:
            tok_clean = tok.strip(";,()")
            if len(tok_clean) >= 2 and tok_clean.upper() not in reserved_kw and tok_clean.upper() not in raw_tables:
                raw_fields.append(tok_clean)

    # Deduplicate raw fields
    raw_fields = list(dict.fromkeys(raw_fields))
    normalized_fields = [normalize_field_name(f) for f in raw_fields if normalize_field_name(f)]
    normalized_fields = list(dict.fromkeys(normalized_fields))

    return {
        "raw_tables": raw_tables,
        "raw_fields": raw_fields,
        "normalized_fields": normalized_fields,
        "vdm_replacements": vdm_replacements
    }


# =============================================================================
# CLEAN CORE GUARDRAILS CLASS
# =============================================================================

class CleanCoreGuardrails:
    """
    Enforces SAP Clean Core compliance gates on ABAP CDS View DDL definitions.
    Prevents core modifications, raw table dependencies, and obsolete syntax.
    """

    DDIC_SYNONYM_MAP = DDIC_SYNONYM_MAP
    normalize_field_name = staticmethod(normalize_field_name)
    parse_abap_snippet = staticmethod(parse_abap_snippet)

    # Primary Raw DDIC Database Tables strictly forbidden in Clean Core
    RAW_TABLE_REPLACEMENTS: Dict[str, str] = {
        # Sales & Distribution (SD)
        "VBAK": "I_SalesOrder",
        "VBAP": "I_SalesOrderItem",
        "VBRK": "I_BillingDocument",
        "VBRP": "I_BillingDocumentItem",
        "LIKP": "I_DeliveryDocument",
        "LIPS": "I_DeliveryDocumentItem",
        "VBKD": "I_SalesOrderHeaderPartner",
        
        # Financials & Controlling (FI/CO)
        "BKPF": "I_JournalEntry",
        "BSEG": "I_JournalEntryItem",
        "ACDOCA": "I_JournalEntryItem",
        "SKA1": "I_GLAccount",
        "SKAT": "I_GLAccountText",
        "KNA1": "I_Customer",
        "LFA1": "I_Supplier",
        "BUT000": "I_BusinessPartner",

        # Materials Management & Sourcing (MM/PUR)
        "EKKO": "I_PurchaseOrder",
        "EKPO": "I_PurchaseOrderItem",
        "MARA": "I_Product",
        "MAKT": "I_ProductDescription",
        "MARC": "I_ProductPlant",
        "MARD": "I_ProductStorageLocation",
        "MSEG": "I_MaterialDocumentItem",
        "MKPF": "I_MaterialDocumentHeader",
        "EBAN": "I_PurchaseRequisition",
        "EBKN": "I_PurchaseRequisitionAccountAssignment"
    }

    @staticmethod
    def validate_code(code_str: str) -> Dict[str, Any]:
        """
        Validates an ABAP CDS definition against Clean Core Gate 1 and Gate 2 rules.

        Args:
            code_str: Complete or partial ABAP CDS DDL definition string.

        Returns:
            Dictionary formatted as:
            {
                "is_valid": bool,
                "violations": List[str]
            }
        """
        violations: List[str] = []

        if not code_str or not code_str.strip():
            return {
                "is_valid": False,
                "violations": ["Input code is empty or whitespace."]
            }

        # Remove line and block comments for accurate token parsing
        clean_code = re.sub(r"//.*$", "", code_str, flags=re.MULTILINE)
        clean_code = re.sub(r"/\*[\s\S]*?\*/", "", clean_code)
        clean_code = re.sub(r"--.*$", "", clean_code, flags=re.MULTILINE)

        # Strip string literals ('...') so labels and descriptions don't trigger false positives
        clean_code_no_strings = re.sub(r"'[^']*'", "''", clean_code)

        # =====================================================================
        # GATE 1: RAW TABLE ACCESS INTERCEPTOR
        # =====================================================================
        table_ref_pattern = re.compile(
            r"\b(?:from|join|association\s+(?:\[.*?\]\s+)?to|composition\s+(?:\[.*?\]\s+)?of)\s+([A-Za-z0-9_/\.]+)",
            re.IGNORECASE
        )

        detected_raw_tables: Set[str] = set()
        for match in table_ref_pattern.finditer(clean_code_no_strings):
            target_symbol = match.group(1).strip().upper()
            if target_symbol in CleanCoreGuardrails.RAW_TABLE_REPLACEMENTS:
                detected_raw_tables.add(target_symbol)

        for raw_tbl, vdm_view in CleanCoreGuardrails.RAW_TABLE_REPLACEMENTS.items():
            direct_pattern = re.compile(rf"\bfrom\s+{raw_tbl}\b|\bjoin\s+{raw_tbl}\b", re.IGNORECASE)
            if direct_pattern.search(clean_code_no_strings):
                detected_raw_tables.add(raw_tbl)

        for raw_tbl in sorted(detected_raw_tables):
            vdm_replacement = CleanCoreGuardrails.RAW_TABLE_REPLACEMENTS.get(raw_tbl, "standard released VDM view")
            violations.append(
                f"[Gate 1: Raw Table Violation] Direct access to raw DDIC database table '{raw_tbl}' detected. "
                f"Clean Core strictly forbids querying raw SAP database tables. "
                f"Remediation: Query released SAP VDM Interface View '{vdm_replacement}' instead."
            )

        # =====================================================================
        # GATE 2: MODERN VIEW ENTITY CHECKER
        # =====================================================================
        legacy_view_match = re.search(
            r"\bdefine\s+(?:root\s+)?view\s+(?!entity\b)([A-Za-z0-9_]+)",
            clean_code,
            re.IGNORECASE
        )
        if legacy_view_match:
            view_name = legacy_view_match.group(1)
            violations.append(
                f"[Gate 2: Modern View Entity Violation] Legacy 'define view {view_name}' syntax detected. "
                f"SAP Clean Core mandates modern 'define view entity {view_name}' syntax without obsolete DDIC SQL views. "
                f"Remediation: Upgrade syntax to 'define view entity {view_name}' and remove any @AbapCatalog.sqlViewName."
            )

        sql_view_match = re.search(
            r"@AbapCatalog\.sqlViewName\s*:\s*['\"]([^'\"]+)['\"]",
            clean_code,
            re.IGNORECASE
        )
        if sql_view_match:
            sql_name = sql_view_match.group(1)
            violations.append(
                f"[Gate 2: Obsolete SQL View Name Violation] Obsolete annotation '@AbapCatalog.sqlViewName: '{sql_name}'' detected. "
                f"Modern CDS View Entities do not generate classic DDIC database views. "
                f"Remediation: Remove @AbapCatalog.sqlViewName completely when using 'define view entity'."
            )

        is_modern_declaration = bool(re.search(
            r"\b(?:define\s+(?:root\s+)?view\s+entity|define\s+extension\s+view\s+entity|extend\s+view\s+entity|define\s+table\s+function|define\s+custom\s+entity)\b",
            clean_code,
            re.IGNORECASE
        ))
        if not is_modern_declaration and not legacy_view_match:
            violations.append(
                "[Gate 2: Missing Modern Declaration] No valid modern CDS definition found. "
                "Code must begin with 'define view entity <Name>', 'define extension view entity <Name> extends <Base>', or 'extend view entity <Base> with <Name>'."
            )

        return {
            "is_valid": len(violations) == 0,
            "violations": violations
        }

    @staticmethod
    def get_remediation_summary(violations: List[str]) -> str:
        """Formats violations into a human-readable Clean Core checklist."""
        if not violations:
            return "[Clean Core Compliance Check Passed: Zero Violations]"
        
        lines = [f"Clean Core Compliance Violations Found ({len(violations)}):"]
        for idx, v in enumerate(violations, 1):
            lines.append(f"  {idx}. {v}")
        return "\n".join(lines)
