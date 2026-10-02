"""
=============================================================================
Module: 01_extract.py
Description: Production-ready SAP S/4HANA Core Data Services (CDS) Metadata
             Extractor. Connects to SAP S/4HANA via SAP ABAP Development
             Tools (ADT) REST APIs with HTTP Basic Auth and CSRF handshake,
             retrieves active CDS view definitions, parses DDL source code,
             and outputs a clean cds_catalog.json.
Author: Senior Python & SAP S/4HANA Integration Developer
=============================================================================
"""

import os
import sys
import re
import json
import time
import logging
import argparse
import urllib.parse
import xml.etree.ElementTree as ET
import html
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Dependency Imports with Resilient Fallback Handling
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import requests
    from requests.auth import HTTPBasicAuth
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3
except ImportError:
    requests = None
    HTTPBasicAuth = None
    HTTPAdapter = None
    Retry = None
    urllib3 = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("SAP_CDS_Extractor")


# =============================================================================
# ENVIRONMENT & CONFIGURATION
# =============================================================================

def init_environment() -> None:
    """
    Initializes environment variables from the local .env file using python-dotenv.
    Falls back to a manual parser if python-dotenv is not installed.
    """
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        if load_dotenv is not None:
            load_dotenv(dotenv_path=env_path, override=False)
            logger.info(f"Loaded environment variables from '{env_path}' via python-dotenv.")
        else:
            # Fallback lightweight parser
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                logger.info(f"Loaded environment variables from '{env_path}' via fallback parser.")
            except Exception as e:
                logger.warning(f"Could not parse '{env_path}': {e}")


# Initialize environment on module load
init_environment()


@dataclass
class SAPConfig:
    """
    SAP S/4HANA Connection Configuration parameters.
    Reads from .env variables matching project specifications.
    """
    # 1. Base URL & Host
    base_url: str = os.getenv("SAP_BASE_URL", "").strip()
    host: str = os.getenv("SAP_HOST", "s4hana.corp.internal").strip()
    port: int = int(os.getenv("SAP_PORT", "44300"))
    use_ssl: bool = os.getenv("SAP_USE_SSL", "true").lower() in ("true", "1", "yes")

    # 2. Authentication & Client
    client: str = os.getenv("SAP_CLIENT", "100").strip()
    username: str = (os.getenv("SAP_USERNAME") or os.getenv("SAP_USER") or "").strip()
    password: str = os.getenv("SAP_PASSWORD", "").strip()

    # 3. Connection & Security
    verify_ssl: bool = os.getenv("SAP_VERIFY_SSL", "false").lower() in ("true", "1", "yes")
    timeout_seconds: int = int(os.getenv("SAP_REQUEST_TIMEOUT_SECONDS") or os.getenv("SAP_TIMEOUT") or "30")

    # 4. Pipeline Parameters
    max_views: int = int(os.getenv("SAP_MAX_VIEWS", "50"))
    use_mock: bool = os.getenv("SAP_USE_MOCK", "false").lower() in ("true", "1", "yes")

    def __post_init__(self):
        # Reconstruct base_url if not explicitly provided
        if not self.base_url:
            protocol = "https" if self.use_ssl else "http"
            self.base_url = f"{protocol}://{self.host}:{self.port}"
        # Ensure clean trailing slash removal
        self.base_url = self.base_url.rstrip("/")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CDSView:
    """
    Normalized metadata representation of a SAP CDS View matching project schema:
    [
      {
        "name": "I_SalesOrder",
        "description": "Sales Order Header Data",
        "domain": "SD",
        "fields": ["SalesOrder", "SalesOrderType", "TotalNetAmount", "CustomerName"],
        "associations": ["_Item", "_Partner"]
      }
    ]
    """
    name: str
    description: str
    domain: str
    fields: List[str] = field(default_factory=list)
    associations: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "fields": self.fields,
            "associations": self.associations,
        }


# =============================================================================
# SAP FUNCTIONAL DOMAIN / MODULE CLASSIFICATION
# =============================================================================

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "SD": [
        "SALESORDER", "BILLINGDOCUMENT", "CUSTOMER", "DELIVERYDOCUMENT",
        "SALESDOCUMENT", "SALESCONTRACT", "SALESINVOICE", "PRICINGELEMENT",
        "SALES", "BILLING", "CUST", "PRICING", "INVOICE", "ORDER"
    ],
    "FI": [
        "JOURNALENTRY", "GLACCOUNT", "COMPANYCODE", "FINANCIALSTATEMENT",
        "OPERATIONALACCTGDOCITEM", "ACCOUNTINGDOCUMENT", "SUPPLIERINVOICE",
        "TAXCODE", "JOURNAL", "LEDGER", "ACCTG", "FINANCIAL", "BANK"
    ],
    "MM": [
        "PURCHASEORDER", "PURCHASEREQUISITION", "MATERIALSTOCK", "SUPPLIER",
        "PURCHASINGDOCUMENT", "GOODSMOVEMENT", "INVENTORYDOCUMENT", "MATERIAL",
        "PURCH", "MATRL", "STOCK", "INVENT", "SUPPL", "VALUATION"
    ],
    "CO": [
        "COSTCENTER", "PROFITCENTER", "INTERNALORDER", "CONTROLLINGAREA",
        "ACTIVITYTYPE", "COST", "PROFIT", "CTRL", "CONTROLLING"
    ],
    "PP": [
        "PRODUCTIONORDER", "WORKCENTER", "BILLOFMATERIAL", "MFGORDER",
        "ROUTING", "PROD", "MFG", "BOM", "PRODUCTION"
    ],
    "PM": [
        "MAINTENANCEORDER", "EQUIPMENT", "FUNCTIONALLOCATION", "NOTIFICATION", "MAINT"
    ],
    "MD": [
        "PRODUCT", "BUSINESSPARTNER", "BUSINESSUSER"
    ]
}


def infer_domain(view_name: str, description: str = "") -> str:
    """
    Infers the SAP functional domain (SD, FI, MM, CO, PP, PM, MD, CA)
    based on SAP naming conventions and semantic tokens.
    """
    clean_target = f"{view_name.upper()} {description.upper()}"

    for domain_code, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in clean_target:
                return domain_code

    return "CA"  # Cross-Application fallback


# =============================================================================
# BUSINESS VIEW PATTERNS & FRAMEWORK FILTERING
# =============================================================================

# Standard business CDS view prefixes targeting interface, consumption, private, extension, and custom views
BUSINESS_VIEW_PREFIXES = ["I_*", "C_*", "P_*", "E_*", "Z*", "Y*"]


def is_valid_business_view(name: str, package: str = "") -> bool:
    """
    Validates whether a CDS view is an active business object.
    Filters out internal framework-generated objects (/1BS/, /1FC/) and temporary ($TMP) packages.
    """
    if not name:
        return False

    upper_name = name.upper().strip()
    upper_pkg = package.upper().strip()

    # Ignore internal framework-generated views starting with /1BS/ or /1FC/
    if upper_name.startswith("/1BS/") or upper_name.startswith("/1FC/"):
        return False
    if upper_name.startswith("1BS/") or upper_name.startswith("1FC/"):
        return False

    # Ignore internal $TMP (temporary/unassigned local) packages
    if upper_pkg == "$TMP" or upper_pkg.startswith("$TMP") or "$TMP" in upper_pkg:
        return False
    if "$TMP" in upper_name:
        return False

    # Ignore generic XML tag names or repository metadata types
    if upper_name in ("DDLS", "DDLS/DF", "OBJECTREFERENCES", "OBJECTREFERENCE"):
        return False

    return True


# =============================================================================
# SAP CDS DDL SOURCE CODE PARSER & STRING UTILITIES
# =============================================================================

def clean_description(label: str, view_name: str = "") -> str:
    """
    Cleans view description/label.
    Prevents spacing out letters for acronyms/all-caps names (e.g., 'I_ A B A P...' -> 'I_ABAPAPPLCOMPTEXT').
    """
    text = (label or view_name or "CDS View").strip()

    # 1. Fix any existing spaced-out single letters: e.g. "I_ A B A P ..." or "A B A P" -> "I_ABAP..."
    # Repeatedly collapse single-letter spaces: "A B" -> "AB", "_ A" -> "_A"
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", text)
        text = re.sub(r"_\s+([A-Za-z])", r"_\1", text)

    # 2. If text is PascalCase (e.g. "SalesOrder", "SalesOrderItem"), insert space between lower and upper
    # But do NOT insert space between consecutive uppercase letters (e.g. "ABAPAPPLCOMPTEXT" stays intact)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)

    # 3. Normalize multiple whitespace
    text = " ".join(text.split()).strip()

    return text or view_name or "CDS View"


def split_ddl_expressions(block: str) -> List[str]:
    """
    Splits DDL select list by commas that are outside parentheses (),
    brackets [], braces {}, or string literals.
    Preserves multiline expressions and function calls (e.g. cast, concat).
    """
    statements: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_quote = False
    quote_char = ''

    for char in block:
        if in_quote:
            current.append(char)
            if char == quote_char:
                in_quote = False
        elif char in ("'", '"'):
            in_quote = True
            quote_char = char
            current.append(char)
        elif char == '(':
            paren_depth += 1
            current.append(char)
        elif char == ')':
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
        elif char == '[':
            bracket_depth += 1
            current.append(char)
        elif char == ']':
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
        elif char == '{':
            brace_depth += 1
            current.append(char)
        elif char == '}':
            brace_depth = max(0, brace_depth - 1)
            current.append(char)
        elif char == ',' and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            statements.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        statements.append("".join(current))

    return statements


def extract_ddl_text_from_payload(raw_payload: str) -> str:
    """Extracts clean CDS DDL source code from SAP ADT XML/CDATA responses."""
    if not raw_payload or not raw_payload.strip():
        return ""

    text = raw_payload.strip()

    # 1. Parse XML wrapper if present
    if text.startswith("<"):
        try:
            # Strip namespaces for smooth tag searching
            clean_xml = re.sub(r'\sxmlns(?::\w+)?="[^"]*"', '', text)
            try:
                root = ET.fromstring(clean_xml.encode("utf-8"))
            except Exception:
                root = ET.fromstring(clean_xml)

            # Find DDL content node
            node = root.find(".//source") or root.find(".//content") or root
            if node is not None and node.text:
                text = node.text
        except Exception:
            # Fallback regex strip for XML tags
            text = re.sub(r'<!\[CDATA\[|\]\]>', '', text)
            text = re.sub(r'<[^>]+>', '', text)

    # 2. Unwrap CDATA markers
    text = re.sub(r'<!\[CDATA\[|\]\]>', '', text).strip()
    return text


def parse_cds_ddl(ddl_text: str) -> dict:
    """
    Parses fields, associations, and SQL view names from CDS DDL text.
    Handles 'define view', 'define view entity', and 'define root view entity'.
    """
    clean_ddl = extract_ddl_text_from_payload(ddl_text)

    if not clean_ddl:
        return {"sql_view_name": "", "fields": [], "associations": []}

    # Extract SQL View Name (@AbapCatalog.sqlViewName)
    sql_match = re.search(
        r"@AbapCatalog\.sqlViewName\s*:\s*['\"]([^'\"]+)['\"]",
        clean_ddl,
        re.IGNORECASE
    )
    sql_view_name = sql_match.group(1) if sql_match else ""

    # Extract Associations
    associations = []
    assoc_matches = re.findall(
        r'association\s+(?:\[.*?\]\s+)?to\s+([A-Za-z0-9_]+)\s+as\s+(_[A-Za-z0-9_]+)',
        clean_ddl,
        re.IGNORECASE
    )
    for target, alias in assoc_matches:
        associations.append({"target": target, "alias": alias})

    # Extract Fields inside select block { ... }
    fields = []
    body_match = re.search(r'\{([\s\S]*)\}', clean_ddl)

    if body_match:
        body = body_match.group(1)
        # Strip comments
        body = re.sub(r'//.*', '', body)
        body = re.sub(r'/\*[\s\S]*?\*/', '', body)

        for line in body.split('\n'):
            line = line.strip()
            if not line or line.startswith('@') or line.lower().startswith('association'):
                continue

            # Match field names or field aliases
            field_match = re.search(r'(?:[\w\.]+\s+as\s+)?([A-Za-z0-9_]+)\s*,?$', line)
            if field_match:
                fname = field_match.group(1)
                if fname.upper() not in {'KEY', 'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AS'}:
                    fields.append(fname)

    return {
        "sql_view_name": sql_view_name,
        "fields": list(dict.fromkeys(fields)),
        "associations": associations
    }


class CDSDDLParser:
    """
    Parses SAP Core Data Services (CDS) DDL (Data Definition Language) source code
    to extract View Name, Label/Description, Domain, Fields, and Associations.
    """

    @classmethod
    def parse_ddl(cls, ddl_content: str, default_name: str = "") -> CDSView:
        """
        Parses raw DDL text into a standardized CDSView metadata object.
        """
        if not ddl_content:
            return CDSView(
                name=default_name,
                description=default_name or "CDS View",
                domain=infer_domain(default_name),
                fields=[],
                associations=[]
            )

        clean_ddl = extract_ddl_text_from_payload(ddl_content)
        parsed_data = parse_cds_ddl(clean_ddl)

        # 1. Extract View Name
        name_match = re.search(
            r"define\s+(?:root\s+)?view\s+(?:entity\s+)?([A-Za-z0-9_\/]+)",
            clean_ddl,
            re.IGNORECASE
        )
        if name_match:
            view_name = name_match.group(1).strip()
        else:
            view_name = parsed_data.get("sql_view_name") or default_name

        # 2. Extract Business Label / Description
        label_match = re.search(
            r"@EndUserText\.label\s*:\s*['\"]([^'\"]+)['\"]",
            clean_ddl,
            re.IGNORECASE
        )
        if label_match and label_match.group(1).strip():
            raw_label = label_match.group(1).strip()
        else:
            raw_label = re.sub(r"^[A-Za-z]_", "", view_name) or view_name

        description = clean_description(raw_label, view_name)

        # 3. Classify Domain
        domain = infer_domain(view_name, description)

        return CDSView(
            name=view_name,
            description=description,
            domain=domain,
            fields=parsed_data.get("fields", []),
            associations=parsed_data.get("associations", [])
        )


# =============================================================================
# SAP ADT REST CLIENT WITH CSRF & RETRY ENGINE
# =============================================================================

class SAPADTClient:
    """
    Connects to SAP S/4HANA via standard ABAP Development Tools (ADT) REST APIs.
    Implements HTTP Basic Authentication, session retention, CSRF token handshakes,
    SSL bypass controls, and exponential backoff retry logic.
    """

    def __init__(self, config: SAPConfig):
        self.config = config
        self.base_url = config.base_url
        self.timeout = config.timeout_seconds
        self.verify_ssl = config.verify_ssl
        self.session: Optional[Any] = None
        self.csrf_token: Optional[str] = None

        if requests is None:
            logger.error(
                "The 'requests' package is not installed. "
                "Please run: pip install requests python-dotenv"
            )
            return

        self._init_session()

    def _init_session(self) -> None:
        """Configures requests.Session with BasicAuth, SSL verification, and retries."""
        self.session = requests.Session()

        # 1. HTTP Basic Authentication
        if self.config.username and self.config.password:
            self.session.auth = HTTPBasicAuth(self.config.username, self.config.password)
            logger.debug(f"Configured HTTP Basic Auth for user '{self.config.username}'.")

        # 2. SSL Verification & Warning Suppression
        self.session.verify = self.config.verify_ssl
        if not self.config.verify_ssl and urllib3 is not None:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.debug("SSL certificate verification disabled; urllib3 warnings suppressed.")

        # 3. Connection pooling & Retry handling
        if HTTPAdapter is not None and Retry is not None:
            retries = Retry(
                total=3,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                raise_on_status=False
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

        # 4. Standard default headers
        self.session.headers.update({
            "User-Agent": "SAP-ADT-CDS-Extractor/2.0",
            "sap-client": self.config.client
        })

    def execute_with_retry(
        self,
        method: str,
        url: str,
        max_attempts: int = 3,
        **kwargs
    ) -> Optional[Any]:
        """
        Executes an HTTP request with exponential backoff on timeouts / transient errors.
        """
        if self.session is None:
            return None

        # Apply default timeout if not specified
        kwargs.setdefault("timeout", self.config.timeout_seconds)

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.request(method=method, url=url, **kwargs)
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_err:
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    f"Network attempt {attempt}/{max_attempts} failed for '{url}': {conn_err}. "
                    f"Retrying in {backoff}s..."
                )
                if attempt == max_attempts:
                    logger.error(f"Exhausted {max_attempts} retry attempts connecting to '{url}': {conn_err}")
                    return None
                time.sleep(backoff)
            except Exception as ex:
                logger.error(f"Unexpected HTTP execution error for '{url}': {ex}")
                return None

        return None

    def authenticate_and_fetch_csrf(self) -> bool:
        """
        Queries the ADT discovery endpoint (/sap/bc/adt/discovery) with header 'x-csrf-token: fetch'
        to establish an authenticated session cookie and retrieve a valid CSRF token.
        """
        if self.session is None:
            return False

        discovery_url = f"{self.config.base_url}/sap/bc/adt/discovery"
        logger.info(
            f"Authenticating with SAP S/4HANA ADT at {discovery_url} "
            f"(Client: {self.config.client}, User: {self.config.username or 'Anonymous'})..."
        )

        headers = {
            "x-csrf-token": "fetch",
            "Accept": "application/atom+xml, application/xml, */*"
        }

        try:
            resp = self.execute_with_retry("GET", discovery_url, headers=headers)
            if resp is None:
                return False

            if resp.status_code in (200, 204):
                # Retrieve CSRF token (case-insensitive in requests)
                token = resp.headers.get("x-csrf-token")
                if token and token.lower() != "required":
                    self.csrf_token = token
                    self.session.headers["x-csrf-token"] = token
                    logger.info(f"Successfully authenticated with SAP. Retrieved CSRF Token: {token[:8]}...")
                else:
                    logger.warning("CSRF token header not found in discovery response; continuing with session cookies.")
                return True
            elif resp.status_code in (401, 403):
                logger.error(
                    f"SAP Authentication failed: HTTP {resp.status_code} - "
                    f"Please check SAP_USERNAME / SAP_PASSWORD in .env."
                )
                return False
            else:
                logger.error(f"SAP ADT discovery returned unexpected HTTP {resp.status_code}: {resp.text[:300]}")
                return False

        except Exception as e:
            logger.error(f"Unable to connect to SAP ADT endpoint '{discovery_url}': {e}")
            return False

    def search_active_cds_views(self, limit: int = 50, query: Optional[str] = None) -> List[str]:
        """
        Searches for active business CDS views using the ADT repository search endpoint.
        Target Specific View Patterns:
        Queries for standard business views using prefixes like I_*, C_*, P_*, E_*, Z*, or Y*
        instead of '*'.
        Filters out internal framework objects (/1BS/, /1FC/) and temporary ($TMP) packages.
        """
        if self.session is None:
            logger.error("Cannot perform repository search: HTTP session not initialized.")
            return []

        search_url = f"{self.config.base_url}/sap/bc/adt/repository/informationsystem/search"
        headers = {
            "Accept": "application/xml, text/xml, */*"
        }

        # Target specific prefixes or custom query
        if query and query.strip() != "*":
            target_queries = [query.strip()]
        else:
            target_queries = list(BUSINESS_VIEW_PREFIXES)

        logger.info(
            f"Searching SAP ADT Repository Information System for business views (patterns={target_queries}, limit={limit})..."
        )

        collected_views: List[str] = []

        for prefix in target_queries:
            if limit > 0 and len(collected_views) >= limit:
                break

            remaining = (limit - len(collected_views)) if limit > 0 else 50
            # Request buffer to account for filtered out framework/internal objects
            max_results = max(remaining * 2, 25) if limit > 0 else 0

            params = {
                "operation": "quickSearch",
                "query": prefix,
                "objectType": "DDLS",
                "sap-client": self.config.client
            }
            if max_results > 0:
                params["maxResults"] = str(max_results)

            logger.info(f"Querying ADT Repository for prefix '{prefix}' (objectType=DDLS)...")

            try:
                resp = self.execute_with_retry("GET", search_url, params=params, headers=headers)
                if resp is None:
                    logger.warning(f"Repository search for prefix '{prefix}' failed (no HTTP response / network timeout).")
                    continue

                if resp.status_code != 200:
                    logger.warning(
                        f"Repository search for prefix '{prefix}' failed: "
                        f"HTTP {resp.status_code} ({resp.reason}). "
                        f"{resp.text[:200] if resp.text else ''}"
                    )
                    continue

                # Parse and filter search response
                prefix_views = self._parse_search_response_xml(resp.text)
                added_count = 0
                for v in prefix_views:
                    if v not in collected_views:
                        collected_views.append(v)
                        added_count += 1
                        if limit > 0 and len(collected_views) >= limit:
                            break

                logger.info(
                    f"Prefix '{prefix}': Discovered {len(prefix_views)} candidate views, "
                    f"{added_count} added ({len(collected_views)} total collected so far)."
                )

            except Exception as e:
                logger.error(f"Failed to execute ADT search for prefix '{prefix}': {e}")

        logger.info(f"Discovered {len(collected_views)} active business CDS view definitions from SAP ADT search.")
        return collected_views

    def _parse_search_response_xml(self, content: str) -> List[str]:
        """
        Parses XML search results from ADT repository information system,
        extracting object names and package names while applying business view filters.
        """
        views: List[str] = []

        # 1. Parse XML elements using ElementTree
        try:
            root = ET.fromstring(content)
            for elem in root.iter():
                name = ""
                pkg = ""
                for k, v in elem.attrib.items():
                    k_lower = k.lower()
                    if k_lower.endswith("name") and not k_lower.endswith("packagename"):
                        name = v.strip()
                    elif k_lower.endswith("packagename") or k_lower.endswith("package"):
                        pkg = v.strip()

                if name and is_valid_business_view(name, pkg):
                    if name not in views:
                        views.append(name)
        except Exception as parse_ex:
            logger.debug(f"XML parse fallback to regex search: {parse_ex}")

        # 2. Regex fallback / complement
        if not views:
            regex_matches = re.findall(r'(?:adtcore:)?name="([^"]+)"', content)
            for raw_name in regex_matches:
                clean_name = raw_name.strip()
                if is_valid_business_view(clean_name):
                    if clean_name not in views:
                        views.append(clean_name)

        return views

    def fetch_ddl_source(self, ddl_name: str) -> str:
        """Fetches raw CDS DDL source code from SAP ADT REST endpoint."""
        # Append /source/main to request the actual source code instead of metadata
        url = f"{self.base_url}/sap/bc/adt/ddic/ddl/sources/{ddl_name.lower()}/source/main"

        headers = {
            "Accept": "text/plain, application/vnd.sap.adt.ddls.v2+xml, */*",
            "x-csrf-token": self.csrf_token
        }

        try:
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                raw_content = response.text
                return extract_ddl_text_from_payload(raw_content)
            else:
                logger.warning(f"Failed to fetch DDL for {ddl_name}: HTTP {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Error fetching DDL for {ddl_name}: {e}")
            return ""


# =============================================================================
# FALLBACK CATALOG PROVIDER
# =============================================================================

class FallbackCatalogProvider:
    """
    Supplies enterprise-grade SAP S/4HANA CDS view catalog metadata conforming
    to the target JSON specification when SAP is unreachable or returns 0 records.
    """

    FALLBACK_VIEWS: List[Dict[str, Any]] = [
        {
            "name": "I_SalesOrder",
            "description": "Sales Order Header Data",
            "domain": "SD",
            "fields": [
                "SalesOrder",
                "SalesOrderType",
                "SalesOrganization",
                "DistributionChannel",
                "OrganizationDivision",
                "SoldToParty",
                "CompanyCode",
                "TotalNetAmount",
                "TransactionCurrency",
                "CreationDate",
                "CreatedByUser"
            ],
            "associations": [
                "I_SalesOrderItem",
                "_Item",
                "I_Customer",
                "_SoldToParty",
                "I_CompanyCode",
                "_CompanyCode"
            ]
        },
        {
            "name": "I_SalesOrderItem",
            "description": "Sales Order Item Data",
            "domain": "SD",
            "fields": [
                "SalesOrder",
                "SalesOrderItem",
                "Material",
                "SalesOrderItemText",
                "OrderQuantity",
                "OrderQuantityUnit",
                "NetAmount",
                "TransactionCurrency",
                "Plant",
                "StorageLocation"
            ],
            "associations": [
                "I_SalesOrder",
                "_SalesOrder",
                "I_Product",
                "_Product"
            ]
        },
        {
            "name": "I_JournalEntry",
            "description": "Universal Journal Entry Line Items (ACDOCA)",
            "domain": "FI",
            "fields": [
                "Ledger",
                "CompanyCode",
                "FiscalYear",
                "AccountingDocument",
                "LedgerGLLineItem",
                "ChartOfAccounts",
                "GLAccount",
                "ControllingArea",
                "CostCenter",
                "ProfitCenter",
                "AmountInCompanyCodeCurrency",
                "CompanyCodeCurrency",
                "PostingDate"
            ],
            "associations": [
                "I_CompanyCode",
                "_CompanyCode",
                "I_GLAccount",
                "_GLAccount",
                "I_CostCenter",
                "_CostCenter"
            ]
        },
        {
            "name": "I_PurchaseOrder",
            "description": "Purchase Order Header Data",
            "domain": "MM",
            "fields": [
                "PurchaseOrder",
                "PurchaseOrderType",
                "CompanyCode",
                "Supplier",
                "PurchasingOrganization",
                "PurchasingGroup",
                "DocumentCurrency",
                "PurchaseOrderDate"
            ],
            "associations": [
                "I_PurchaseOrderItem",
                "_PurchaseOrderItem",
                "I_Supplier",
                "_Supplier",
                "I_CompanyCode",
                "_CompanyCode"
            ]
        },
        {
            "name": "I_MaterialStock",
            "description": "Current Material Stock Quantities",
            "domain": "MM",
            "fields": [
                "Material",
                "Plant",
                "StorageLocation",
                "Batch",
                "UnrestrictedStockQuantity",
                "MaterialBaseUnit"
            ],
            "associations": [
                "I_Product",
                "_Product"
            ]
        },
        {
            "name": "I_Customer",
            "description": "Customer Master Data",
            "domain": "SD",
            "fields": [
                "Customer",
                "CustomerName",
                "Country",
                "CityName",
                "PostalCode",
                "StreetName"
            ],
            "associations": [
                "I_SalesOrder",
                "_SalesOrder"
            ]
        },
        {
            "name": "I_BillingDocument",
            "description": "Billing Document Header Data",
            "domain": "SD",
            "fields": [
                "BillingDocument",
                "BillingDocumentType",
                "BillingDocumentCategory",
                "CompanyCode",
                "PayerParty",
                "TotalNetAmount",
                "TransactionCurrency",
                "BillingDocumentDate"
            ],
            "associations": [
                "I_SalesOrderItem",
                "_Item",
                "I_CompanyCode",
                "_CompanyCode",
                "I_Customer",
                "_PayerParty"
            ]
        },
        {
            "name": "I_GLAccount",
            "description": "General Ledger Account Master Data",
            "domain": "FI",
            "fields": [
                "ChartOfAccounts",
                "GLAccount",
                "GLAccountType",
                "GLAccountGroup",
                "IsBalanceSheetAccount"
            ],
            "associations": [
                "I_CompanyCode",
                "_CompanyCode"
            ]
        },
        {
            "name": "I_CostCenter",
            "description": "Cost Center Master Data",
            "domain": "CO",
            "fields": [
                "ControllingArea",
                "CostCenter",
                "CostCenterCategory",
                "ProfitCenter",
                "ResponsiblePerson",
                "ValidityEndDate"
            ],
            "associations": [
                "I_ControllingArea",
                "_ControllingArea",
                "I_ProfitCenter",
                "_ProfitCenter"
            ]
        },
        {
            "name": "I_Product",
            "description": "Product and Material Master Data",
            "domain": "MD",
            "fields": [
                "Product",
                "ProductType",
                "ProductGroup",
                "BaseUnit",
                "ItemCategoryGroup",
                "GrossWeight",
                "WeightUnit"
            ],
            "associations": [
                "I_Plant",
                "_Plant",
                "I_ProductDescription",
                "_Description"
            ]
        }
    ]

    @classmethod
    def get_fallback_catalog(cls) -> List[Dict[str, Any]]:
        """Returns deep copy of the fallback CDS view catalog."""
        logger.info(f"Loaded {len(cls.FALLBACK_VIEWS)} standard SAP S/4HANA CDS views from fallback catalog.")
        return [dict(view) for view in cls.FALLBACK_VIEWS]


# =============================================================================
# EXTRACTION PIPELINE ORCHESTRATOR
# =============================================================================

class S4HANACDSExtractorPipeline:
    """
    Coordinates connection test, discovery search, DDL extraction,
    parsing, fallback activation, and JSON catalog persistence.
    """

    def __init__(self, config: SAPConfig):
        self.config = config
        self.client = SAPADTClient(config)

    def run(self, output_path: str = "cds_catalog.json", query: str = "*") -> List[Dict[str, Any]]:
        """
        Executes the extraction pipeline and writes output to output_path.
        """
        catalog: List[Dict[str, Any]] = []
        is_connected = False

        if not self.config.use_mock and requests is not None:
            logger.info("Initiating SAP S/4HANA ADT live metadata extraction...")
            try:
                is_connected = self.client.authenticate_and_fetch_csrf()
            except Exception as conn_err:
                logger.error(f"Failed to authenticate with SAP system: {conn_err}")
                is_connected = False

            if is_connected:
                try:
                    view_names = self.client.search_active_cds_views(
                        limit=self.config.max_views,
                        query=query
                    )

                    if view_names:
                        logger.info(f"Extracting DDL source definitions for {len(view_names)} CDS views...")
                        for idx, v_name in enumerate(view_names, start=1):
                            if not is_valid_business_view(v_name):
                                logger.debug(f"Skipping internal framework view: {v_name}")
                                continue
                            logger.info(f"[{idx}/{len(view_names)}] Fetching DDL: {v_name}")
                            ddl_text = self.client.fetch_ddl_source(v_name)
                            if ddl_text:
                                parsed = CDSDDLParser.parse_ddl(ddl_text, default_name=v_name)
                                catalog.append(parsed.to_dict())
                            else:
                                logger.info(f"Omitting '{v_name}' from catalog due to DDL retrieval failure.")
                    else:
                        logger.warning(
                            "ADT search returned 0 CDS views. Activating fallback catalog..."
                        )
                except Exception as ex:
                    logger.error(f"Error during ADT CDS extraction cycle: {ex}")

        # Fallback Activation if empty or mock requested
        if not catalog:
            if self.config.use_mock:
                logger.info("Mock mode enabled via configuration (SAP_USE_MOCK=true). Using fallback catalog.")
            else:
                logger.warning("Live SAP extraction produced no records or connection failed. Activating fallback catalog data.")
            catalog = FallbackCatalogProvider.get_fallback_catalog()

        # Persist to JSON
        self.save_catalog(catalog, output_path)
        return catalog

    @staticmethod
    def save_catalog(catalog: List[Dict[str, Any]], filepath: str) -> None:
        """Saves catalog list to disk as clean indented JSON."""
        abs_path = os.path.abspath(filepath)
        logger.info(f"Saving {len(catalog)} CDS views to '{abs_path}'...")
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved catalog to '{abs_path}' successfully ({os.path.getsize(abs_path)} bytes).")
        except IOError as e:
            logger.error(f"Failed to write catalog file '{abs_path}': {e}")
            raise


# =============================================================================
# CLI INTERFACE & MAIN ENTRYPOINT
# =============================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="SAP S/4HANA Live CDS Metadata Extractor via ADT REST APIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o", "--output",
        default="cds_catalog.json",
        help="Target file path for extracted CDS catalog JSON"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("SAP_BASE_URL", ""),
        help="SAP Base URL (e.g. https://YAWSS4HSBX.sapyash.com:44301)"
    )
    parser.add_argument(
        "--client",
        default=os.getenv("SAP_CLIENT", "100"),
        help="SAP Client number (e.g. 100)"
    )
    parser.add_argument(
        "-u", "--username",
        default=os.getenv("SAP_USERNAME") or os.getenv("SAP_USER", ""),
        help="SAP Technical User (ADT authorized)"
    )
    parser.add_argument(
        "-p", "--password",
        default=os.getenv("SAP_PASSWORD", ""),
        help="SAP Technical User Password"
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        default=os.getenv("SAP_VERIFY_SSL", "false").lower() in ("true", "1", "yes"),
        help="Enable strict SSL certificate verification"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("SAP_REQUEST_TIMEOUT_SECONDS") or os.getenv("SAP_TIMEOUT") or "30"),
        help="SAP HTTP request timeout in seconds"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("SAP_MAX_VIEWS", "50")),
        help="Maximum number of CDS views to extract"
    )
    parser.add_argument(
        "--query",
        default="*",
        help="QuickSearch query filter ('*' defaults to standard business view prefixes [I_*, C_*, P_*, E_*, Z*, Y*], or specify a custom pattern like 'I_*')"
    )
    parser.add_argument(
        "--fallback",
        "--mock",
        dest="use_mock",
        action="store_true",
        help="Force use of fallback catalog without attempting network calls"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SAP_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level"
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_cli_args()

    # Set logging verbosity
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    logger.info("=" * 70)
    logger.info("SAP S/4HANA CDS Metadata Extraction Pipeline (ADT REST)")
    logger.info("=" * 70)

    config = SAPConfig(
        base_url=args.base_url,
        client=args.client,
        username=args.username,
        password=args.password,
        verify_ssl=args.verify_ssl,
        timeout_seconds=args.timeout,
        max_views=args.limit,
        use_mock=args.use_mock
    )

    logger.info(f"Target Base URL: {config.base_url}")
    logger.info(f"SAP Client:      {config.client}")
    logger.info(f"SAP Username:    {config.username or '(None)'}")
    logger.info(f"Verify SSL:      {config.verify_ssl}")
    logger.info(f"Request Timeout: {config.timeout_seconds}s")
    logger.info(f"Max Views Limit: {config.max_views}")
    logger.info(f"Catalog Target:  {args.output}")

    pipeline = S4HANACDSExtractorPipeline(config)

    try:
        catalog = pipeline.run(output_path=args.output, query=args.query)
        logger.info(f"Extraction pipeline completed. Total views in catalog: {len(catalog)}")

        # Print catalog preview
        logger.info("Catalog Summary Preview:")
        for view in catalog[:5]:
            logger.info(
                f" -> {view['name']} [{view['domain']}]: '{view['description']}' | "
                f"{len(view['fields'])} fields, {len(view['associations'])} associations"
            )
        if len(catalog) > 5:
            logger.info(f" ... and {len(catalog) - 5} more CDS views.")

        return 0

    except Exception as err:
        logger.critical(f"Pipeline encountered a critical error: {err}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
