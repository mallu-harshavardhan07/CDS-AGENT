"""
=============================================================================
Module: build_sqlite_catalog.py
Description: Production SQLite Metadata Catalog Builder for SAP S/4HANA CDS Views.
             Extracts, structures, and indexes high-fidelity CDS metadata from
             the persistent ChromaDB collection (./cds_vector_db) into a fast,
             queryable SQLite relational database (cds_metadata.db).
             Enforces SAP Clean Core classification: VDM type, Release Contract,
             DDIC field list extraction, and functional domain assignment.
Author: SAP S/4HANA Cloud & Clean Core Architecture Engineering
=============================================================================
"""

import os
import sys
import json
import time
import re
import sqlite3
import argparse
from typing import Dict, List, Any, Tuple, Optional

# Load environment configuration
from dotenv import load_dotenv
load_dotenv()

# ChromaDB import
try:
    import chromadb
except ImportError:
    chromadb = None


# =============================================================================
# FIELD EXTRACTION & METADATA PARSER
# =============================================================================

def extract_fields_from_ddl(ddl_code: str) -> str:
    """
    Extracts view element aliases and column definitions from the DDL select block.
    Returns a comma-separated string of unique field names.
    """
    if not ddl_code:
        return ""

    idx = ddl_code.lower().find("define view")
    if idx == -1:
        idx = ddl_code.lower().find("extend view")
    if idx == -1:
        idx = ddl_code.lower().find("define table function")
    if idx == -1:
        return ""

    after_def = ddl_code[idx:]
    open_brace = after_def.find("{")
    close_brace = after_def.rfind("}")
    if open_brace == -1 or close_brace == -1:
        return ""

    body = after_def[open_brace + 1:close_brace]

    # Clean annotations and comments
    clean_body = re.sub(r"@[A-Za-z0-9_\.]+(?:\s*:\s*(?:\{[^}]*\}|\[[^]]*\]|[^\n,;]+))?", "", body)
    clean_body = re.sub(r"/\*[\s\S]*?\*/|//.*|--.*", "", clean_body)

    fields = set()
    # 1. Match aliases: 'as <identifier>'
    for m in re.finditer(r"\bas\s+([A-Za-z0-9_]+)\b", clean_body, re.IGNORECASE):
        alias = m.group(1)
        if alias.lower() not in ["preserving", "type", "select", "distinct", "where", "group", "having"]:
            fields.add(alias)

    # 2. Match field identifiers before commas or newlines
    for line in clean_body.split("\n"):
        line = line.strip().rstrip(",")
        if not line or "as " in line.lower():
            continue
        m_simple = re.findall(r"\b(?:key\s+)?(?:[A-Za-z0-9_]+\.)?([A-Za-z0-9_]+)\b", line)
        for f in m_simple:
            if len(f) >= 2 and f.lower() not in [
                "key", "case", "when", "then", "else", "end", "cast", "and", "or",
                "not", "null", "association", "composition", "redirected", "to", "on"
            ]:
                fields.add(f)

    # Return clean comma-separated list
    sorted_fields = sorted(list(fields))
    return ", ".join(sorted_fields[:400])


def infer_domain(view_name: str, base_tables: str, description: str) -> str:
    """
    Infers the functional domain of a CDS view (SD, MM, FI, PP, QM, PM, CROSS, GENERAL).
    """
    text = f"{view_name} {base_tables} {description}".lower()

    # Sales & Distribution (SD)
    if any(k in text for k in [
        "sales", "vbak", "vbap", "vbrk", "vbrp", "likp", "lips", "vbkd", "billing",
        "invoice", "pricing", "delivery", "order item", "customer return", "salesorder"
    ]):
        return "SD"

    # Materials Management & Sourcing (MM)
    if any(k in text for k in [
        "purchase", "purchasing", "ekko", "ekpo", "mara", "marc", "mard", "mseg", "mkpf",
        "eban", "ebkn", "supplier", "vendor", "lfa1", "product", "material", "inventory",
        "stock", "warehouse"
    ]):
        return "MM"

    # Financial Accounting & Controlling (FI)
    if any(k in text for k in [
        "journal", "ledger", "acdoca", "bkpf", "bseg", "ska1", "skat", "glaccount",
        "company code", "controlling", "profit center", "cost center", "fiscal", "accounting",
        "asset", "payment", "bank"
    ]):
        return "FI"

    # Production Planning (PP)
    if any(k in text for k in ["production", "routing", "work center", "bom", "bill of material", "afko", "afpo"]):
        return "PP"

    # Plant Maintenance (PM)
    if any(k in text for k in ["maintenance", "equipment", "functional location", "notification", "maint"]):
        return "PM"

    # Quality Management (QM)
    if any(k in text for k in ["inspection", "quality", "qals"]):
        return "QM"

    # Master Data / Cross-Application (CROSS)
    if any(k in text for k in ["customer", "businesspartner", "kna1", "but000", "address", "country", "currency", "unit"]):
        return "CROSS"

    return "GENERAL"


def parse_cds_record(view_id: str, meta: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, int, str]:
    """
    Parses and standardizes raw CDS view metadata into relational catalog attributes.

    Returns:
        (view_name, sql_view_name, description, fields_list, domain, vdm_type, release_contract, is_released, base_tables)
    """
    meta = meta or {}
    view_name = meta.get("view_name") or view_id
    ddl_code = meta.get("ddl_code") or ""
    description = meta.get("description") or ""

    # Parse annotations from annotations_json if available
    annos = []
    if meta.get("annotations_json"):
        try:
            annos = json.loads(meta["annotations_json"])
        except Exception:
            annos = []

    # 1. SQL View Name (Classic DDIC SQL View)
    sql_view_name = ""
    sql_m = re.search(r"@AbapCatalog\.sqlViewName\s*:\s*['\"]([^'\"]+)['\"]", ddl_code, re.IGNORECASE)
    if sql_m:
        sql_view_name = sql_m.group(1).strip()
    elif annos:
        for a in annos:
            m = re.search(r"@AbapCatalog\.sqlViewName\s*:\s*['\"]([^'\"]+)['\"]", a, re.IGNORECASE)
            if m:
                sql_view_name = m.group(1).strip()
                break

    # 2. VDM View Type (#BASIC, #COMPOSITE, #CONSUMPTION, etc.)
    vdm_type = ""
    vdm_m = re.search(r"@VDM\.viewType\s*:\s*(#\w+)", ddl_code, re.IGNORECASE)
    if vdm_m:
        vdm_type = vdm_m.group(1).upper()
    elif annos:
        for a in annos:
            m = re.search(r"@VDM\.viewType\s*:\s*(#\w+)", a, re.IGNORECASE)
            if m:
                vdm_type = m.group(1).upper()
                break

    if not vdm_type:
        if re.search(r"@Analytics\.query\s*:\s*true", ddl_code, re.IGNORECASE):
            vdm_type = "#CONSUMPTION"
        elif view_name.startswith("I_"):
            vdm_type = "#BASIC"
        elif view_name.startswith("C_"):
            vdm_type = "#CONSUMPTION"
        elif view_name.startswith("P_") or view_name.startswith("Z19_"):
            vdm_type = "#PRIVATE"
        elif view_name.startswith("Z") or view_name.startswith("Y"):
            vdm_type = "#CUSTOM"
        else:
            vdm_type = "#UNKNOWN"

    # 3. Clean Core Release Contract & Release Status
    # Action Item 1: Enforce is_released = 1 and #PUBLIC_LOCAL_API for I_* and C_*
    # Hard-filter private/unreleased internal views (Z19_*, P_*, _*) to is_released = 0
    if view_name.startswith("Z19_") or view_name.startswith("P_") or view_name.startswith("_"):
        release_contract = "NOT_RELEASED"
        is_released = 0
    elif view_name.startswith("I_") or view_name.startswith("C_"):
        release_contract = "#PUBLIC_LOCAL_API"
        is_released = 1
    elif view_name.startswith("Z") or view_name.startswith("Y"):
        release_contract = "CUSTOM_TENANT"
        is_released = 1
    else:
        release_contract = "NOT_RELEASED"
        is_released = 0

    # 4. Base Tables & Referenced Views
    base_tables_list = []
    matches = re.findall(r"\b(?:from|join)\s+([A-Za-z0-9_/\.]+)", ddl_code, re.IGNORECASE)
    reserved_keywords = {
        "SELECT", "AS", "INNER", "LEFT", "RIGHT", "CROSS", "OUTER", "JOIN",
        "FROM", "WHERE", "ON", "UNION", "ALL", "EXCEPT", "INTERSECT", "AND", "OR"
    }
    for match in matches:
        clean_target = match.strip().strip("'").strip('"')
        if clean_target.upper() not in reserved_keywords and clean_target not in base_tables_list and clean_target != view_name:
            base_tables_list.append(clean_target)

    base_tables_str = ", ".join(base_tables_list)

    # 5. Extract fields list & infer domain
    fields_list_str = extract_fields_from_ddl(ddl_code)
    domain_str = infer_domain(view_name, base_tables_str, description)

    return (
        view_name,
        sql_view_name,
        description,
        fields_list_str,
        domain_str,
        vdm_type,
        release_contract,
        is_released,
        base_tables_str
    )


# =============================================================================
# SQLITE CATALOG BUILDER CLASS
# =============================================================================

class SQLiteCatalogBuilder:
    """
    Manages the creation, population, and indexing of cds_metadata.db.
    """

    def __init__(
        self,
        db_path: str = "cds_metadata.db",
        chroma_path: str = "./cds_vector_db",
        collection_name: str = "cds_views"
    ):
        self.db_path = os.path.abspath(db_path)
        self.chroma_path = os.path.abspath(chroma_path)
        self.collection_name = collection_name
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """Establishes SQLite connection with optimized WAL mode for fast writes."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        return self.conn

    def init_schema(self, reset: bool = False) -> None:
        """Creates the cds_views table, view alias cds_views_catalog, and performance indexes."""
        if not self.conn:
            self.connect()

        cur = self.conn.cursor()

        if reset:
            print(f"[Info] Reset flag set. Dropping existing tables and views...")
            try:
                cur.execute("DROP VIEW IF EXISTS cds_views_catalog;")
            except Exception:
                pass
            try:
                cur.execute("DROP TABLE IF EXISTS cds_views_catalog;")
            except Exception:
                pass
            try:
                cur.execute("DROP VIEW IF EXISTS cds_views;")
            except Exception:
                pass
            try:
                cur.execute("DROP TABLE IF EXISTS cds_views;")
            except Exception:
                pass

        # Create primary table: cds_views
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cds_views (
            view_name TEXT PRIMARY KEY,
            sql_view_name TEXT,
            description TEXT,
            fields_list TEXT,
            domain TEXT,
            vdm_type TEXT,
            release_contract TEXT,
            is_released INTEGER,
            base_tables TEXT
        );
        """)

        # Create backward-compatible VIEW alias: cds_views_catalog
        cur.execute("""
        CREATE VIEW IF NOT EXISTS cds_views_catalog AS
        SELECT view_name, sql_view_name, description, fields_list, domain, vdm_type, release_contract, is_released, base_tables
        FROM cds_views;
        """)

        # Create SQL Indexes for fast lookups & filtering
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cds_view_name ON cds_views (view_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cds_is_released ON cds_views (is_released);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cds_domain ON cds_views (domain);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cds_vdm_type ON cds_views (vdm_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cds_release_contract ON cds_views (release_contract);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cds_sql_view_name ON cds_views (sql_view_name);")

        self.conn.commit()
        print(f"[Success] Initialized table 'cds_views' and view alias 'cds_views_catalog' in '{self.db_path}'.")

    def build_catalog_from_chroma(self, batch_size: int = 5000) -> int:
        """
        Pulls all metadata in batches from ChromaDB and ingests into SQLite.
        """
        if not chromadb:
            raise ImportError("chromadb is not installed. Please install via pip.")

        if not os.path.exists(self.chroma_path):
            raise FileNotFoundError(f"ChromaDB path '{self.chroma_path}' does not exist.")

        print(f"[Info] Connecting to ChromaDB at '{self.chroma_path}'...")
        client = chromadb.PersistentClient(path=self.chroma_path)
        collection = client.get_collection(self.collection_name)
        total_views = collection.count()

        print(f"[Info] Found {total_views:,} active CDS views in ChromaDB collection '{self.collection_name}'.")
        print(f"[Info] Ingesting metadata into SQLite catalog in chunks of {batch_size:,}...")

        start_time = time.perf_counter()
        inserted_count = 0
        offset = 0

        insert_sql = """
        INSERT OR REPLACE INTO cds_views (
            view_name,
            sql_view_name,
            description,
            fields_list,
            domain,
            vdm_type,
            release_contract,
            is_released,
            base_tables
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        cur = self.conn.cursor()

        while offset < total_views:
            batch_t0 = time.perf_counter()
            fetch_res = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"]
            )

            ids = fetch_res.get("ids", [])
            metadatas = fetch_res.get("metadatas", [])

            if not ids:
                break

            records_to_insert: List[Tuple[str, str, str, str, str, str, str, int, str]] = []
            for doc_id, meta in zip(ids, metadatas):
                rec = parse_cds_record(view_id=doc_id, meta=meta)
                records_to_insert.append(rec)

            cur.executemany(insert_sql, records_to_insert)
            self.conn.commit()

            inserted_count += len(records_to_insert)
            offset += len(ids)

            batch_elapsed = time.perf_counter() - batch_t0
            pct = (inserted_count / total_views) * 100.0
            print(f"  |-> Ingested {inserted_count:,} / {total_views:,} views ({pct:.1f}%) [Batch: {batch_elapsed:.2f}s]")

        total_elapsed = time.perf_counter() - start_time
        print(f"\n[Success] Ingestion Complete! Populated {inserted_count:,} CDS records in {total_elapsed:.2f}s.")
        return inserted_count

    def print_catalog_statistics(self) -> None:
        """Computes and prints statistical summaries of the catalog."""
        if not self.conn:
            self.connect()

        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cds_views;")
        total_count = cur.fetchone()[0]

        cur.execute("SELECT is_released, COUNT(*) FROM cds_views GROUP BY is_released;")
        rel_counts = dict(cur.fetchall())
        released_count = rel_counts.get(1, 0)
        unreleased_count = rel_counts.get(0, 0)

        cur.execute("SELECT domain, COUNT(*) FROM cds_views GROUP BY domain ORDER BY COUNT(*) DESC;")
        domain_counts = cur.fetchall()

        cur.execute("SELECT vdm_type, COUNT(*) FROM cds_views GROUP BY vdm_type ORDER BY COUNT(*) DESC LIMIT 8;")
        vdm_counts = cur.fetchall()

        cur.execute("SELECT release_contract, COUNT(*) FROM cds_views GROUP BY release_contract ORDER BY COUNT(*) DESC LIMIT 6;")
        contract_counts = cur.fetchall()

        cur.execute("SELECT COUNT(*) FROM cds_views WHERE fields_list != '';")
        views_with_fields = cur.fetchone()[0]

        divider = "=" * 80
        print("\n" + divider)
        print("SAP S/4HANA SQLITE METADATA CATALOG REPORT (cds_metadata.db)")
        print(divider)
        print(f"Total CDS Views Cataloged:      {total_count:,}")
        print(f"Views with Extracted Fields:   {views_with_fields:,} ({(views_with_fields/max(1,total_count))*100:.1f}%)")
        print(f"Clean Core Released Views:     {released_count:,} ({(released_count/max(1,total_count))*100:.1f}%)")
        print(f"Private / Internal Views:      {unreleased_count:,} ({(unreleased_count/max(1,total_count))*100:.1f}%)")
        print("-" * 80)
        print("BREAKDOWN BY FUNCTIONAL DOMAIN:")
        for dom, cnt in domain_counts:
            print(f"  - {dom:<22} : {cnt:>7,} views ({(cnt/max(1,total_count))*100:.1f}%)")
        print("-" * 80)
        print("BREAKDOWN BY VDM TYPE:")
        for vdm, cnt in vdm_counts:
            print(f"  - {vdm:<22} : {cnt:>7,} views")
        print("-" * 80)
        print("BREAKDOWN BY RELEASE CONTRACT:")
        for contract, cnt in contract_counts:
            print(f"  - {contract:<22} : {cnt:>7,} views")
        print(divider + "\n")


# =============================================================================
# CLI PARSER & ENTRYPOINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SQLite Metadata Catalog for SAP CDS Views",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("SQLITE_CATALOG_PATH", "cds_metadata.db"),
        help="Path to SQLite catalog database"
    )
    parser.add_argument(
        "--chroma-path",
        default=os.getenv("VECTOR_DB_PATH") or os.getenv("CHROMA_DB_PATH", "./cds_vector_db"),
        help="Path to persistent ChromaDB database"
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("VECTOR_COLLECTION_NAME") or os.getenv("CHROMA_COLLECTION_NAME", "cds_views"),
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for pulling and ingesting views"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the catalog table and indexes"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("SAP S/4HANA SQLITE METADATA CATALOG BUILDER")
    print("=" * 80)
    print(f"Target Database:    {os.path.abspath(args.db_path)}")
    print(f"ChromaDB Store:     {os.path.abspath(args.chroma_path)}")
    print(f"Chroma Collection:  {args.collection}")
    print(f"Batch Size:         {args.batch_size:,}")
    print(f"Reset Table:        {args.reset}")
    print("-" * 80)

    builder = SQLiteCatalogBuilder(
        db_path=args.db_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection
    )

    try:
        builder.connect()
        builder.init_schema(reset=args.reset)
        builder.build_catalog_from_chroma(batch_size=args.batch_size)
        builder.print_catalog_statistics()
        print("[SUCCESS] SQLite Metadata Catalog is fully built and ready for RAG operations!\n")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Catalog build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if builder.conn:
            builder.conn.close()


if __name__ == "__main__":
    sys.exit(main())
