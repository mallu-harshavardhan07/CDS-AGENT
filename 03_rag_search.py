"""
=============================================================================
Module: 03_rag_search.py
Description: Production Filtered ChromaDB Vector Search Engine with LLM Query
             Intent Expander for SAP S/4HANA CDS Views.
             - Strict ChromaDB Vector Filtering: Standard (Clean Core) vs Custom.
             - Zero SQLite overhead: Completely eliminates cds_metadata.db runtime dependency.
             - LLM Query Intent Expander: Enriches raw requirements into dense SAP VDM terminology.
             - Exact Technical Name Fast-Path with O(1) ChromaDB ID lookup.
Author: SAP S/4HANA Cloud & Clean Core Architecture Engineering
=============================================================================
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Any, Optional, Set, Tuple

# Load environment configuration
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

# ChromaDB & Embedding Functions
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

# DDIC Synonym & Field Mapper imports
try:
    from cds_guardrails import normalize_field_name, parse_abap_snippet, CleanCoreGuardrails, DDIC_SYNONYM_MAP
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from cds_guardrails import normalize_field_name, parse_abap_snippet, CleanCoreGuardrails, DDIC_SYNONYM_MAP
    except ImportError:
        CleanCoreGuardrails = None
        DDIC_SYNONYM_MAP = {}
        normalize_field_name = lambda x: re.sub(r"[^A-Za-z0-9_]", "", x.split("-")[-1].split(".")[-1]).strip().lower()
        parse_abap_snippet = lambda x: {"raw_tables": [], "raw_fields": [], "normalized_fields": [], "vdm_replacements": []}


# =============================================================================
# QUERY PRE-PROCESSING & TEXT NORMALIZATION
# =============================================================================

def clean_nl_query(query_text: str) -> str:
    """
    Strips meta-prompt fluff before passing to ChromaDB dense vector retrieval.
    """
    if not query_text:
        return ""

    fluff_patterns = [
        r"\bstandard\s+released\s+interface\s+views?\s+(?:for|of)?\b",
        r"\breleased\s+vdm\s+interface\s+views?\s+(?:for|of)?\b",
        r"\bstandard\s+vdm\s+views?\s+(?:for|of)?\b",
        r"\breleased\s+vdm\s+views?\s+(?:for|of)?\b",
        r"\bstandard\s+views?\s+(?:for|of)?\b",
        r"\breleased\s+interface\s+views?\s+(?:for|of)?\b",
        r"\breleased\s+views?\s+(?:for|of)?\b",
        r"\binterface\s+views?\s+(?:for|of)?\b",
        r"\bvdm\s+views?\s+(?:for|of)?\b",
        r"\bcustom\s+cds\s+views?\s+(?:for|of)?\b",
        r"\bcds\s+views?\s+(?:for|of)?\b",
        r"\bviews?\s+(?:for|of)?\b",
        r"\bwhere\s+can\s+(?:i|we)\s+find\b",
        r"\bshow\s+me\s+(?:the\s+)?\b",
        r"\b(?:i|we)\s+want\s+to\s+(?:see|find)\b",
        r"\b(?:i|we)\s+need\b",
        r"\bfind\s+(?:cds\s+)?views?\s+(?:for|of)?\b",
        r"\bgive\s+me\b",
        r"\bget\s+me\b",
        r"\bsearch\s+for\b"
    ]
    cleaned = query_text.strip()
    for pat in fluff_patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[,\.:;?\-!\s]+|[,\.:;?\-!\s]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else query_text.strip()


# =============================================================================
# LLM QUERY INTENT EXPANDER
# =============================================================================

def expand_query_intent(
    domain: Optional[str] = None,
    fields: Optional[str] = None,
    requirement: Optional[str] = None
) -> str:
    """
    Expands raw user inputs (domain, requested fields, requirements) into a
    dense, SAP VDM-optimized search prompt before vector lookup.
    
    Example:
      Input: Domain="SD", Fields="SalesOrder, TotalNetAmount", Requirement="Header details"
      Expanded: "SAP S/4HANA released interface view for sales order header details projecting SalesOrder, TotalNetAmount, SoldToParty, CreationDate, I_SalesOrder"
    """
    req_clean = (requirement or "").strip()
    dom_clean = (domain or "").strip()
    fld_clean = (fields or "").strip()

    # Normalize fields and resolve any legacy DDIC synonyms (e.g. KUNNR -> Customer, VBELN -> SalesOrder)
    normalized_flds: List[str] = []
    if fld_clean:
        # Check if raw ABAP query
        if re.search(r"\bselect\b|\bfrom\b", fld_clean, re.I):
            parsed = parse_abap_snippet(fld_clean)
            normalized_flds.extend(parsed.get("normalized_fields", []))
        else:
            tokens = [t.strip() for t in re.split(r"[,;\n\r\t]+", fld_clean) if t.strip()]
            for t in tokens:
                norm = normalize_field_name(t)
                normalized_flds.append(norm)

    normalized_flds = list(dict.fromkeys(normalized_flds))

    # Attempt Live LLM Intent Expansion if API key is configured
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and not api_key.startswith("your_openai"):
        try:
            import requests
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            model = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
            
            prompt_content = (
                f"Business Requirement: {req_clean or 'General CDS View'}\n"
                f"Domain: {dom_clean or 'Cross-Domain'}\n"
                f"Requested Fields: {', '.join(normalized_flds) if normalized_flds else fld_clean or 'Standard fields'}\n\n"
                "Rephrase the above into a single search-optimized SAP S/4HANA CDS prompt that mentions canonical "
                "released VDM interface view names (e.g. I_SalesOrder, I_Customer, I_BillingDocument, I_PurchaseOrder) "
                "and projected attributes. Return ONLY the single expanded sentence."
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a Senior SAP S/4HANA Clean Core Architect."},
                    {"role": "user", "content": prompt_content}
                ],
                "temperature": 0.2
            }
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                expanded = resp.json()["choices"][0]["message"]["content"].strip()
                if expanded and len(expanded) > 10:
                    return expanded
        except Exception:
            pass  # Fall through to deterministic VDM rule-based expander

    # Deterministic SAP VDM Intent Expander (Clean Core Rule-Based Fallback)
    combined_txt = f"{req_clean} {dom_clean} {' '.join(normalized_flds)}".lower()

    canonical_entities = []
    canonical_attributes = []
    process_phrase = req_clean if req_clean else "business entity"

    if any(k in combined_txt for k in ["sales", "order", "vbak", "vbap", "salesorder"]):
        canonical_entities.extend(["I_SalesOrder", "I_SalesOrderItem"])
        canonical_attributes.extend(["SalesOrder", "SalesOrderType", "SoldToParty", "TotalNetAmount", "TransactionCurrency", "CreationDate"])
        if not req_clean:
            process_phrase = "sales order header and item details"
    elif any(k in combined_txt for k in ["customer", "kna1", "client", "soldto", "payer"]):
        canonical_entities.extend(["I_Customer", "I_BusinessPartner"])
        canonical_attributes.extend(["Customer", "CustomerName", "CityName", "Country", "PostingIsBlocked"])
        if not req_clean:
            process_phrase = "customer master data and relationship"
    elif any(k in combined_txt for k in ["billing", "invoice", "vbrk", "vbrp"]):
        canonical_entities.extend(["I_BillingDocument", "I_BillingDocumentItem"])
        canonical_attributes.extend(["BillingDocument", "BillingDocumentType", "BillingDocumentDate", "TotalNetAmount", "PayerParty"])
        if not req_clean:
            process_phrase = "billing document header and line items"
    elif any(k in combined_txt for k in ["purchase", "po", "procurement", "ekko", "ekpo", "supplier"]):
        canonical_entities.extend(["I_PurchaseOrder", "I_PurchaseOrderItem", "I_Supplier"])
        canonical_attributes.extend(["PurchaseOrder", "PurchaseOrderItem", "Supplier", "Plant", "OrderQuantity"])
        if not req_clean:
            process_phrase = "purchase order and supplier details"
    elif any(k in combined_txt for k in ["journal", "finance", "accounting", "ledger", "bkpf", "bseg", "acdoca"]):
        canonical_entities.extend(["I_JournalEntry", "I_JournalEntryItem", "I_GLAccount"])
        canonical_attributes.extend(["AccountingDocument", "CompanyCode", "FiscalYear", "GLAccount", "AmountInTransactionCurrency"])
        if not req_clean:
            process_phrase = "general ledger journal entry items"
    elif any(k in combined_txt for k in ["material", "product", "stock", "mara", "makt", "marc"]):
        canonical_entities.extend(["I_Product", "I_ProductPlant", "I_MaterialStock"])
        canonical_attributes.extend(["Product", "ProductType", "Plant", "StorageLocation", "BaseUnit"])
        if not req_clean:
            process_phrase = "material and product master stock"

    # Merge user fields with canonical attributes
    final_fields = list(dict.fromkeys(normalized_flds + canonical_attributes))
    fields_str = ", ".join(final_fields[:6]) if final_fields else "core business attributes"
    entities_str = ", ".join(canonical_entities) if canonical_entities else "I_CDSView"

    expanded_query = (
        f"SAP S/4HANA released interface view for {process_phrase} projecting {fields_str}, {entities_str}"
    )
    return expanded_query


# =============================================================================
# HELPER: DDL FIELD EXTRACTOR
# =============================================================================

def extract_fields_from_ddl(ddl: str) -> List[str]:
    """
    Extracts field identifiers from a CDS view DDL definition string.
    """
    if not ddl:
        return []

    # Find the select body after 'as select from ... {'
    match = re.search(r"as\s+select\s+from[\s\S]*?\{([\s\S]*)\}\s*;?\s*$", ddl.strip(), re.IGNORECASE)
    if match:
        body = match.group(1)
    else:
        # Fallback to last curly-brace block
        blocks = list(re.finditer(r"\{([\s\S]*?)\}", ddl))
        body = blocks[-1].group(1) if blocks else ""

    if not body:
        return []

    # Strip line & block comments
    body = re.sub(r"//.*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"/\*[\s\S]*?\*/", "", body)

    fields = []
    for item in re.split(r",|\n", body):
        item = item.strip()
        if not item or item.startswith(("@", "_", "$", "association", "composition", "define", "key *")):
            continue
        parts = item.split()
        if not parts:
            continue
        lower_parts = [p.lower() for p in parts]
        if "as" in lower_parts:
            idx = lower_parts.index("as")
            field_name = parts[idx + 1] if idx + 1 < len(parts) else parts[-1]
        else:
            field_name = parts[-1]
        field_name = re.sub(r"[^A-Za-z0-9_]", "", field_name.split(".")[-1])
        if field_name and len(field_name) > 1 and field_name.lower() not in (
            "key", "case", "when", "then", "else", "end", "and", "or", "as", "select", "from", "distinct"
        ):
            fields.append(field_name)

    return list(dict.fromkeys(fields))


# =============================================================================
# AUTOMATED FALLBACK INDEX BUILDER FROM COMMITTED JSON METADATA
# =============================================================================

def build_vector_store_from_json(
    json_path: Optional[str] = None,
    db_path: str = "./cds_vector_db",
    collection_name: str = "cds_views",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> int:
    """
    Automated fallback index builder that populates ChromaDB directly from
    the committed extracted CDS metadata JSON file (extracted_views.json or cds_catalog.json).
    Ensures that any developer cloning the repository can immediately build the
    vector index without requiring SAP ADT or RFC access.
    """
    if chromadb is None or embedding_functions is None:
        raise ImportError(
            "ChromaDB and SentenceTransformers are required to build the vector store.\n"
            "Please run: pip install chromadb sentence-transformers"
        )

    resolved_json = None
    candidate_paths = [
        json_path,
        os.getenv("EXTRACTED_VIEWS_PATH"),
        "extracted_views.json",
        "cds_catalog.json",
        "cds_mapping.json"
    ]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for cand in candidate_paths:
        if not cand:
            continue
        if os.path.isabs(cand) and os.path.exists(cand):
            resolved_json = cand
            break
        rel_cwd = os.path.abspath(cand)
        if os.path.exists(rel_cwd):
            resolved_json = rel_cwd
            break
        rel_script = os.path.join(base_dir, cand)
        if os.path.exists(rel_script):
            resolved_json = rel_script
            break

    if not resolved_json:
        raise FileNotFoundError(
            f"Cannot auto-build ChromaDB index: No extracted CDS metadata JSON file found.\n"
            f"Searched: {candidate_paths}"
        )

    print(f"[Auto-Index] Initializing ChromaDB vector store from '{resolved_json}'...")
    with open(resolved_json, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    if isinstance(catalog_data, dict):
        if "index_to_cds" in catalog_data:
            views_list = list(catalog_data["index_to_cds"].values())
        else:
            views_list = list(catalog_data.values())
    elif isinstance(catalog_data, list):
        views_list = catalog_data
    else:
        raise ValueError(f"Unsupported JSON catalog format in '{resolved_json}'.")

    print(f"[Auto-Index] Found {len(views_list)} view definitions. Building embeddings into '{db_path}'...")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    transformer_name = model_name.replace("sentence-transformers/", "") if "/" in model_name else model_name
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=transformer_name)

    collection = client.get_or_create_collection(name=collection_name, embedding_function=emb_fn)

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for idx, item in enumerate(views_list):
        view_name = (item.get("view_name") or item.get("cds_name") or item.get("name") or f"CDS_{idx}").strip()
        ddl_src = (item.get("ddl_source_name") or item.get("source_name") or view_name).strip()
        desc = (item.get("description") or item.get("label") or "").strip()
        annotations = item.get("annotations", [])
        ddl_code = (item.get("ddl_code") or item.get("source") or "").strip()

        doc_id = view_name
        if doc_id in seen_ids:
            doc_id = f"{view_name}_{idx}"
        seen_ids.add(doc_id)

        doc_parts = [
            f"CDS View Name: {view_name}",
            f"DDL Source Name: {ddl_src}",
            f"Description: {desc}"
        ]
        if annotations:
            annos_str = ", ".join(str(a) for a in annotations) if isinstance(annotations, list) else str(annotations)
            doc_parts.append(f"Annotations: {annos_str}")
        if ddl_code:
            doc_parts.append(f"DDL Code:\n{ddl_code}")
        elif "document_text" in item:
            doc_parts.append(item["document_text"])

        doc_text = "\n\n".join(doc_parts)

        first_lines = [line.strip() for line in ddl_code.splitlines() if line.strip()]
        snippet = "\n".join(first_lines[:8]) if first_lines else f"define view {view_name}"

        is_standard = (
            (view_name.startswith("I_") or view_name.startswith("C_"))
            and not view_name.startswith("Z19_")
            and not view_name.startswith("P_")
        )
        is_rel_val = 1 if is_standard else 0
        v_type_val = "standard" if is_standard else "custom"

        meta: Dict[str, Any] = {
            "view_name": view_name,
            "ddl_source_name": ddl_src,
            "description": desc,
            "annotations_json": json.dumps(annotations) if annotations else "[]",
            "ddl_code": ddl_code,
            "ddl_snippet": snippet,
            "has_ddl_code": bool(ddl_code),
            "is_released": is_rel_val,
            "view_type": v_type_val
        }

        batch_ids.append(doc_id)
        batch_docs.append(doc_text)
        batch_metas.append(meta)

    chunk_size = 128
    for c_idx in range(0, len(batch_ids), chunk_size):
        collection.upsert(
            ids=batch_ids[c_idx : c_idx + chunk_size],
            documents=batch_docs[c_idx : c_idx + chunk_size],
            metadatas=batch_metas[c_idx : c_idx + chunk_size]
        )

    final_cnt = collection.count()
    print(f"[Auto-Index] Successfully built vector store with {final_cnt:,} views from '{resolved_json}'.")
    return final_cnt


# =============================================================================
# FILTERED CHROMADB VECTOR SEARCH ENGINE CLASS
# =============================================================================

class CDSRagEngine:
    """
    Production-grade SAP CDS View Retrieval Engine powered by ChromaDB vector metadata filtering.
    Zero SQLite runtime dependencies.
    Automatically auto-indexes from committed JSON metadata if ./cds_vector_db is missing.
    """

    clean_nl_query = staticmethod(clean_nl_query)
    expand_query_intent = staticmethod(expand_query_intent)
    build_vector_store_from_json = staticmethod(build_vector_store_from_json)

    def __init__(
        self,
        db_path: str = "./cds_vector_db",
        collection_name: str = "cds_views",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name

        if chromadb is None or embedding_functions is None:
            raise ImportError(
                "ChromaDB and SentenceTransformers are required.\n"
                "Please run: pip install chromadb sentence-transformers"
            )

        # Check if vector store exists and has records; if not, auto-build from JSON metadata
        needs_build = False
        if not os.path.exists(self.db_path):
            needs_build = True
        else:
            try:
                test_client = chromadb.PersistentClient(path=self.db_path)
                test_col = test_client.get_collection(name=self.collection_name)
                if test_col.count() == 0:
                    needs_build = True
            except Exception:
                needs_build = True

        if needs_build:
            print(f"[Info] ChromaDB vector store not found or empty at '{self.db_path}'. Triggering automated fallback indexing from JSON metadata...")
            build_vector_store_from_json(
                db_path=self.db_path,
                collection_name=self.collection_name,
                model_name=self.model_name
            )

        self.client = chromadb.PersistentClient(path=self.db_path)

        transformer_name = (
            self.model_name.replace("sentence-transformers/", "")
            if "/" in self.model_name
            else self.model_name
        )
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=transformer_name
        )

        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            raise RuntimeError(
                f"Collection '{self.collection_name}' could not be opened in '{self.db_path}': {e}"
            )

    def count(self) -> int:
        """Returns the total number of vectors in the ChromaDB collection."""
        return self.collection.count()

    def vector_search(
        self,
        query_text: str,
        scope: str = "Standard",
        top_k: int = 5,
        domain: Optional[str] = None,
        fields: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes strict metadata-filtered vector search against ChromaDB:
        - If scope == 'Standard' (Default): Filters with where={'is_released': 1}, strictly
          returning released S/4HANA interface views (I_*, C_*). Zero custom Z19_/custom views.
        - If scope == 'Custom': Filters with where={'is_released': 0}, returning custom views.
        - Checks for exact technical name match for instantaneous O(1) retrieval.
        - Scores and ranks candidates using dense semantic similarity and field coverage.
        """
        k = max(1, top_k)
        results: List[Dict[str, Any]] = []
        seen_views: Set[str] = set()

        is_standard_scope = not (scope and "custom" in scope.lower())
        target_released_val = 1 if is_standard_scope else 0
        where_filter = {"is_released": target_released_val}

        # ---------------------------------------------------------------------
        # 1. Parse & Normalize Requested Fields & Tokens
        # ---------------------------------------------------------------------
        normalized_fields: List[str] = []
        if fields:
            tokens = [t.strip() for t in re.split(r"[,;\n\r\t]+", fields) if t.strip()]
            for tok in tokens:
                norm = normalize_field_name(tok)
                if norm:
                    normalized_fields.append(norm)
        normalized_fields = list(dict.fromkeys(normalized_fields))

        # ---------------------------------------------------------------------
        # 2. Stage 1: Fast O(1) Exact Technical Name Lookup via ChromaDB IDs
        # ---------------------------------------------------------------------
        q_clean = query_text.strip()
        possible_technical_names: List[str] = []

        # Check if entire query or any token looks like a CDS view name (e.g. I_Customer, I_SalesOrder, Z19_...)
        for token in re.findall(r"\b[A-Za-z0-9_]{3,30}\b", q_clean):
            if token.startswith(("I_", "C_", "Z", "Y")) or "_" in token:
                possible_technical_names.append(token)

        for cand_name in possible_technical_names:
            try:
                exact_get = self.collection.get(
                    ids=[cand_name],
                    include=["metadatas", "documents"]
                )
                if exact_get["ids"] and exact_get["ids"][0] not in seen_views:
                    m = exact_get["metadatas"][0]
                    v_rel = m.get("is_released", 1 if cand_name.startswith(("I_", "C_")) else 0)

                    # Only accept if matching the active scope filter
                    if v_rel == target_released_val:
                        v_name = exact_get["ids"][0]
                        seen_views.add(v_name)
                        desc = m.get("description", "SAP CDS View")
                        ddl_code = m.get("ddl_code") or m.get("ddl_snippet") or ""
                        vdm_type = "#CONSUMPTION" if v_name.startswith("C_") else ("#BASIC" if v_name.startswith("I_") else "#CUSTOM")
                        contract = "#PUBLIC_LOCAL_API" if v_rel == 1 else "NOT_RELEASED"

                        results.append({
                            "rank": len(results) + 1,
                            "view_name": v_name,
                            "ddl_source_name": m.get("ddl_source_name", v_name),
                            "description": desc,
                            "similarity_score": 100.0,
                            "score": 100.0,
                            "field_score": 100.0,
                            "vector_score": 100.0,
                            "matched_fields": normalized_fields,
                            "domain": domain or "CROSS",
                            "fields_list": ", ".join(extract_fields_from_ddl(ddl_code)),
                            "distance": 0.0,
                            "ddl_snippet": m.get("ddl_snippet", ddl_code[:400]),
                            "ddl_code": ddl_code,
                            "annotations": m.get("annotations_json", "[]"),
                            "metadata": m,
                            "match_type": "EXACT_TECHNICAL_NAME",
                            "vdm_type": vdm_type,
                            "is_released": v_rel,
                            "release_contract": contract,
                            "view_type": m.get("view_type", "standard" if v_rel == 1 else "custom")
                        })
            except Exception:
                pass

        if len(results) >= k and (q_clean.lower() == results[0]["view_name"].lower()):
            return results[:k]

        # ---------------------------------------------------------------------
        # 3. Stage 2: ChromaDB Dense Semantic Vector Search with Metadata Filter
        # ---------------------------------------------------------------------
        # Pre-process & expand query intent
        cleaned_prompt = clean_nl_query(q_clean)
        expanded_prompt = expand_query_intent(domain=domain, fields=fields, requirement=cleaned_prompt)

        # Retrieve top candidates with metadata filter
        n_query = min(self.count(), max(k * 4, 25))
        try:
            query_res = self.collection.query(
                query_texts=[expanded_prompt],
                where=where_filter,
                n_results=n_query,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            print(f"[Warning] Filtered query failed: {e}. Retrying without secondary clauses.")
            query_res = self.collection.query(
                query_texts=[expanded_prompt],
                where={"view_type": "standard" if is_standard_scope else "custom"},
                n_results=n_query,
                include=["metadatas", "documents", "distances"]
            )

        if query_res and query_res["ids"] and query_res["ids"][0]:
            cand_ids = query_res["ids"][0]
            cand_metas = query_res["metadatas"][0]
            cand_dists = query_res["distances"][0]

            for vid, meta, dist in zip(cand_ids, cand_metas, cand_dists):
                if vid in seen_views:
                    continue
                seen_views.add(vid)

                # Strict scope verification
                is_rel = meta.get("is_released", 1 if vid.startswith(("I_", "C_")) else 0)
                if is_rel != target_released_val:
                    continue

                raw_score = round(max(0.0, 1.0 - float(dist)) * 100.0, 1)
                ddl_code = meta.get("ddl_code") or meta.get("ddl_snippet") or ""
                fields_in_view = extract_fields_from_ddl(ddl_code)
                fields_in_view_lower = [f.lower() for f in fields_in_view]

                # Calculate field matching coverage
                matched = []
                for rf in normalized_fields:
                    if rf in fields_in_view_lower or any(rf in f for f in fields_in_view_lower):
                        matched.append(rf)

                if normalized_fields:
                    field_score = round((len(matched) / len(normalized_fields)) * 100.0, 1)
                    composite_score = round((0.40 * field_score) + (0.60 * raw_score), 1)
                else:
                    field_score = raw_score
                    composite_score = raw_score

                # VDM clean core boost (+5% if interface view matching entity)
                if vid.startswith("I_") and is_standard_scope:
                    composite_score = min(99.0, composite_score + 5.0)

                vdm_type = "#CONSUMPTION" if vid.startswith("C_") else ("#BASIC" if vid.startswith("I_") else "#CUSTOM")
                contract = "#PUBLIC_LOCAL_API" if is_rel == 1 else "NOT_RELEASED"

                results.append({
                    "rank": len(results) + 1,
                    "view_name": vid,
                    "ddl_source_name": meta.get("ddl_source_name", vid),
                    "description": meta.get("description", "SAP CDS View"),
                    "similarity_score": composite_score,
                    "score": composite_score,
                    "field_score": field_score,
                    "vector_score": raw_score,
                    "matched_fields": matched,
                    "domain": domain or "CROSS",
                    "fields_list": ", ".join(fields_in_view[:25]),
                    "distance": round(float(dist), 4),
                    "ddl_snippet": meta.get("ddl_snippet", ddl_code[:400]),
                    "ddl_code": ddl_code,
                    "annotations": meta.get("annotations_json", "[]"),
                    "metadata": meta,
                    "match_type": "FILTERED_VECTOR_SEARCH",
                    "vdm_type": vdm_type,
                    "is_released": is_rel,
                    "release_contract": contract,
                    "view_type": meta.get("view_type", "standard" if is_rel == 1 else "custom")
                })

        # Sort results: Exact technical match first, then by score descending
        results.sort(key=lambda x: (1 if x.get("match_type") == "EXACT_TECHNICAL_NAME" else 0, x.get("score", 0.0)), reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results[:k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        scope: str = "Standard"
    ) -> List[Dict[str, Any]]:
        """Convenience alias for vector_search."""
        return self.vector_search(query_text=query, scope=scope, top_k=top_k)

    def hybrid_search(
        self,
        domain: Optional[str] = None,
        raw_fields_input: Optional[str] = None,
        nl_prompt: Optional[str] = None,
        top_k: int = 5,
        scope: str = "Standard"
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper for hybrid search using vector_search."""
        q = nl_prompt or raw_fields_input or "SAP CDS View"
        return self.vector_search(
            query_text=q,
            scope=scope,
            top_k=top_k,
            domain=domain,
            fields=raw_fields_input
        )

    def format_llm_prompt(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Formats retrieved CDS view candidates into a Clean Core LLM prompt."""
        header = (
            "You are an SAP S/4HANA Clean Core ABAP CDS Architect.\n"
            "Generate production-ready ABAP CDS View Entities strictly adhering to SAP Clean Core extensibility:\n"
            "- Gate 1: NEVER query raw DDIC tables (VBAK, VBAP, KNA1, etc.). Query released VDM views instead.\n"
            "- Gate 2: ALWAYS use modern 'define view entity' or 'define extension view entity' syntax.\n\n"
            f"User Requirement: \"{query}\"\n\n"
            "Retrieved Top S/4HANA CDS Candidates:\n"
        )
        candidates_str = []
        for r in results:
            candidates_str.append(
                f"- View: {r['view_name']} ({r['score']}%) | VDM: {r['vdm_type']} | Contract: {r['release_contract']}\n"
                f"  Description: {r['description']}\n"
                f"  DDL Snippet:\n{r['ddl_snippet']}\n"
            )
        return header + "\n".join(candidates_str)


# =============================================================================
# CLI PARSER & MAIN ENTRYPOINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAP CDS View Filtered ChromaDB Vector Search Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Search query or exact technical view name"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to return"
    )
    parser.add_argument(
        "-s", "--scope",
        type=str,
        default="Standard",
        choices=["Standard", "Custom"],
        help="Vector catalog scope: 'Standard' (Clean Core released views) or 'Custom'"
    )
    parser.add_argument(
        "-d", "--domain",
        type=str,
        default=None,
        help="Optional business domain filter"
    )
    parser.add_argument(
        "-f", "--fields",
        type=str,
        default=None,
        help="Optional comma-separated fields/technical names"
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("VECTOR_DB_PATH", "./cds_vector_db"),
        help="Directory path of persistent ChromaDB database"
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("VECTOR_COLLECTION_NAME", "cds_views"),
        help="ChromaDB collection name"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        engine = CDSRagEngine(db_path=args.db_path, collection_name=args.collection_name)
        print(f"\n[Info] Connected to ChromaDB at '{args.db_path}' ({engine.count():,} views indexed).")
        print(f"[Info] Executing Vector Search for: '{args.query}' [Scope: {args.scope}]...\n")

        results = engine.vector_search(
            query_text=args.query,
            scope=args.scope,
            top_k=args.top_k,
            domain=args.domain,
            fields=args.fields
        )

        divider = "=" * 80
        print(divider)
        print(f"{'Rank':<5} {'View Name':<32} {'Score':<8} {'Scope':<10} {'Description'}")
        print(divider)
        for r in results:
            print(f"{r['rank']:<5} {r['view_name']:<32} {r['score']:<8.1f}% {r['view_type']:<10} {r['description'][:30]}")
        print(divider)
        return 0
    except Exception as e:
        print(f"[ERROR] Search failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
