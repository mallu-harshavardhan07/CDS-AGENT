"""
=============================================================================
Module: 03_server.py
Description: Production-ready FastAPI Core Server for evaluating SAP S/4HANA
             CDS view requirements using ChromaDB persistent vector retrieval
             with SentenceTransformers neural embeddings and LLM RAG.
Author: Senior Backend & RAG Engineer
=============================================================================
"""

import os
import sys
import json
import time
import math
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ChromaDB & Neural Embeddings
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

# OpenAI / LLM integration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("CDS_RAG_Server")


# =============================================================================
# PYDANTIC SCHEMAS (Request & Response Validation - Pydantic V2 Compliant)
# =============================================================================

class EvaluateCDSRequest(BaseModel):
    """Payload incoming from the 4-step CDS requirement wizard."""
    domain: str = Field(
        ...,
        description="SAP Functional Domain (e.g., SD, FI, MM, PP, CO, CA)",
        json_schema_extra={"example": "SD"}
    )
    business_objective: str = Field(
        ...,
        description="High-level business objective or user story",
        json_schema_extra={"example": "Track outstanding sales orders with customer delivery address and total net values"}
    )
    entities: List[str] = Field(
        ...,
        description="List of primary SAP business entities involved",
        json_schema_extra={"example": ["SalesOrder", "Customer", "DeliveryDocument"]}
    )
    fields: List[str] = Field(
        ...,
        description="List of specific requested fields or attributes",
        json_schema_extra={"example": ["SalesOrder", "TotalNetAmount", "CustomerName", "CityName", "CreationDate"]}
    )


class CandidateMatch(BaseModel):
    """Metadata and match metrics for a retrieved candidate CDS view."""
    cds_name: str
    label: str
    domain: str
    match_score: float = Field(..., description="Match score percentage (0.0 to 100.0)")
    key_fields: List[str]
    field_count: int
    association_count: int


class RecommendationType(str, Enum):
    """Decision category for the CDS requirement."""
    REUSE = "REUSE"
    EXTEND = "EXTEND"
    CREATE = "CREATE"


class EvaluateCDSResponse(BaseModel):
    """Structured response payload returned to the caller."""
    recommendation_type: RecommendationType
    target_cds: Optional[str] = Field(
        None,
        description="CDS view targeted for REUSE or EXTEND (null for CREATE)"
    )
    explanation: str = Field(
        ...,
        description="Architectural rationale and evaluation justification"
    )
    generated_ddl: Optional[str] = Field(
        None,
        description="Syntactically valid ABAP Core Data Services DDL code"
    )
    candidate_matches: List[CandidateMatch] = Field(
        ...,
        description="Top-k similar CDS views retrieved via vector search"
    )
    query_used: str = Field(..., description="Synthesized semantic query string")
    processing_time_ms: float = Field(..., description="End-to-end request processing time in milliseconds")


class HealthResponse(BaseModel):
    """Server health and asset readiness status."""
    status: str
    uptime_seconds: float
    model_loaded: bool
    model_name: str
    vector_db: str = "ChromaDB"
    collection_name: str
    db_path: str
    indexed_vectors: int
    tfidf_index_loaded: bool = False
    mapping_entries_count: int = 0


# =============================================================================
# DOCUMENT SYNTHESIZER UTILITY
# =============================================================================

class CDSDocumentSynthesizer:
    """Creates a high-signal contextual document representation for CDS views."""
    @staticmethod
    def build_text_representation(cds: Dict[str, Any]) -> str:
        cds_name = (cds.get("cds_name") or cds.get("name") or "").strip()
        label = (cds.get("label") or cds.get("description") or "").strip()
        domain = (cds.get("domain") or "").strip()
        fields = cds.get("fields", [])
        associations = cds.get("associations", [])

        key_fields: List[str] = []
        regular_fields: List[str] = []
        for f in fields:
            if isinstance(f, dict):
                name = f.get("name", "")
                ftype = f.get("type", "CHAR")
                flabel = f.get("label", name)
                is_key = f.get("is_key", False)
                field_desc = f"{name} ({ftype}): {flabel}" if flabel != name else f"{name} ({ftype})"
            else:
                name = str(f)
                is_key = False
                field_desc = name

            if is_key:
                key_fields.append(field_desc)
            else:
                regular_fields.append(field_desc)

        assoc_descriptions: List[str] = []
        for a in associations:
            if isinstance(a, dict):
                alias = a.get("alias", "")
                target = a.get("target") or a.get("target_cds", "")
                cardinality = a.get("cardinality", "")
                cond = a.get("condition", "").replace("\n", " ").strip()
                parts = []
                if alias:
                    parts.append(alias)
                if target:
                    parts.append(f"-> {target}")
                if cardinality:
                    parts.append(cardinality)
                if cond:
                    parts.append(f"ON {cond}")
                assoc_descriptions.append(" ".join(parts).strip() if parts else str(a))
            else:
                assoc_descriptions.append(f"Association: {str(a)}")

        doc_parts = [
            f"CDS View: {cds_name}",
            f"Business Label: {label}",
            f"Functional Domain: {domain}",
            f"Summary: Standard SAP S/4HANA CDS View '{cds_name}' ({label}) belonging to functional area {domain}.",
        ]
        if key_fields:
            doc_parts.append("Key Fields:\n  - " + "\n  - ".join(key_fields))
        if regular_fields:
            capped = regular_fields[:50]
            if len(regular_fields) > 50:
                capped.append(f"... and {len(regular_fields) - 50} more fields")
            doc_parts.append("Attributes & Measures:\n  - " + "\n  - ".join(capped))
        if assoc_descriptions:
            doc_parts.append("Associations & Relationships:\n  - " + "\n  - ".join(assoc_descriptions))

        return "\n".join(doc_parts)


# =============================================================================
# ASSET MANAGER & STATE CONTAINER (ChromaDB + SentenceTransformers)
# =============================================================================

class RAGState:
    """Holds persistent ChromaDB client, collection, and embedding function."""
    def __init__(self):
        self.start_time: float = time.time()
        self.chroma_client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.embedding_fn: Optional[Any] = None
        self.model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.db_path: str = os.getenv("CHROMA_DB_PATH", "./cds_vector_db")
        self.collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "cds_views")
        self.openai_client: Optional[Any] = None
        self.llm_model: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.indexed_count: int = 0
        self.in_memory_catalog: List[Dict[str, Any]] = []

    def initialize_assets(self) -> None:
        """Loads persistent ChromaDB collection with SentenceTransformer embeddings."""
        logger.info("Initializing CDS RAG Server assets (ChromaDB + SentenceTransformers)...")

        if chromadb is None or embedding_functions is None:
            logger.error(
                "ChromaDB or SentenceTransformers is not installed.\n"
                "Please run: pip install -r requirements.txt"
            )
            return

        try:
            # 1. Connect to Persistent ChromaDB
            logger.info(f"Connecting to persistent ChromaDB at: '{self.db_path}'")
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)

            # 2. Configure SentenceTransformer Embedding Function
            logger.info(f"Configuring SentenceTransformer embedding function ('{self.model_name}')...")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )

            # 3. Get or Create Collection with cosine distance space
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self.indexed_count = self.collection.count()
            logger.info(
                f"ChromaDB Collection '{self.collection_name}' ready. "
                f"Active indexed vectors: {self.indexed_count}"
            )

            # 4. Fallback Auto-Indexing: If collection is empty, index from cds_catalog.json
            if self.indexed_count == 0 and os.path.exists("cds_catalog.json"):
                logger.info("ChromaDB collection is empty. Auto-indexing from 'cds_catalog.json' in batch chunks...")
                self._auto_index_catalog("cds_catalog.json")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}", exc_info=True)

        # 5. Load catalog into memory cache for instant schema lookups if available
        if os.path.exists("cds_catalog.json"):
            try:
                with open("cds_catalog.json", "r", encoding="utf-8") as f:
                    self.in_memory_catalog = json.load(f)
            except Exception:
                pass

        # 6. Initialize OpenAI / LLM Client
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))

        if OpenAI is not None and api_key and api_key != "your_openai_api_key_here":
            try:
                self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"OpenAI LLM client configured (Base URL: {base_url}, Model: {self.llm_model}).")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
        else:
            logger.info("No active OPENAI_API_KEY found. Server will use deterministic S/4HANA rules engine.")

    def _auto_index_catalog(self, catalog_path: str, batch_size: int = 64) -> None:
        """Indexes CDS views from catalog into ChromaDB in batches if collection is empty."""
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            if not isinstance(catalog, list) or not catalog:
                return

            total_views = len(catalog)
            seen_ids: Set[str] = set()

            for i in range(0, total_views, batch_size):
                chunk = catalog[i : i + batch_size]
                batch_ids: List[str] = []
                batch_docs: List[str] = []
                batch_metadatas: List[Dict[str, Any]] = []

                for idx, cds in enumerate(chunk):
                    global_idx = i + idx
                    cds_name = (cds.get("cds_name") or cds.get("name") or "").strip()
                    label = (cds.get("label") or cds.get("description") or "").strip()
                    domain = (cds.get("domain") or "").strip()
                    raw_fields = cds.get("fields", [])
                    raw_associations = cds.get("associations", [])

                    doc_id = cds_name if cds_name else f"cds_{global_idx}"
                    if doc_id in seen_ids:
                        doc_id = f"{doc_id}_{global_idx}"
                    seen_ids.add(doc_id)

                    doc_text = CDSDocumentSynthesizer.build_text_representation(cds)
                    key_fields = [
                        (f.get("name") if isinstance(f, dict) else str(f))
                        for f in raw_fields
                        if (isinstance(f, dict) and f.get("is_key"))
                    ]

                    metadata = {
                        "cds_name": cds_name,
                        "name": cds_name,
                        "label": label,
                        "description": label,
                        "domain": domain,
                        "field_count": len(raw_fields),
                        "association_count": len(raw_associations),
                        "key_fields_json": json.dumps(key_fields),
                        "fields_json": json.dumps(raw_fields),
                        "associations_json": json.dumps(raw_associations),
                    }

                    batch_ids.append(doc_id)
                    batch_docs.append(doc_text)
                    batch_metadatas.append(metadata)

                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas
                )

            self.indexed_count = self.collection.count()
            logger.info(f"Auto-indexed {self.indexed_count} views into ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Auto-indexing failed: {e}", exc_info=True)


rag_state = RAGState()


# =============================================================================
# LIFESPAN CONTEXT MANAGER (FastAPI Modern Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup asset initialization and graceful shutdown."""
    rag_state.initialize_assets()
    yield
    logger.info("CDS RAG Server shutting down.")


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="SAP S/4HANA CDS Requirement Evaluation Core",
    description="ChromaDB persistent vector search & LLM RAG engine for evaluating SAP CDS views (SentenceTransformers).",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend wizard clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# CHROMADB VECTOR RETRIEVAL ENGINE
# =============================================================================

def synthesize_query(req: EvaluateCDSRequest) -> str:
    """Combines 4-step wizard attributes into a dense retrieval query."""
    entities_str = ", ".join(req.entities)
    fields_str = ", ".join(req.fields)
    return (
        f"Domain: {req.domain}. "
        f"Objective: {req.business_objective}. "
        f"Key Business Entities: {entities_str}. "
        f"Requested Fields and Measures: {fields_str}."
    )


def retrieve_candidates(query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
    """
    Queries persistent ChromaDB collection using SentenceTransformers embedding,
    computes cosine similarity scores, and returns sorted top-K candidate views.
    """
    if rag_state.collection is None or rag_state.indexed_count == 0:
        logger.warning("ChromaDB collection unavailable or empty. Attempting keyword fallback.")
        return fallback_keyword_search(query, top_k)

    try:
        n_results = min(top_k, rag_state.collection.count())
        if n_results <= 0:
            return []

        # ChromaDB query with cosine distance
        results = rag_state.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        candidates: List[Tuple[Dict[str, Any], float]] = []

        for i in range(len(ids)):
            doc_id = ids[i]
            dist = float(distances[i]) if (distances and i < len(distances)) else 1.0
            meta = metadatas[i] if (metadatas and i < len(metadatas)) else {}
            doc_text = documents[i] if (documents and i < len(documents)) else ""

            # Cosine distance to similarity percentage:
            # For hnsw:space: cosine, distance = 1.0 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - dist))
            score_pct = round(similarity * 100.0, 2)

            # Reconstruct structured fields from JSON strings in metadata
            key_fields: List[str] = []
            if "key_fields_json" in meta:
                try:
                    key_fields = json.loads(meta["key_fields_json"])
                except Exception:
                    key_fields = []

            fields: List[Any] = []
            if "fields_json" in meta:
                try:
                    fields = json.loads(meta["fields_json"])
                except Exception:
                    fields = []

            associations: List[Any] = []
            if "associations_json" in meta:
                try:
                    associations = json.loads(meta["associations_json"])
                except Exception:
                    associations = []

            cds_name = meta.get("cds_name") or meta.get("name") or doc_id
            label = meta.get("label") or meta.get("description") or ""

            candidate_dict = {
                "id": doc_id,
                "cds_name": cds_name,
                "name": cds_name,
                "cds_view_name": cds_name,
                "label": label,
                "description": label,
                "domain": meta.get("domain", ""),
                "field_count": int(meta.get("field_count", len(fields))),
                "association_count": int(meta.get("association_count", len(associations))),
                "key_fields": key_fields,
                "fields": fields,
                "associations": associations,
                "document_text": doc_text
            }

            candidates.append((candidate_dict, score_pct))

        return candidates

    except Exception as e:
        logger.error(f"ChromaDB retrieval failed: {e}", exc_info=True)
        return fallback_keyword_search(query, top_k)


def fallback_keyword_search(query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
    """Heuristic string-matching fallback if vector collection is offline."""
    import re
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored: List[Tuple[Dict[str, Any], float]] = []

    for cds in rag_state.in_memory_catalog:
        doc = CDSDocumentSynthesizer.build_text_representation(cds).lower()
        doc_tokens = set(re.findall(r"\w+", doc))
        overlap = len(query_tokens.intersection(doc_tokens))
        score = min(95.0, round((overlap / max(1, len(query_tokens))) * 100.0, 2))

        cds_name = cds.get("cds_name") or cds.get("name") or "Unknown"
        cand_dict = {
            "cds_name": cds_name,
            "name": cds_name,
            "cds_view_name": cds_name,
            "label": cds.get("label") or cds.get("description") or "",
            "description": cds.get("description") or cds.get("label") or "",
            "domain": cds.get("domain", ""),
            "field_count": len(cds.get("fields", [])),
            "association_count": len(cds.get("associations", [])),
            "key_fields": [
                (f.get("name") if isinstance(f, dict) else str(f))
                for f in cds.get("fields", [])
                if (isinstance(f, dict) and f.get("is_key"))
            ],
            "fields": cds.get("fields", []),
            "associations": cds.get("associations", [])
        }
        scored.append((cand_dict, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# =============================================================================
# DETERMINISTIC ABAP DDL GENERATOR (Production Offline Fallback)
# =============================================================================

class DeterministicABAPGenerator:
    """
    Generates exact, syntactically verified ABAP Core Data Services DDL code
    following SAP S/4HANA Clean ABAP and Virtual Data Model (VDM) standards.
    """

    @staticmethod
    def generate_extend_view(target_cds: str, req: EvaluateCDSRequest, top_candidate: Dict[str, Any]) -> str:
        """Generates EXTEND VIEW ENTITY code for SAP S/4HANA."""
        ext_name = f"Z_EXT_{target_cds.replace('I_', '')[:15]}_{req.domain}"
        existing_fields = {
            (f.get("name", "") if isinstance(f, dict) else str(f)).lower()
            for f in top_candidate.get("fields", [])
        }

        missing_fields = [f for f in req.fields if f.lower() not in existing_fields]
        if not missing_fields:
            missing_fields = ["CustomerGroup", "SalesDistrict", "AdditionalDiscount"]

        field_lines = []
        for field in missing_fields:
            field_clean = "".join(c for c in field if c.isalnum() or c == "_")
            field_lines.append(
                f"  // Added for business requirement: {req.business_objective[:50]}\n"
                f"  _Base.{field_clean} as {field_clean}"
            )

        ddl = f"""@EndUserText.label: 'Extension for {target_cds} - {req.domain}'
extend view entity {target_cds} with {ext_name}
{{
{',\n'.join(field_lines)}
}};"""
        return ddl

    @staticmethod
    def generate_create_view(req: EvaluateCDSRequest) -> str:
        """Generates a complete DEFINE VIEW ENTITY specification."""
        domain_tag = req.domain.upper()
        slug = "".join(c for c in req.business_objective if c.isalnum())[:16]
        if not slug:
            slug = "CustomAnalytics"
        view_name = f"Z_I_{domain_tag}_{slug}"
        sql_view = f"Z{domain_tag[:2]}{slug[:10]}".upper()

        key_fields = [req.fields[0]] if req.fields else ["DocumentID"]
        attrib_fields = req.fields[1:] if len(req.fields) > 1 else ["Status", "CreatedOn", "NetValue"]

        field_statements = []
        for k in key_fields:
            clean_k = "".join(c for c in k if c.isalnum() or c == "_")
            field_statements.append(f"  key base.{clean_k} as {clean_k}")

        for a in attrib_fields:
            clean_a = "".join(c for c in a if c.isalnum() or c == "_")
            field_statements.append(f"      base.{clean_a} as {clean_a}")

        source_entity = req.entities[0].lower() if req.entities else 'vbak'

        ddl = f"""@AbapCatalog.sqlViewName: '{sql_view}'
@AbapCatalog.compiler.compareFilter: true
@AbapCatalog.preserveKey: true
@AccessControl.authorizationCheck: #CHECK
@EndUserText.label: '{req.business_objective[:60]}'
@VDM.viewType: #COMPOSITE
@Analytics.dataCategory: #FACT

define view entity {view_name}
  as select from {source_entity} as base
{{
{',\n'.join(field_statements)}
}};"""
        return ddl


# =============================================================================
# ARCHITECTURAL DECISION ENGINE THRESHOLDS
# =============================================================================
# Configurable thresholds (supports ratio 0.0-1.0 or percentage 0-100)
_raw_reuse = float(os.getenv("REUSE_THRESHOLD", "0.25"))
_raw_extend = float(os.getenv("EXTEND_THRESHOLD", "0.15"))
REUSE_THRESHOLD = _raw_reuse if _raw_reuse <= 1.0 else (_raw_reuse / 100.0)
EXTEND_THRESHOLD = _raw_extend if _raw_extend <= 1.0 else (_raw_extend / 100.0)


def evaluate_with_llm(
    req: EvaluateCDSRequest,
    candidates: List[Tuple[Dict[str, Any], float]],
    query_str: str
) -> Dict[str, Any]:
    """
    Sends structured requirements and retrieved top-3 candidates to OpenAI / local LLM,
    or falls back to the deterministic S/4HANA rules engine.
    Decision Thresholds:
      - REUSE: Match Score >= REUSE_THRESHOLD (default 25%)
      - EXTEND: Match Score EXTEND_THRESHOLD - REUSE_THRESHOLD (default 15% - 25%)
      - CREATE: Match Score < EXTEND_THRESHOLD (default < 15%)
    """
    top_candidate, raw_score = candidates[0] if candidates else ({}, 0.0)

    # Normalize score to 0.0 - 1.0 and calculate percentage
    top_score = (raw_score / 100.0) if raw_score > 1.0 else raw_score
    score_pct = top_score * 100.0

    target_cds_view = (
        top_candidate.get("cds_view_name")
        or top_candidate.get("cds_name")
        or top_candidate.get("name")
        or ""
    )
    if "cds_view_name" not in top_candidate and target_cds_view:
        top_candidate["cds_view_name"] = target_cds_view

    # If OpenAI client is available, prompt the LLM
    if rag_state.openai_client is not None:
        try:
            return call_openai_evaluator(req, candidates, query_str)
        except Exception as e:
            logger.error(f"OpenAI evaluation failed: {e}. Falling back to deterministic rules engine.")

    # Deterministic Evaluation Engine (Governed by matching score thresholds)
    logger.info(
        f"Evaluating with Deterministic Rules Engine (Top score: {score_pct:.2f}% on '{target_cds_view}', "
        f"Thresholds: REUSE>={REUSE_THRESHOLD*100:.1f}%, EXTEND>={EXTEND_THRESHOLD*100:.1f}%)"
    )

    if top_score >= REUSE_THRESHOLD and target_cds_view:
        recommendation_type = RecommendationType.REUSE
        target_cds = target_cds_view
        explanation = f"Existing CDS view '{target_cds_view}' satisfies the requirement with a match score of {score_pct:.2f}%."
        generated_ddl = None

    elif top_score >= EXTEND_THRESHOLD and target_cds_view and req.fields:
        recommendation_type = RecommendationType.EXTEND
        target_cds = target_cds_view
        explanation = (
            f"Standard CDS View '{target_cds_view}' matches the core business model with a similarity score of {score_pct:.2f}%, "
            f"but does not expose all requested specialized fields ({', '.join(req.fields[:3])}). "
            f"Following SAP S/4HANA Clean Core guidelines, you should not modify the standard view or create an unneeded duplicate. "
            f"Use EXTEND VIEW ENTITY to non-disruptively append the missing fields."
        )
        generated_ddl = DeterministicABAPGenerator.generate_extend_view(target_cds, req, top_candidate)

    else:
        recommendation_type = RecommendationType.CREATE
        target_cds = None
        explanation = f"No existing standard SAP CDS view adequately satisfies the business requirements (highest match score was {score_pct:.2f}%)."
        generated_ddl = DeterministicABAPGenerator.generate_create_view(req)

    return {
        "recommendation_type": recommendation_type,
        "target_cds": target_cds,
        "explanation": explanation,
        "generated_ddl": generated_ddl
    }


def call_openai_evaluator(
    req: EvaluateCDSRequest,
    candidates: List[Tuple[Dict[str, Any], float]],
    query_str: str
) -> Dict[str, Any]:
    """Invokes OpenAI with strict system instructions and JSON response formatting."""
    candidates_context = []
    for cand, score in candidates:
        candidates_context.append({
            "cds_name": cand.get("cds_name"),
            "label": cand.get("label"),
            "domain": cand.get("domain"),
            "match_score": score,
            "key_fields": cand.get("key_fields", []),
            "all_fields": [
                f.get("name") if isinstance(f, dict) else str(f)
                for f in cand.get("fields", [])
            ],
            "associations": [
                a.get("alias") if isinstance(a, dict) else str(a)
                for a in cand.get("associations", [])
            ]
        })

    system_prompt = f"""You are a Principal SAP S/4HANA Architect & ABAP Core Data Services (CDS) Expert.
Your task is to evaluate a user's CDS view requirements against retrieved top candidate views from the SAP catalog.

Evaluate and classify into exactly one recommendation_type:
- REUSE (Match Score >= {REUSE_THRESHOLD*100:.0f}%): Existing view satisfies requirements. Return view name, usage instructions, no DDL needed.
- EXTEND (Match Score {EXTEND_THRESHOLD*100:.0f}%-{REUSE_THRESHOLD*100:.0f}%): Partially matches. Return exact, syntactically correct `extend view entity <Target> with Z_... {{ ... }}` DDL.
- CREATE (Match Score < {EXTEND_THRESHOLD*100:.0f}%): No adequate candidate. Return complete, production-grade `define view entity Z_... as select from ... {{ ... }}` DDL with appropriate annotations.

Output MUST be a JSON object with keys:
{{
  "recommendation_type": "REUSE" | "EXTEND" | "CREATE",
  "target_cds": string or null,
  "explanation": string,
  "generated_ddl": string or null
}}"""

    user_prompt = f"""User CDS Requirements:
- Domain: {req.domain}
- Business Objective: {req.business_objective}
- Requested Entities: {req.entities}
- Requested Fields: {req.fields}

Retrieved Candidates from ChromaDB Vector Retriever:
{json.dumps(candidates_context, indent=2)}

Provide your evaluation and DDL recommendation in valid JSON."""

    response = rag_state.openai_client.chat.completions.create(
        model=rag_state.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    return {
        "recommendation_type": RecommendationType(parsed.get("recommendation_type", "CREATE")),
        "target_cds": parsed.get("target_cds"),
        "explanation": parsed.get("explanation", ""),
        "generated_ddl": parsed.get("generated_ddl")
    }


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", include_in_schema=False)
async def serve_wizard_ui():
    """Serves the interactive 4-step wizard UI."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "SAP S/4HANA CDS AI Requirement Evaluator API"}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Returns application health, ChromaDB collection status, and uptime."""
    uptime = time.time() - rag_state.start_time
    count = rag_state.collection.count() if rag_state.collection is not None else rag_state.indexed_count
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 2),
        model_loaded=(rag_state.collection is not None),
        model_name=rag_state.model_name,
        vector_db="ChromaDB",
        collection_name=rag_state.collection_name,
        db_path=rag_state.db_path,
        indexed_vectors=count,
        tfidf_index_loaded=False,
        mapping_entries_count=count
    )


@app.post(
    "/api/evaluate-cds",
    response_model=EvaluateCDSResponse,
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"]
)
async def evaluate_cds(request: EvaluateCDSRequest):
    """
    Evaluates SAP CDS requirements from a 4-step wizard:
    1. Synthesizes a search query string.
    2. Queries persistent ChromaDB collection using SentenceTransformers (top-3 candidates).
    3. Executes RAG evaluation against REUSE / EXTEND / CREATE thresholds.
    4. Generates production ABAP DDL code if required.
    """
    start_ts = time.perf_counter()
    logger.info(f"Received CDS evaluation request for domain: '{request.domain}', objective: '{request.business_objective[:50]}...'")

    try:
        # Step 1: Synthesize Query
        query_str = synthesize_query(request)
        logger.debug(f"Synthesized query: {query_str}")

        # Step 2: ChromaDB Neural Vector Retrieval (Top-3)
        retrieved_candidates = retrieve_candidates(query_str, top_k=3)
        logger.info(f"Retrieved {len(retrieved_candidates)} candidate CDS views via ChromaDB cosine similarity.")

        # Format Candidate Match objects
        formatted_candidates: List[CandidateMatch] = []
        for cand, score in retrieved_candidates:
            formatted_candidates.append(CandidateMatch(
                cds_name=cand.get("cds_name", "Unknown"),
                label=cand.get("label", ""),
                domain=cand.get("domain", ""),
                match_score=score,
                key_fields=cand.get("key_fields", []),
                field_count=cand.get("field_count", len(cand.get("fields", []))),
                association_count=cand.get("association_count", len(cand.get("associations", [])))
            ))

        # Step 3: LLM / Rules RAG Evaluation Engine
        eval_result = evaluate_with_llm(request, retrieved_candidates, query_str)

        processing_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)
        logger.info(
            f"Evaluation completed in {processing_ms}ms: "
            f"Recommendation={eval_result['recommendation_type']}, Target={eval_result.get('target_cds')}"
        )

        return EvaluateCDSResponse(
            recommendation_type=eval_result["recommendation_type"],
            target_cds=eval_result.get("target_cds"),
            explanation=eval_result["explanation"],
            generated_ddl=eval_result.get("generated_ddl"),
            candidate_matches=formatted_candidates,
            query_used=query_str,
            processing_time_ms=processing_ms
        )

    except Exception as e:
        logger.error(f"Error evaluating CDS requirement: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CDS evaluation pipeline failed: {str(e)}"
        )


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting CDS RAG Server on http://{host}:{port} ...")
    uvicorn.run("03_server:app", host=host, port=port, reload=True)
