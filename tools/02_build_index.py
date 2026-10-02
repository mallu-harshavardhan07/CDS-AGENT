"""
=============================================================================
Module: 02_build_index.py
Description: Production-ready ChromaDB Vector Search Index Builder with
             SentenceTransformers embeddings for SAP S/4HANA CDS Views.
             Processes CDS catalogs in batch chunks (default: 128) to
             efficiently scale to thousands of views, persisting vectors to
             ./cds_vector_db.
             Supports both high-fidelity DDL definitions and legacy schemas.
Author: SAP S/4HANA Cloud & AI Integration Engineering
=============================================================================
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from typing import Dict, List, Any, Optional, Set

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# ChromaDB & Neural Embedding utilities
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("CDS_Chroma_Indexer")


# =============================================================================
# BENCHMARK TIMER UTILITY
# =============================================================================

class BenchmarkTimer:
    """High-resolution execution timer for profiling pipeline phases."""
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"Starting stage: [{self.stage_name}]...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        logger.info(f"Completed stage: [{self.stage_name}] in {self.elapsed:.4f} seconds.")


# =============================================================================
# DOCUMENT SYNTHESIZER
# =============================================================================

class CDSDocumentSynthesizer:
    """
    Transforms structured CDS catalog records into rich, semantically dense
    searchable text representations combining view_name, description, annotations,
    and ddl_code (with graceful fallback for legacy schema structures).
    """

    @staticmethod
    def build_text_representation(cds: Dict[str, Any]) -> str:
        """
        Creates a high-signal contextual document combining:
        1. CDS View Name & DDL Source Name
        2. Description / Label
        3. Annotations (@VDM, @Analytics, @EndUserText, @ObjectModel)
        4. DDL Source Code (or legacy fields & associations if DDL code is absent)
        """
        view_name = (
            cds.get("view_name") or
            cds.get("cds_name") or
            cds.get("name") or
            ""
        ).strip()

        ddl_source_name = (
            cds.get("ddl_source_name") or
            cds.get("source_name") or
            view_name
        ).strip()

        description = (
            cds.get("description") or
            cds.get("label") or
            ""
        ).strip()

        annotations = cds.get("annotations", [])
        ddl_code = (cds.get("ddl_code") or cds.get("source") or "").strip()

        # Build document payload
        doc_parts: List[str] = [
            f"CDS View Name: {view_name}",
            f"DDL Source Name: {ddl_source_name}",
            f"Description: {description}",
        ]

        # Add annotations if present
        if annotations:
            if isinstance(annotations, list):
                annos_str = ", ".join(str(a) for a in annotations)
            else:
                annos_str = str(annotations)
            doc_parts.append(f"Annotations: {annos_str}")

        # Add full DDL code if present
        if ddl_code:
            doc_parts.append(f"DDL Code:\n{ddl_code}")
        else:
            # Fallback to legacy fields and associations format if DDL is not provided
            domain = (cds.get("domain") or "").strip()
            if domain:
                doc_parts.append(f"Functional Domain: {domain}")

            raw_fields = cds.get("fields", [])
            if raw_fields:
                field_names = [f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in raw_fields]
                doc_parts.append("Fields: " + ", ".join(field_names[:50]))

            raw_assocs = cds.get("associations", [])
            if raw_assocs:
                assoc_list = []
                for a in raw_assocs:
                    if isinstance(a, dict):
                        alias = a.get("alias", "")
                        tgt = a.get("target") or a.get("target_cds", "")
                        assoc_list.append(f"{alias} -> {tgt}".strip(" ->"))
                    else:
                        assoc_list.append(str(a))
                doc_parts.append("Associations: " + ", ".join(assoc_list))

        return "\n\n".join(doc_parts)


# =============================================================================
# CHROMADB VECTOR INDEX BUILDER (BATCH PROCESSING)
# =============================================================================

class CDSChromaIndexBuilder:
    """
    Builds and manages a persistent ChromaDB vector collection using
    SentenceTransformers neural embeddings. Scales to thousands of CDS views
    via robust batch chunking (default: 128).
    """

    def __init__(
        self,
        db_path: str = "./cds_vector_db",
        collection_name: str = "cds_views",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        # Allow short form or full HuggingFace path
        self.model_name = model_name
        self.batch_size = max(1, batch_size)

        if chromadb is None or embedding_functions is None:
            raise ImportError(
                "ChromaDB and SentenceTransformers are required.\n"
                "Please install them via: pip install chromadb sentence-transformers"
            )

        logger.info(f"Initializing Persistent ChromaDB client at: '{self.db_path}'")
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Standardize model name for SentenceTransformerEmbeddingFunction
        transformer_name = (
            self.model_name.replace("sentence-transformers/", "")
            if "/" in self.model_name
            else self.model_name
        )
        logger.info(f"Configuring SentenceTransformer embedding function: '{transformer_name}'")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=transformer_name
        )

    def load_catalog(self, catalog_path: str) -> List[Dict[str, Any]]:
        """Reads and validates the input CDS catalog JSON file."""
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file not found: '{catalog_path}'")

        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        if not isinstance(catalog, list):
            raise ValueError(f"Catalog file must contain a JSON array of CDS views. Found: {type(catalog)}")

        logger.info(f"Loaded {len(catalog)} CDS view definitions from '{catalog_path}'.")
        return catalog

    def build_index(
        self,
        catalog_path: str = "cds_catalog.json",
        reset_collection: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end vector indexing in batch chunks:
        1. Loads cds_catalog.json
        2. Configures ChromaDB persistent collection with cosine distance metric
        3. Iterates over views in configurable batch chunks (default 128)
        4. Synthesizes documents combining view_name, description, ddl_code, annotations
        5. Upserts documents and metadata into ChromaDB collection
        6. Executes validation query to verify retrieval performance
        """
        total_start = time.perf_counter()
        catalog = self.load_catalog(catalog_path)
        total_views = len(catalog)

        if total_views == 0:
            logger.warning("CDS Catalog is empty. No views to index.")
            return {"total_indexed": 0, "total_runtime_sec": 0.0}

        # Handle collection reset if requested
        if reset_collection:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Existing collection '{self.collection_name}' deleted for clean rebuild (--reset).")
            except Exception:
                pass

        # Create or retrieve collection with cosine distance space
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        total_batches = math.ceil(total_views / self.batch_size)
        logger.info(
            f"Indexing {total_views} CDS views into ChromaDB collection '{self.collection_name}' "
            f"across {total_batches} batch chunk(s) (batch_size={self.batch_size})..."
        )

        seen_ids: Set[str] = set()
        indexed_count = 0
        batch_times: List[float] = []

        with BenchmarkTimer(f"Batch Upsert ({total_batches} Chunks)"):
            for batch_idx in range(0, total_views, self.batch_size):
                chunk = catalog[batch_idx : batch_idx + self.batch_size]
                batch_num = (batch_idx // self.batch_size) + 1
                batch_start = time.perf_counter()

                batch_ids: List[str] = []
                batch_docs: List[str] = []
                batch_metadatas: List[Dict[str, Any]] = []

                for i, cds in enumerate(chunk):
                    global_idx = batch_idx + i
                    view_name = (
                        cds.get("view_name") or
                        cds.get("cds_name") or
                        cds.get("name") or
                        f"CDS_{global_idx}"
                    ).strip()

                    ddl_source_name = (
                        cds.get("ddl_source_name") or
                        cds.get("source_name") or
                        view_name
                    ).strip()

                    description = (
                        cds.get("description") or
                        cds.get("label") or
                        ""
                    ).strip()

                    annotations = cds.get("annotations", [])
                    ddl_code = (cds.get("ddl_code") or cds.get("source") or "").strip()

                    # Ensure unique, stable document ID
                    doc_id = view_name
                    if doc_id in seen_ids:
                        doc_id = f"{view_name}_{global_idx}"
                    seen_ids.add(doc_id)

                    # Synthesize semantic text payload
                    doc_text = CDSDocumentSynthesizer.build_text_representation(cds)

                    # Prepare snippet for prompt / display preview
                    first_lines = [line.strip() for line in ddl_code.splitlines() if line.strip()]
                    ddl_snippet = "\n".join(first_lines[:8]) if first_lines else f"define view {view_name}"

                    # Clean Core scope calculation
                    is_standard = (
                        (view_name.startswith("I_") or view_name.startswith("C_"))
                        and not view_name.startswith("Z19_")
                        and not view_name.startswith("P_")
                    )
                    is_rel_val = 1 if is_standard else 0
                    v_type_val = "standard" if is_standard else "custom"

                    # ChromaDB metadata values must be primitive types (str, int, float, bool)
                    metadata: Dict[str, Any] = {
                        "view_name": view_name,
                        "ddl_source_name": ddl_source_name,
                        "description": description,
                        "annotations_json": json.dumps(annotations),
                        "ddl_code": ddl_code,
                        "ddl_snippet": ddl_snippet,
                        "has_ddl_code": bool(ddl_code),
                        "is_released": is_rel_val,
                        "view_type": v_type_val,
                    }

                    # Preserve legacy fields if available
                    if "domain" in cds:
                        metadata["domain"] = str(cds["domain"])
                    if "fields" in cds:
                        metadata["fields_json"] = json.dumps(cds["fields"])
                    if "associations" in cds:
                        metadata["associations_json"] = json.dumps(cds["associations"])

                    batch_ids.append(doc_id)
                    batch_docs.append(doc_text)
                    batch_metadatas.append(metadata)

                # Upsert batch into ChromaDB collection
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas
                )

                batch_elapsed = time.perf_counter() - batch_start
                batch_times.append(batch_elapsed)
                indexed_count += len(chunk)

                throughput = len(chunk) / batch_elapsed if batch_elapsed > 0 else 0
                pct = (indexed_count / total_views) * 100.0
                logger.info(
                    f"[Batch {batch_num:>2}/{total_batches}] Indexed {len(chunk):>3} views "
                    f"({indexed_count:>4}/{total_views} - {pct:>5.1f}%) in {batch_elapsed:.3f}s "
                    f"({throughput:.1f} views/sec)"
                )

        total_runtime = time.perf_counter() - total_start
        final_count = collection.count()

        # Execute sample verification query
        logger.info("Executing retrieval verification test query on ChromaDB...")
        test_query = "Find CDS views for billing document items with customer details"
        verification_results = collection.query(
            query_texts=[test_query],
            n_results=min(3, final_count)
        )

        logger.info("=" * 75)
        logger.info("CHROMADB VECTOR INDEXING SUMMARY")
        logger.info("=" * 75)
        logger.info(f"Vector Database Path:        {os.path.abspath(self.db_path)}")
        logger.info(f"Collection Name:             {self.collection_name}")
        logger.info(f"Neural Embedding Model:      {self.model_name}")
        logger.info(f"Total CDS Views Cataloged:   {total_views}")
        logger.info(f"Total Vectors in Collection: {final_count}")
        logger.info(f"Batch Chunk Size:            {self.batch_size}")
        logger.info(f"Total Batches Processed:     {total_batches}")
        logger.info(f"Total Indexing Runtime:      {total_runtime:.3f} seconds")
        if total_runtime > 0:
            logger.info(f"Average Throughput:          {total_views / total_runtime:.1f} views/second")
        logger.info("-" * 75)
        logger.info(f"Verification Test Query:     '{test_query}'")
        if verification_results.get("ids") and verification_results["ids"][0]:
            top_ids = verification_results["ids"][0]
            top_distances = verification_results["distances"][0]
            for rank, (cand_id, dist) in enumerate(zip(top_ids, top_distances), 1):
                similarity_pct = max(0.0, min(100.0, (1.0 - dist) * 100.0))
                logger.info(f"  Match #{rank}: {cand_id:<32} [Score: {similarity_pct:.2f}% | Distance: {dist:.4f}]")
        logger.info("=" * 75)

        return {
            "total_views": total_views,
            "final_vectors": final_count,
            "total_batches": total_batches,
            "total_runtime_sec": round(total_runtime, 4),
            "db_path": os.path.abspath(self.db_path),
            "collection_name": self.collection_name
        }


# =============================================================================
# CLI PARSER & MAIN ENTRYPOINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for building the ChromaDB vector index."""
    parser = argparse.ArgumentParser(
        description="Build Persistent ChromaDB Vector Index for SAP S/4HANA CDS Views",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_catalog = "extracted_views.json" if os.path.exists("extracted_views.json") else "cds_catalog.json"
    parser.add_argument(
        "-i", "--input-catalog",
        default=default_catalog,
        help="Path to extracted CDS catalog JSON file (extracted_views.json or cds_catalog.json)"
    )
    parser.add_argument(
        "-d", "--db-path",
        default=os.getenv("CHROMA_DB_PATH", "./cds_vector_db"),
        help="Directory path for persistent ChromaDB storage"
    )
    parser.add_argument(
        "-c", "--collection-name",
        default=os.getenv("CHROMA_COLLECTION_NAME", "cds_views"),
        help="ChromaDB collection identifier name"
    )
    parser.add_argument(
        "-m", "--model-name",
        default=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        help="SentenceTransformers neural embedding model name"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=128,
        help="Number of CDS views to process per batch chunk"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing collection before indexing"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    logger.info("=" * 75)
    logger.info("SAP S/4HANA CDS ChromaDB Vector Index Pipeline (SentenceTransformers)")
    logger.info("=" * 75)

    try:
        indexer = CDSChromaIndexBuilder(
            db_path=args.db_path,
            collection_name=args.collection_name,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        indexer.build_index(
            catalog_path=args.input_catalog,
            reset_collection=args.reset
        )
        logger.info("ChromaDB vector collection built and persisted successfully.")
        return 0
    except Exception as e:
        logger.critical(f"ChromaDB Index build failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
