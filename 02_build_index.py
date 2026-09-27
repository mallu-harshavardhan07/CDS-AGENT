"""
=============================================================================
Module: 02_build_index.py
Description: Production-ready ChromaDB Vector Search Index Builder with
             SentenceTransformers embeddings for SAP S/4HANA CDS Views.
             Processes CDS catalogs in batch chunks to efficiently scale
             to thousands of views, persisting vectors to ./cds_vector_db.
Author: Senior AI / RAG Engineer
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
    searchable text strings combining name, description, domain, fields, and associations.
    """

    @staticmethod
    def build_text_representation(cds: Dict[str, Any]) -> str:
        """
        Creates a high-signal contextual document combining:
        1. CDS View Name & @EndUserText.label description
        2. Functional Domain / Module (SD, FI, MM, PP, CA, etc.)
        3. Primary Key Fields & Key Attribute Labels
        4. Field schema with data types & descriptions
        5. Associations, target views, and join relationships
        """
        cds_name = (cds.get("cds_name") or cds.get("name") or "").strip()
        label = (cds.get("label") or cds.get("description") or "").strip()
        domain = (cds.get("domain") or "").strip()
        fields = cds.get("fields", [])
        associations = cds.get("associations", [])

        # Categorize fields into Keys vs Attributes
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

        # Format Associations & Related Entities
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

        # Construct structured searchable representation
        doc_parts = [
            f"CDS View: {cds_name}",
            f"Business Label: {label}",
            f"Functional Domain: {domain}",
            f"Summary: Standard SAP S/4HANA CDS View '{cds_name}' ({label}) belonging to functional area {domain}.",
        ]

        if key_fields:
            doc_parts.append("Key Fields:\n  - " + "\n  - ".join(key_fields))

        if regular_fields:
            # Show up to first 50 fields to prevent exceeding embedding model context window
            capped_fields = regular_fields[:50]
            if len(regular_fields) > 50:
                capped_fields.append(f"... and {len(regular_fields) - 50} more fields")
            doc_parts.append("Attributes & Measures:\n  - " + "\n  - ".join(capped_fields))

        if assoc_descriptions:
            doc_parts.append("Associations & Relationships:\n  - " + "\n  - ".join(assoc_descriptions))

        return "\n".join(doc_parts)


# =============================================================================
# CHROMADB VECTOR INDEX BUILDER (BATCH PROCESSING)
# =============================================================================

class CDSChromaIndexBuilder:
    """
    Builds and manages a persistent ChromaDB vector collection using
    SentenceTransformers neural embeddings. Scales to thousands of CDS views
    via robust batch chunking.
    """

    def __init__(
        self,
        db_path: str = "./cds_vector_db",
        collection_name: str = "cds_views",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.batch_size = max(1, batch_size)

        if chromadb is None or embedding_functions is None:
            raise ImportError(
                "ChromaDB and SentenceTransformers are required.\n"
                "Please install them via: pip install chromadb sentence-transformers"
            )

        logger.info(f"Initializing Persistent ChromaDB client at: '{self.db_path}'")
        self.client = chromadb.PersistentClient(path=self.db_path)

        logger.info(f"Configuring SentenceTransformer embedding function: '{self.model_name}'")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
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
        2. Configures ChromaDB persistent collection with cosine distance
        3. Iterates over views in configurable batch chunks (default 64)
        4. Synthesizes documents, extracts schema metadata, and upserts to ChromaDB
        5. Executes validation query to verify retrieval performance
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
                logger.info(f"Existing collection '{self.collection_name}' deleted for clean rebuild.")
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
                    cds_name = (cds.get("cds_name") or cds.get("name") or "").strip()
                    label = (cds.get("label") or cds.get("description") or "").strip()
                    domain = (cds.get("domain") or "").strip()
                    raw_fields = cds.get("fields", [])
                    raw_associations = cds.get("associations", [])

                    # Ensure unique, stable document ID
                    doc_id = cds_name if cds_name else f"cds_view_{global_idx}"
                    if doc_id in seen_ids:
                        doc_id = f"{doc_id}_{global_idx}"
                    seen_ids.add(doc_id)

                    # Synthesize semantic text
                    doc_text = CDSDocumentSynthesizer.build_text_representation(cds)

                    # Extract primary key fields
                    key_fields = []
                    for f in raw_fields:
                        if isinstance(f, dict) and f.get("is_key"):
                            key_fields.append(f.get("name", ""))

                    # ChromaDB metadata values must be primitive types (str, int, float, bool)
                    # Complex nested lists/dicts are serialized to JSON strings
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
        test_query = "Sales order delivery status and customer billing document"
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
    parser.add_argument(
        "-i", "--input-catalog",
        default="cds_catalog.json",
        help="Path to extracted CDS catalog JSON file"
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
        default=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        help="SentenceTransformers neural embedding model name"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=64,
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
