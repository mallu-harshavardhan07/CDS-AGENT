"""
=============================================================================
Module: 04_delta_sync.py
Description: Production-ready Incremental Delta Sync Pipeline for SAP S/4HANA
             Custom CDS Views (Z_... / Y_...). Uses pure-Python TF-IDF
             vectorizer (scikit-learn) and atomic disk persistence into
             cds_tfidf.pkl, eliminating native C++ DLL dependencies.
Author: DevOps & Data Engineering Developer
=============================================================================
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
from typing import Dict, List, Any, Tuple, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("CDS_Delta_Sync")


# =============================================================================
# BENCHMARK TIMER UTILITY
# =============================================================================

class BenchmarkTimer:
    """High-resolution execution timer for profiling delta sync stages."""
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
    """Standardized semantic document synthesizer matching 02_build_index.py."""
    @staticmethod
    def build_text_representation(cds: Dict[str, Any]) -> str:
        cds_name = (cds.get("cds_name") or cds.get("name") or "").strip()
        label = (cds.get("label") or cds.get("description") or "").strip()
        domain = cds.get("domain", "").strip()
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
            else:
                name = str(f)
                ftype = "CHAR"
                flabel = str(f)
                is_key = False
            field_desc = f"{name} ({ftype}): {flabel}"
            if is_key:
                key_fields.append(field_desc)
            else:
                regular_fields.append(field_desc)

        assoc_descriptions: List[str] = []
        for a in associations:
            if isinstance(a, dict):
                alias = a.get("alias", "")
                target = a.get("target_cds", "")
                cardinality = a.get("cardinality", "[0..1]")
                cond = a.get("condition", "").replace("\n", " ").strip()
                assoc_descriptions.append(f"{alias} -> {target} {cardinality} ON {cond}")
            else:
                assoc_descriptions.append(f"Association: {str(a)}")

        doc_parts = [
            f"CDS View: {cds_name}",
            f"Business Label: {label}",
            f"Functional Domain: {domain}",
            f"Summary: Custom/Standard SAP S/4HANA CDS View '{cds_name}' ({label}) belonging to {domain}.",
        ]

        if key_fields:
            doc_parts.append("Key Fields:\n  - " + "\n  - ".join(key_fields))

        if regular_fields:
            doc_parts.append("Attributes & Measures:\n  - " + "\n  - ".join(regular_fields))

        if assoc_descriptions:
            doc_parts.append("Associations & Relationships:\n  - " + "\n  - ".join(assoc_descriptions))

        return "\n".join(doc_parts)


# =============================================================================
# ATOMIC DISK PERSISTENCE UTILITIES
# =============================================================================

def atomic_save_json(filepath: str, data: Any, indent: int = 2) -> None:
    """Safely writes JSON data to disk using an atomic rename operation."""
    abs_path = os.path.abspath(filepath)
    temp_path = f"{abs_path}.tmp.{int(time.time() * 1000)}"
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, abs_path)
    logger.debug(f"Atomically committed: '{abs_path}'")


def atomic_save_pickle(filepath: str, data: Any) -> None:
    """Safely writes pickled model data to disk using an atomic rename operation."""
    abs_path = os.path.abspath(filepath)
    temp_path = f"{abs_path}.tmp.{int(time.time() * 1000)}"
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    with open(temp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, abs_path)
    logger.debug(f"Atomically committed TF-IDF pickle: '{abs_path}'")


# =============================================================================
# DELTA SYNC ENGINE (TF-IDF Scikit-Learn)
# =============================================================================

class CDSDeltaSyncEngine:
    """
    Orchestrates validation, deduplication, incremental TF-IDF re-indexing,
    and atomic state synchronization.
    """

    def __init__(
        self,
        catalog_path: str = "cds_catalog.json",
        pickle_path: str = "cds_tfidf.pkl",
        mapping_path: str = "cds_mapping.json",
        dry_run: bool = False
    ):
        self.catalog_path = catalog_path
        self.pickle_path = pickle_path
        self.mapping_path = mapping_path
        self.dry_run = dry_run

        self.catalog: List[Dict[str, Any]] = []
        self.mapping_data: Dict[str, Any] = {}
        self.index_to_cds: Dict[str, Any] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None

    def load_existing_assets(self) -> None:
        """Loads catalog, mapping lookup table, and TF-IDF pickle safely."""
        logger.info("Loading existing catalog and TF-IDF model assets...")

        # 1. Load Catalog
        if os.path.exists(self.catalog_path):
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)
            logger.info(f"Loaded {len(self.catalog)} records from catalog '{self.catalog_path}'.")
        else:
            logger.warning(f"Catalog file '{self.catalog_path}' not found. Starting with empty catalog.")
            self.catalog = []

        # 2. Load Mapping Table
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping_data = json.load(f)
                self.index_to_cds = self.mapping_data.get("index_to_cds", {})
            logger.info(f"Loaded {len(self.index_to_cds)} mapped records from '{self.mapping_path}'.")
        else:
            logger.warning(f"Mapping file '{self.mapping_path}' not found. Initializing new mapping.")
            self.mapping_data = {
                "metadata": {
                    "total_views": 0,
                    "retriever": "TF-IDF (scikit-learn)",
                    "catalog_source": os.path.abspath(self.catalog_path),
                },
                "index_to_cds": {}
            }
            self.index_to_cds = self.mapping_data["index_to_cds"]

        # 3. Load TF-IDF Pickle
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, "rb") as f:
                    data = pickle.load(f)
                    self.vectorizer = data.get("vectorizer")
                    self.tfidf_matrix = data.get("tfidf_matrix")
                logger.info(f"Loaded TF-IDF model from '{self.pickle_path}'.")
            except Exception as e:
                logger.warning(f"Failed to load pickle '{self.pickle_path}': {e}")

    def load_delta_payload(self, delta_file_path: Optional[str]) -> List[Dict[str, Any]]:
        """Loads incoming delta views from JSON file or generates sample custom views."""
        if delta_file_path and os.path.exists(delta_file_path):
            logger.info(f"Reading delta payload from '{delta_file_path}'...")
            with open(delta_file_path, "r", encoding="utf-8") as f:
                delta_views = json.load(f)
            if not isinstance(delta_views, list):
                raise ValueError("Delta file must contain a JSON array of CDS views.")
            logger.info(f"Read {len(delta_views)} delta CDS views from '{delta_file_path}'.")
            return delta_views

        logger.info("No delta file provided or found. Utilizing built-in custom SAP CDS delta views (Z_... / Y_...)...")
        return self._get_sample_delta_views()

    def _get_sample_delta_views(self) -> List[Dict[str, Any]]:
        """Sample newly created custom S/4HANA CDS views for demonstration."""
        return [
            {
                "cds_name": "Z_I_SalesOrderCustomAnalytics",
                "label": "Custom Sales Order Analytics with Loyalty Category",
                "domain": "Sales & Distribution (SD)",
                "fields": [
                    {"name": "SalesOrder", "type": "CHAR", "is_key": True, "label": "Sales Order"},
                    {"name": "SoldToParty", "type": "CHAR", "is_key": False, "label": "Customer ID"},
                    {"name": "CustomerLoyaltyTier", "type": "CHAR", "is_key": False, "label": "Loyalty Tier (Gold/Silver)"},
                    {"name": "TotalNetAmount", "type": "CURR", "is_key": False, "label": "Total Net Amount"},
                    {"name": "CustomMarginPercentage", "type": "DEC", "is_key": False, "label": "Calculated Profit Margin"}
                ],
                "associations": [
                    {
                        "alias": "_Customer",
                        "target_cds": "I_Customer",
                        "cardinality": "[0..1]",
                        "condition": "$projection.SoldToParty = _Customer.Customer"
                    }
                ]
            },
            {
                "cds_name": "Z_I_MaterialValuationRealtime",
                "label": "Real-time Material Valuation and Stock Margin",
                "domain": "Materials Management (MM)",
                "fields": [
                    {"name": "Material", "type": "CHAR", "is_key": True, "label": "Material Number"},
                    {"name": "Plant", "type": "CHAR", "is_key": True, "label": "Plant"},
                    {"name": "StorageLocation", "type": "CHAR", "is_key": True, "label": "Storage Location"},
                    {"name": "ValuationAmount", "type": "CURR", "is_key": False, "label": "Current Valuation Amount"},
                    {"name": "Currency", "type": "CHAR", "is_key": False, "label": "Currency"}
                ],
                "associations": [
                    {
                        "alias": "_Product",
                        "target_cds": "I_Product",
                        "cardinality": "[0..1]",
                        "condition": "$projection.Material = _Product.Product"
                    }
                ]
            }
        ]

    def deduplicate_and_plan(
        self, delta_views: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compares delta views against existing mapping by cds_name."""
        existing_names_map: Dict[str, str] = {}
        for idx_str, meta in self.index_to_cds.items():
            name = meta.get("cds_name")
            if name:
                existing_names_map[name.upper()] = idx_str

        to_insert: List[Dict[str, Any]] = []
        to_update: List[Dict[str, Any]] = []

        for view in delta_views:
            cds_name = view.get("cds_name", "").strip()
            if not cds_name:
                continue

            if cds_name.upper() in existing_names_map:
                existing_id = existing_names_map[cds_name.upper()]
                view["_mapped_id"] = existing_id
                to_update.append(view)
                logger.info(f"Delta View '{cds_name}': ALREADY EXISTS (ID: {existing_id}) -> Marked for Metadata UPDATE.")
            else:
                to_insert.append(view)
                logger.info(f"Delta View '{cds_name}': NEW VIEW -> Marked for INSERTION.")

        return to_insert, to_update

    def run_sync(self, delta_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Executes incremental sync and re-vectorizes TF-IDF model."""
        sync_start = time.perf_counter()
        benchmarks: Dict[str, float] = {}

        # Stage 1: Load Assets
        with BenchmarkTimer("Load Assets") as timer:
            self.load_existing_assets()
            benchmarks["load_assets_sec"] = timer.elapsed

        # Stage 2: Deduplication
        with BenchmarkTimer("Deduplication & Validation") as timer:
            delta_views = self.load_delta_payload(delta_file_path)
            new_views, updated_views = self.deduplicate_and_plan(delta_views)
            benchmarks["deduplication_sec"] = timer.elapsed

        if not new_views and not updated_views:
            logger.info("No delta changes detected. Pipeline up-to-date.")
            return {"status": "NOOP", "new_count": 0, "updated_count": 0}

        # Stage 3: Merge Catalog and Mapping Table
        with BenchmarkTimer("Update Catalog & Mapping Table") as timer:
            current_total = len(self.index_to_cds)

            # Insert new views
            for i, view in enumerate(new_views):
                assigned_id = current_total + i
                doc_text = CDSDocumentSynthesizer.build_text_representation(view)
                key_fields = [f["name"] for f in view.get("fields", []) if f.get("is_key")]

                self.index_to_cds[str(assigned_id)] = {
                    "id": assigned_id,
                    "cds_name": view.get("cds_name"),
                    "label": view.get("label"),
                    "domain": view.get("domain"),
                    "key_fields": key_fields,
                    "field_count": len(view.get("fields", [])),
                    "association_count": len(view.get("associations", [])),
                    "fields": view.get("fields", []),
                    "associations": view.get("associations", []),
                    "document_text": doc_text,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "is_custom": True
                }
                self.catalog.append(view)

            # Update existing views
            for view in updated_views:
                mapped_id = str(view["_mapped_id"])
                doc_text = CDSDocumentSynthesizer.build_text_representation(view)
                key_fields = [f["name"] for f in view.get("fields", []) if f.get("is_key")]

                self.index_to_cds[mapped_id].update({
                    "label": view.get("label"),
                    "domain": view.get("domain"),
                    "key_fields": key_fields,
                    "field_count": len(view.get("fields", [])),
                    "association_count": len(view.get("associations", [])),
                    "fields": view.get("fields", []),
                    "associations": view.get("associations", []),
                    "document_text": doc_text,
                    "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                })

                for c_idx, c_item in enumerate(self.catalog):
                    if c_item.get("cds_name") == view.get("cds_name"):
                        self.catalog[c_idx] = {k: v for k, v in view.items() if k != "_mapped_id"}
                        break

            self.mapping_data["metadata"]["total_views"] = len(self.index_to_cds)
            self.mapping_data["metadata"]["last_delta_sync"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            benchmarks["merge_sec"] = timer.elapsed

        # Stage 4: Re-vectorize TF-IDF Matrix with all documents
        with BenchmarkTimer("TF-IDF Vectorization") as timer:
            if TfidfVectorizer is None:
                raise ImportError("scikit-learn is missing. Install via: pip install scikit-learn")

            documents = []
            for idx_str, item in sorted(self.index_to_cds.items(), key=lambda x: int(x[0])):
                documents.append(item.get("document_text", ""))

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            benchmarks["tfidf_vectorization_sec"] = timer.elapsed
            logger.info(f"Re-vectorized TF-IDF matrix for {len(documents)} views. Features: {self.tfidf_matrix.shape[1]}")

        # Stage 5: Atomic Persistence
        with BenchmarkTimer("Atomic Disk Persistence") as timer:
            if not self.dry_run:
                # 1. Atomic save TF-IDF Pickle
                pickle_payload = {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "cds_mapping": self.index_to_cds,
                    "catalog": self.catalog,
                    "total_views": len(self.index_to_cds),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                }
                atomic_save_pickle(self.pickle_path, pickle_payload)

                # 2. Atomic save Mapping JSON
                atomic_save_json(self.mapping_path, self.mapping_data)

                # 3. Atomic save Catalog JSON
                clean_catalog = [{k: v for k, v in item.items() if k != "_mapped_id"} for item in self.catalog]
                atomic_save_json(self.catalog_path, clean_catalog)
                logger.info("All delta assets successfully saved atomically to disk.")
            else:
                logger.info("DRY-RUN MODE: Skipped disk writes. In-memory validation complete.")
            benchmarks["atomic_persistence_sec"] = timer.elapsed

        total_elapsed = time.perf_counter() - sync_start
        benchmarks["total_sync_sec"] = total_elapsed

        logger.info("=" * 70)
        logger.info("TF-IDF DELTA SYNC REPORT")
        logger.info("=" * 70)
        logger.info(f"New Views Inserted:             {len(new_views)}")
        logger.info(f"Existing Views Updated:         {len(updated_views)}")
        logger.info(f"Total Views in Catalog:         {len(self.index_to_cds)}")
        logger.info(f"TF-IDF Vocabulary Size:         {self.tfidf_matrix.shape[1]:,} features")
        logger.info(f"Total Delta Sync Runtime:       {benchmarks['total_sync_sec']:.4f} s")
        logger.info("=" * 70)

        return {
            "status": "SUCCESS",
            "dry_run": self.dry_run,
            "new_views": len(new_views),
            "updated_views": len(updated_views),
            "total_views": len(self.index_to_cds),
            "benchmarks": benchmarks
        }


# =============================================================================
# CLI PARSER & MAIN ENTRYPOINT
# =============================================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAP S/4HANA CDS Incremental Delta Sync (Pure-Python TF-IDF Stack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--delta-file",
        default=None,
        help="Path to JSON file containing delta custom CDS views (Z_... / Y_...)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and benchmark delta sync without writing changes to disk"
    )
    parser.add_argument(
        "--catalog",
        default="cds_catalog.json",
        help="Path to cds_catalog.json"
    )
    parser.add_argument(
        "--pickle",
        default="cds_tfidf.pkl",
        help="Path to cds_tfidf.pkl"
    )
    parser.add_argument(
        "--mapping",
        default="cds_mapping.json",
        help="Path to cds_mapping.json"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    logger.info("=" * 70)
    logger.info("SAP S/4HANA CDS Incremental Delta Sync (TF-IDF Scikit-Learn)")
    logger.info("=" * 70)

    try:
        engine = CDSDeltaSyncEngine(
            catalog_path=args.catalog,
            pickle_path=args.pickle,
            mapping_path=args.mapping,
            dry_run=args.dry_run
        )
        result = engine.run_sync(delta_file_path=args.delta_file)
        logger.info(f"Delta sync process completed with status: {result['status']}.")
        return 0
    except Exception as e:
        logger.critical(f"Delta sync pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
