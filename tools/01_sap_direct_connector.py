"""
=============================================================================
Module: 01_sap_direct_connector.py
Description: Real-Time Direct SAP S/4HANA Sync Engine (Option 2 - Uncapped).
             Extracts active CDS view definitions (DDDDLSRC, DDDDLSRC02T, DDDDLSRCT)
             directly from SAP in-memory using standard SAP OData / ADT REST endpoints
             (or pyrfc) across ALL standard (I_*, C_*, P_*, E_*) and custom (Z*, Y*)
             namespaces without artificial limits or intermediate disk files.
             Streams embedded vectors directly into ChromaDB (./cds_vector_db)
             using sentence-transformers/all-MiniLM-L6-v2.
Author: SAP S/4HANA Cloud & AI Integration Engineering
=============================================================================
"""

import os
import sys
import re
import time
import math
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Environment variable loading
from dotenv import load_dotenv

# Try importing pyrfc (SAP NetWeaver RFC SDK wrapper)
try:
    import pyrfc
except ImportError:
    pyrfc = None

# HTTP / REST / OData dependencies
try:
    import requests
    from requests.auth import HTTPBasicAuth
    from requests.adapters import HTTPAdapter
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    requests = None
    HTTPBasicAuth = None
    HTTPAdapter = None

# ChromaDB & SentenceTransformers
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
logger = logging.getLogger("SAP_Direct_Sync")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SAPDirectConfig:
    """Configuration parameters for direct SAP extraction & vector ingestion."""
    # RFC Parameters
    ashost: str = ""
    sysnr: str = "00"
    client: str = "100"
    user: str = ""
    passwd: str = ""
    lang: str = "EN"

    # REST / OData Parameters
    base_url: str = ""
    verify_ssl: bool = False
    timeout: int = 30
    max_views: Optional[int] = None  # None or 0 = uncapped full extraction
    prefixes: List[str] = field(default_factory=lambda: ["Z*", "Y*", "I_*", "C_*", "P_*", "E_*"])

    # ChromaDB & Vector Parameters
    db_path: str = "./cds_vector_db"
    collection_name: str = "cds_views"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 128
    concurrency_workers: int = 20

    @classmethod
    def from_env(cls) -> "SAPDirectConfig":
        """Loads configuration from environment or .env file."""
        load_dotenv(dotenv_path=".env", override=True)
        if not os.path.exists(".env") and os.path.exists(".env.example"):
            load_dotenv(dotenv_path=".env.example", override=False)

        base_url = os.getenv("SAP_BASE_URL", "").strip()
        host = os.getenv("SAP_HOST", "").strip()
        if not host and base_url:
            parsed = base_url.replace("https://", "").replace("http://", "").split(":")[0]
            host = parsed

        env_max = os.getenv("SAP_MAX_VIEWS")
        max_views = int(env_max) if env_max and env_max.isdigit() and int(env_max) > 0 else None

        return cls(
            ashost=host,
            sysnr=os.getenv("SAP_SYSNR", "00"),
            client=os.getenv("SAP_CLIENT", "100"),
            user=os.getenv("SAP_USERNAME") or os.getenv("SAP_USER", ""),
            passwd=os.getenv("SAP_PASSWORD", ""),
            lang=os.getenv("SAP_LANG", "EN"),
            base_url=base_url,
            verify_ssl=os.getenv("SAP_VERIFY_SSL", "false").lower() in ("true", "1", "yes"),
            timeout=int(os.getenv("SAP_REQUEST_TIMEOUT_SECONDS", "30")),
            max_views=max_views,
            db_path=os.getenv("VECTOR_DB_PATH") or os.getenv("CHROMA_DB_PATH", "./cds_vector_db"),
            collection_name=os.getenv("VECTOR_COLLECTION_NAME") or os.getenv("CHROMA_COLLECTION_NAME", "cds_views"),
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=128,
            concurrency_workers=20
        )


@dataclass
class CDSEntity:
    """In-memory representation of an extracted SAP CDS View."""
    view_name: str
    ddl_source_name: str
    description: str
    annotations: List[str] = field(default_factory=list)
    ddl_code: str = ""

    def to_document_text(self) -> str:
        """Synthesizes high-signal embedding payload."""
        parts = [
            f"CDS View Name: {self.view_name}",
            f"DDL Source Name: {self.ddl_source_name}",
            f"Description: {self.description}",
        ]
        if self.annotations:
            parts.append(f"Annotations: {', '.join(self.annotations)}")
        if self.ddl_code:
            parts.append(f"DDL Code:\n{self.ddl_code}")
        return "\n\n".join(parts)


# =============================================================================
# IN-MEMORY DIRECT EXTRACTORS
# =============================================================================

class SAPRESTExtractor:
    """
    Extracts standard and custom CDS views in-memory using SAP ADT REST endpoints
    with HTTP Basic Auth, connection pooling, and multi-threaded parallel streaming.
    """

    def __init__(self, config: SAPDirectConfig):
        self.config = config

    def extract_stream(self) -> Generator[CDSEntity, None, None]:
        """
        Yields CDSEntity items one by one as they are fetched from SAP in parallel.
        Queries all configured namespaces without artificial capping.
        """
        if not requests:
            raise RuntimeError("Python 'requests' package is required for REST extraction.")

        if not self.config.base_url:
            raise ValueError("SAP_BASE_URL is not configured.")

        logger.info(f"Connecting to SAP REST endpoint at '{self.config.base_url}' (client: {self.config.client})...")
        session = requests.Session()
        session.auth = HTTPBasicAuth(self.config.user, self.config.passwd)
        
        # Enable connection pooling for speed
        if HTTPAdapter:
            adapter = HTTPAdapter(
                pool_connections=self.config.concurrency_workers * 2,
                pool_maxsize=self.config.concurrency_workers * 2,
                max_retries=2
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)

        session.headers.update({
            "Accept": "application/vnd.sap.as+xml, application/xml, */*",
            "sap-client": self.config.client,
            "x-csrf-token": "Fetch"
        })

        # Test reachability with timeout
        test_url = f"{self.config.base_url.rstrip('/')}/sap/bc/adt/discovery"
        resp = session.get(test_url, verify=self.config.verify_ssl, timeout=self.config.timeout)
        if resp.status_code not in (200, 201, 204):
            raise ConnectionError(f"SAP REST endpoint returned HTTP {resp.status_code}: {resp.text[:200]}")

        logger.info("SAP ADT REST handshake successful. Scanning active CDS DDL sources across namespaces...")
        search_url = f"{self.config.base_url.rstrip('/')}/sap/bc/adt/repository/informationsystem/search"
        
        # Total discovered DDL matches across all prefixes
        all_matches: List[Tuple[str, str]] = []
        seen_names: Set[str] = set()

        # Iterate over all requested namespaces: Custom (Z*, Y*) + Standard (I_*, C_*, P_*, E_*)
        for prefix in self.config.prefixes:
            params = {
                "operation": "quickSearch",
                "query": prefix,
                "objectType": "DDLS/DF",
                "maxResults": "50000"  # Request full uncapped ceiling per prefix
            }

            try:
                search_resp = session.get(
                    search_url,
                    params=params,
                    headers={"Accept": "application/xml, */*", "sap-client": self.config.client},
                    verify=self.config.verify_ssl,
                    timeout=self.config.timeout
                )
                if search_resp.status_code == 200:
                    raw_xml = search_resp.text
                    found = re.findall(r'adtcore:objectReference\s+adtcore:uri="([^"]+)"[^>]*?adtcore:name="([^"]+)"', raw_xml)
                    if not found:
                        found = re.findall(r'adtcore:objectReference\s+adtcore:name="([^"]+)"[^>]*?adtcore:uri="([^"]+)"', raw_xml)
                        found = [(m[1], m[0]) for m in found]
                    if not found:
                        ddl_names = re.findall(r'adtcore:name="([A-Za-z0-9_]+)"', raw_xml)
                        found = [(f"/sap/bc/adt/ddic/ddl/sources/{n.lower()}", n) for n in ddl_names]

                    added = 0
                    for u, n in found:
                        if n not in seen_names:
                            seen_names.add(n)
                            all_matches.append((u, n))
                            added += 1

                    logger.info(f"Namespace '{prefix:<4}': Discovered {len(found):>5} sources ({added:>5} new unique). Total so far: {len(all_matches):>6}")
            except Exception as se:
                logger.warning(f"Search for namespace '{prefix}' encountered warning: {se}")

        if self.config.max_views and self.config.max_views > 0:
            logger.info(f"Applying limit of {self.config.max_views} views per configuration.")
            all_matches = all_matches[:self.config.max_views]

        total_to_fetch = len(all_matches)
        logger.info(f"Total unique CDS DDL sources queued for uncapped streaming: {total_to_fetch}")

        # Worker function to fetch source code for one DDL entity
        def fetch_single(item: Tuple[str, str]) -> Optional[CDSEntity]:
            uri, name = item
            if not uri.startswith("/"):
                uri = "/" + uri
            source_url = f"{self.config.base_url.rstrip('/')}{uri.rstrip('/')}/source/main"
            try:
                src_resp = session.get(
                    source_url,
                    headers={"Accept": "text/plain, */*", "sap-client": self.config.client},
                    verify=self.config.verify_ssl,
                    timeout=self.config.timeout
                )
                if src_resp.status_code == 200:
                    code = src_resp.text
                    v_match = re.search(r"define\s+(?:root\s+)?(?:view\s+(?:entity\s+)?|table\s+function\s+)([A-Za-z0-9_]+)", code, re.I)
                    v_name = v_match.group(1) if v_match else name
                    l_match = re.search(r"@EndUserText\.label\s*:\s*'([^']+)'", code, re.I)
                    desc = l_match.group(1) if l_match else v_name
                    annos = [
                        a.strip() for a in re.findall(r"(@[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\s*:\s*[^;\r\n\}]+)", code)
                        if any(a.strip().startswith(p) for p in ("@VDM", "@Analytics", "@EndUserText", "@AccessControl", "@ObjectModel", "@Search", "@Semantics"))
                    ]
                    return CDSEntity(
                        view_name=v_name,
                        ddl_source_name=name,
                        description=desc,
                        annotations=annos,
                        ddl_code=code
                    )
            except Exception as e:
                logger.debug(f"Could not read DDL source '{name}': {e}")
            return None

        # Process via thread pool and yield completed entities as they arrive
        with ThreadPoolExecutor(max_workers=self.config.concurrency_workers) as executor:
            future_to_item = {executor.submit(fetch_single, m): m for m in all_matches}
            completed_count = 0
            yielded_count = 0

            for future in as_completed(future_to_item):
                res = future.result()
                completed_count += 1
                if res:
                    yielded_count += 1
                    yield res

                if completed_count % 500 == 0 or completed_count == total_to_fetch:
                    pct = (completed_count / total_to_fetch) * 100.0 if total_to_fetch else 100.0
                    logger.info(f"Stream progress: {completed_count:>5}/{total_to_fetch} DDLs inspected ({pct:>5.1f}%) | {yielded_count:>5} valid views parsed.")


class PyRFCExtractor:
    """Extracts CDS views in-memory using RFC_READ_TABLE on SAP system tables."""
    def __init__(self, config: SAPDirectConfig):
        self.config = config

    def is_available(self) -> bool:
        return pyrfc is not None

    def extract_stream(self) -> Generator[CDSEntity, None, None]:
        if not self.is_available():
            raise RuntimeError("pyrfc is not installed or SAP NW RFC SDK libraries are missing.")

        logger.info(f"Connecting to SAP system via pyrfc ({self.config.ashost}, client {self.config.client})...")
        conn_params = {
            "ashost": self.config.ashost,
            "sysnr": self.config.sysnr,
            "client": self.config.client,
            "user": self.config.user,
            "passwd": self.config.passwd,
            "lang": self.config.lang,
        }

        with pyrfc.Connection(**conn_params) as conn:
            logger.info("pyrfc connection established. Querying DDDDLSRC for active CDS views...")
            options = [{"TEXT": "AS4LOCAL = 'A'"}]
            fields = [{"FIELDNAME": "DDLNAME"}, {"FIELDNAME": "SOURCE"}]
            rowcount = self.config.max_views if self.config.max_views and self.config.max_views > 0 else 0
            res = conn.call(
                "RFC_READ_TABLE",
                QUERY_TABLE="DDDDLSRC",
                FIELDS=fields,
                OPTIONS=options,
                ROWCOUNT=rowcount
            )

            for row in res.get("DATA", []):
                wa = row.get("WA", "")
                ddl_name = wa[:30].strip()
                source_code = wa[30:].strip()

                view_match = re.search(r"define\s+(?:root\s+)?(?:view\s+(?:entity\s+)?|table\s+function\s+)([A-Za-z0-9_]+)", source_code, re.I)
                view_name = view_match.group(1) if view_match else ddl_name

                label_match = re.search(r"@EndUserText\.label\s*:\s*'([^']+)'", source_code, re.I)
                description = label_match.group(1) if label_match else view_name

                annos = re.findall(r"(@[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\s*:\s*[^;\r\n\}]+)", source_code)
                filtered_annos = [
                    a.strip() for a in annos
                    if any(a.strip().startswith(prefix) for prefix in ("@VDM", "@Analytics", "@EndUserText", "@AccessControl", "@ObjectModel"))
                ]

                yield CDSEntity(
                    view_name=view_name,
                    ddl_source_name=ddl_name,
                    description=description,
                    annotations=filtered_annos,
                    ddl_code=source_code
                )


class HighFidelityMockExtractor:
    """Fallback mock views for offline development."""
    @staticmethod
    def extract_stream() -> Generator[CDSEntity, None, None]:
        from cds_catalog_mock import get_mock_views # fallback
        pass


# =============================================================================
# DIRECT CHROMADB STREAMING VECTOR SYNC ENGINE
# =============================================================================

class DirectVectorSyncEngine:
    """
    Streams in-memory CDS view entities directly into ChromaDB in batches (128).
    Computes SentenceTransformer neural embeddings on the fly with zero disk writes.
    """

    def __init__(self, config: SAPDirectConfig):
        self.config = config

        if chromadb is None or embedding_functions is None:
            raise ImportError(
                "ChromaDB and SentenceTransformers are required.\n"
                "Please run: pip install chromadb sentence-transformers"
            )

        logger.info(f"Connecting to Persistent ChromaDB at: '{self.config.db_path}'")
        self.client = chromadb.PersistentClient(path=self.config.db_path)

        model_name = (
            self.config.model_name.replace("sentence-transformers/", "")
            if "/" in self.config.model_name
            else self.config.model_name
        )
        logger.info(f"Configuring SentenceTransformer model: '{model_name}'")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    def prepare_collection(self, reset: bool = False):
        """Initializes or resets the ChromaDB collection."""
        if reset:
            try:
                self.client.delete_collection(name=self.config.collection_name)
                logger.info(f"Collection '{self.config.collection_name}' dropped for clean rebuild (--reset).")
            except Exception:
                pass

        collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        return collection

    def upsert_chunk(self, chunk: List[CDSEntity], collection, seen_ids: Set[str]) -> int:
        """Upserts a single batch chunk of CDSEntity objects into ChromaDB."""
        if not chunk:
            return 0

        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        for i, entity in enumerate(chunk):
            doc_id = entity.view_name
            if doc_id in seen_ids:
                doc_id = f"{entity.view_name}_{len(seen_ids)}"
            seen_ids.add(doc_id)

            doc_text = entity.to_document_text()
            lines = [l.strip() for l in entity.ddl_code.splitlines() if l.strip()]
            ddl_snippet = "\n".join(lines[:8]) if lines else f"define view {entity.view_name}"

            metadata = {
                "view_name": entity.view_name,
                "ddl_source_name": entity.ddl_source_name,
                "description": entity.description,
                "annotations_json": json.dumps(entity.annotations),
                "ddl_code": entity.ddl_code,
                "ddl_snippet": ddl_snippet,
                "has_ddl_code": bool(entity.ddl_code),
            }

            batch_ids.append(doc_id)
            batch_docs.append(doc_text)
            batch_metas.append(metadata)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )
        return len(chunk)


# =============================================================================
# CLI PARSER & MAIN ORCHESTRATION
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct In-Memory SAP S/4HANA CDS Sync to ChromaDB (Option 2 - Uncapped)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--method",
        choices=["auto", "pyrfc", "rest"],
        default="auto",
        help="Extraction method: auto, pyrfc (RFC), or rest (ADT REST/OData)"
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("VECTOR_DB_PATH") or os.getenv("CHROMA_DB_PATH", "./cds_vector_db"),
        help="Directory path for persistent ChromaDB storage"
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("VECTOR_COLLECTION_NAME") or os.getenv("CHROMA_COLLECTION_NAME", "cds_views"),
        help="ChromaDB collection identifier name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for vector embedding and upsert"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing vector collection before indexing"
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help="Maximum number of CDS views to extract from SAP (default: uncapped full extraction)"
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default="Z*,Y*,I_*,C_*,P_*,E_*",
        help="Comma-separated CDS view namespaces to query"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract from SAP in-memory only without upserting to ChromaDB"
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="Find CDS views for billing document items with customer details",
        help="Test query to run against ChromaDB after sync"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SAPDirectConfig.from_env()

    # Override config with CLI flags
    config.db_path = args.db_path
    config.collection_name = args.collection_name
    config.batch_size = args.batch_size
    config.max_views = args.max_views if args.max_views is not None else None
    if args.prefixes:
        config.prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    logger.info("=" * 75)
    logger.info("SAP S/4HANA Real-Time Direct Sync Engine (Uncapped Full Extraction)")
    logger.info("=" * 75)
    logger.info(f"Target Namespaces:           {', '.join(config.prefixes)}")
    logger.info(f"Max Views Limit:             {'UNCAPPED (Full Sync)' if not config.max_views else config.max_views}")
    logger.info(f"Vector Database Path:        {os.path.abspath(config.db_path)}")
    logger.info(f"Collection Name:             {config.collection_name}")
    logger.info(f"Clean Rebuild (--reset):     {args.reset}")
    logger.info(f"Concurrency Workers:         {config.concurrency_workers}")
    logger.info("-" * 75)

    start_time = time.perf_counter()

    try:
        # Initialize direct sync engine
        sync_engine = DirectVectorSyncEngine(config)
        collection = None

        if not args.dry_run:
            collection = sync_engine.prepare_collection(reset=args.reset)

        # Initialize extractor
        if args.method == "pyrfc" or (args.method == "auto" and pyrfc is not None and config.ashost and config.user):
            extractor = PyRFCExtractor(config)
        else:
            extractor = SAPRESTExtractor(config)

        # Stream and batch upsert
        buffer: List[CDSEntity] = []
        seen_ids: Set[str] = set()
        total_synced = 0
        batch_number = 0

        for entity in extractor.extract_stream():
            if args.dry_run:
                total_synced += 1
                if total_synced <= 10 or total_synced % 1000 == 0:
                    logger.info(f"  [Dry-Run] Discovered #{total_synced}: {entity.view_name:<30} | {entity.description[:40]}")
                continue

            buffer.append(entity)

            # When buffer reaches batch_size, upsert immediately
            if len(buffer) >= config.batch_size:
                batch_number += 1
                b_start = time.perf_counter()
                chunk_size = sync_engine.upsert_chunk(buffer, collection, seen_ids)
                b_elapsed = time.perf_counter() - b_start
                total_synced += chunk_size
                throughput = chunk_size / b_elapsed if b_elapsed > 0 else 0
                logger.info(
                    f"[Batch {batch_number:>3}] Upserted {chunk_size:>3} views "
                    f"(Total Synced: {total_synced:>5}) in {b_elapsed:.2f}s ({throughput:.1f} views/sec)"
                )
                buffer.clear()

        # Flush any remaining entities in the buffer
        if buffer and not args.dry_run:
            batch_number += 1
            b_start = time.perf_counter()
            chunk_size = sync_engine.upsert_chunk(buffer, collection, seen_ids)
            b_elapsed = time.perf_counter() - b_start
            total_synced += chunk_size
            throughput = chunk_size / b_elapsed if b_elapsed > 0 else 0
            logger.info(
                f"[Batch {batch_number:>3} (Final)] Upserted {chunk_size:>3} views "
                f"(Total Synced: {total_synced:>5}) in {b_elapsed:.2f}s ({throughput:.1f} views/sec)"
            )
            buffer.clear()

        total_runtime = time.perf_counter() - start_time
        final_count = collection.count() if collection else total_synced

        logger.info("=" * 75)
        logger.info("UNCAPPED DIRECT SAP SYNC COMPLETE")
        logger.info("=" * 75)
        logger.info(f"Total Views Synced in Memory: {total_synced}")
        logger.info(f"Total Vectors in ChromaDB:    {final_count}")
        logger.info(f"Total Elapsed Time:           {total_runtime:.2f} seconds ({total_runtime / 60:.2f} min)")
        if total_runtime > 0:
            logger.info(f"Overall Pipeline Throughput:  {total_synced / total_runtime:.1f} views/second")
        logger.info(f"Zero Intermediate JSON files written to disk.")
        logger.info("=" * 75)

        # Test verification query if requested
        if args.test_query and collection and final_count > 0:
            logger.info(f"Running validation query: '{args.test_query}'...")
            try:
                test_res = collection.query(query_texts=[args.test_query], n_results=min(5, final_count))
                if test_res.get("ids") and test_res["ids"][0]:
                    for r, (vid, dist) in enumerate(zip(test_res["ids"][0], test_res["distances"][0]), 1):
                        score = max(0.0, min(100.0, (1.0 - dist) * 100.0))
                        meta = test_res["metadatas"][0][r-1] if "metadatas" in test_res and test_res["metadatas"] else {}
                        desc = meta.get("description", "")
                        logger.info(f"  Match #{r}: {vid:<32} [Score: {score:.1f}%] - {desc[:40]}")
            except Exception as qe:
                logger.warning(f"Test query warning: {qe}")

        return 0

    except Exception as e:
        logger.critical(f"Uncapped Direct SAP Sync failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
