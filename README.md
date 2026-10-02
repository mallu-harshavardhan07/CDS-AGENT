# SAP S/4HANA CDS Metadata Extractor

A production-grade Python extraction tool designed for SAP S/4HANA data integration pipelines. It connects to SAP S/4HANA via standard **ABAP Development Tools (ADT) REST APIs**, extracts active Core Data Services (CDS) view definitions, parses annotations, fields, and associations, and compiles a clean data catalog in JSON format.

---

## Features

- **Standard SAP REST/ADT Integration**: Queries `/sap/bc/adt/repository/informationsystem/search` and `/sap/bc/adt/ddic/ddls/sources/{name}` with CSRF token handshake and session retention.
- **Robust DDL Parser**: Extracts `@EndUserText.label`, functional domain tags (SD, FI, MM, CO, PP, etc.), key fields (`is_key: true`), data types, and CDS associations (`[0..1]`, `[0..*]`, join conditions).
- **Graceful Mock Fallback**: If the SAP server is offline, unreachable, or in a development sandbox without SAP VPN access, the extractor automatically falls back to an enterprise-grade mock dataset of standard SAP VDM views (`I_SalesOrder`, `I_SalesOrderItem`, `I_JournalEntry`, `I_PurchaseOrder`, `I_MaterialStock`, `I_Customer`) so downstream pipelines can be tested immediately.
- **Zero Mandatory Dependencies**: Built with Python 3 standard library (`urllib`, `ssl`, `json`, `re`, `logging`, `dataclasses`).

---

## Output JSON Schema

The extracted catalog is saved directly to `cds_catalog.json` with the following clean structure:

```json
[
  {
    "name": "I_SalesOrder",
    "description": "Sales Order Header Data",
    "domain": "SD",
    "fields": [
      "SalesOrder",
      "SalesOrderType",
      "TotalNetAmount",
      "CustomerName"
    ],
    "associations": [
      "_Item",
      "_Partner"
    ]
  }
]
```

---

## Pipeline Architecture

1. **Phase 1: Metadata Extraction (`01_extract.py`)**
   - Connects to SAP S/4HANA ADT REST endpoints with CSRF token handshake and session retention.
   - Parses active CDS DDL definitions into structured JSON ([cds_catalog.json](file:///c:/Users/harshavardhan.mallu/Desktop/cds-agent/cds_catalog.json)).
   - Gracefully falls back to high-fidelity mock S/4HANA VDM views if SAP is offline.

2. **Phase 2: ChromaDB Neural Vector Indexing (`02_build_index.py`)**
   - Synthesizes searchable text representations combining CDS Name, Label, Domain, Key Fields, Attributes, and Association topology.
   - Computes high-density neural embeddings using **SentenceTransformers** (`all-MiniLM-L6-v2`).
   - Processes the CDS catalog in **configurable batch chunks** (default: 64 views per chunk) to seamlessly scale to thousands of views without memory bottlenecks.
   - Persists the vector collection to `./cds_vector_db` using **ChromaDB's PersistentClient** with cosine distance metric (`hnsw:space: cosine`).
   - Eliminates fragile in-memory arrays and provides persistent on-disk retrieval with sub-millisecond query latency.

3. **Phase 3: FastAPI RAG Evaluation Core (`03_server.py`)**
   - Connects to the persistent ChromaDB collection in `./cds_vector_db` on startup (with automatic batch indexing fallback if empty).
   - Keeps total loaded CDS views in sync on startup and across the `/health` endpoint and UI status pill.
   - Queries ChromaDB for top-K candidates using SentenceTransformer neural embeddings and calculates cosine similarity percentages.
   - Evaluates against calibrated Clean Core decision thresholds:
     - **REUSE** (Match Score >= `REUSE_THRESHOLD`, default 25%): Recommends standard SAP CDS view with usage guidelines.
     - **EXTEND** (Match Score >= `EXTEND_THRESHOLD` and < `REUSE_THRESHOLD`, default 15%-25%): Generates `EXTEND VIEW ENTITY` ABAP DDL code to append missing fields non-disruptively.
     - **CREATE** (Match Score < `EXTEND_THRESHOLD`, default < 15%): Generates full `DEFINE VIEW ENTITY Z_...` ABAP DDL code following Virtual Data Model (VDM) standards.
   - Supports OpenAI API, local LLMs (Ollama / vLLM), or deterministic S/4HANA rules engine.
   - Serves the 4-step wizard web interface directly at `http://localhost:8000`.

4. **Phase 4: Incremental Delta Sync (`04_delta_sync.py`)**
   - Automatically ingests newly created or modified custom S/4HANA views (`Z_...` / `Y_...`).
   - Deduplicates views against existing mapping by `cds_name`.
   - Re-vectorizes TF-IDF representations and saves `cds_tfidf.pkl` using atomic rename operations.
   - Profiles execution metrics and supports dry-run testing (`--dry-run`).

---

## Output Catalog & Index Schema

- **Extracted Metadata Catalog**: [cds_catalog.json](file:///c:/Users/harshavardhan.mallu/Desktop/cds-agent/cds_catalog.json)
- **Serialized TF-IDF Model**: `cds_tfidf.pkl`
- **Lookup & Metadata Mapping**: [cds_mapping.json](file:///c:/Users/harshavardhan.mallu/Desktop/cds-agent/cds_mapping.json)
- **Sample Delta Payload**: [delta_views_sample.json](file:///c:/Users/harshavardhan.mallu/Desktop/cds-agent/delta_views_sample.json)

---

## Quickstart

### 1. Installation
Install core requirements:
```bash
pip install -r requirements.txt
```

### 2. Option 1: SAP ABAP CDS Extraction via File (`Z_EXPORT_CDS_CATALOG.abap`)
In SAP GUI (SE38/SE80) on SAP S/4HANA:
1. Create executable report `Z_EXPORT_CDS_CATALOG`.
2. Paste the source code from [Z_EXPORT_CDS_CATALOG.abap](file:///c:/Users/mallu/CDS-Agent/Z_EXPORT_CDS_CATALOG.abap).
3. Execute (`F8`), specify selection criteria (e.g., `s_ddl = I_*` for standard VDM views), and save the extracted `cds_catalog.json` to your local machine.
4. Populate ChromaDB from the JSON file:
   ```bash
   python 02_build_index.py --batch-size 128 --reset
   ```

### 3. Option 2: Direct Real-Time SAP In-Memory Connector (`01_sap_direct_connector.py`)
Directly extracts active CDS views (`DDDDLSRC`) in-memory via `pyrfc` (RFC) or standard SAP OData / ADT REST endpoints and immediately embeds & upserts into ChromaDB without intermediate JSON file writes:
```bash
# Direct in-memory sync and vector indexing (auto detects pyrfc / REST / fallback):
python 01_sap_direct_connector.py --reset

# Specific method or dry-run:
python 01_sap_direct_connector.py --method auto --batch-size 128 --reset
python 01_sap_direct_connector.py --dry-run
```

### 4. Part 3: RAG Search & Retrieval Engine (`03_rag_search.py`)
Search CDS views using natural language and generate Clean Core LLM prompts:
```bash
# Direct semantic search with top 3 results and LLM prompt:
python 03_rag_search.py -q "Find CDS views for billing document items with customer details" --top-k 3 --show-prompt

# Launch interactive REPL session:
python 03_rag_search.py
```

### 5. Part 4: Clean Core ABAP CDS Code Generator Agent (`04_cds_agent.py`)
Automatically generate production-ready, Clean Core compliant ABAP CDS View Entities based on natural language queries and RAG context:
```bash
# Run in deterministic Clean Core mock mode:
python 04_cds_agent.py -q "Create a custom CDS view entity for sales order header and item details with customer name" --mock

# Export generated view entity directly to an .asddls file:
python 04_cds_agent.py -q "Create a custom CDS view entity for sales order header and item details with customer name" --mock -o Z_SalesOrderDetails.asddls
```

### 6. Part 5: Launch Streamlit Web UI Dashboard (`app.py`)
Launch the interactive Streamlit dashboard on `http://localhost:8501`:
```bash
streamlit run app.py
```
Features:
- Instant semantic search across all **89,401 live CDS views** in ChromaDB.
- Clean Core automated decision badge (`REUSE`, `EXTEND`, `CREATE`).
- Syntax-highlighted ABAP CDS DDL editor with one-click `.asddls` file download.

### 7. Part 6: Start FastAPI Core Server & 4-Step Wizard Web UI (`03_server.py`)
```bash
python 03_server.py
# Or using uvicorn directly:
uvicorn 03_server:app --host 0.0.0.0 --port 8000 --reload
```
Once the server is running, navigate directly to **`http://localhost:8000`** in any web browser to access the 4-Step Wizard Web Interface! (Or open `index.html` directly).

### 8. Part 7: Run Incremental Delta Sync
**Dry-run Validation:**
```bash
python 04_delta_sync.py --delta-file delta_views_sample.json --dry-run
```

**Execute Atomic Sync:**
```bash
python 04_delta_sync.py --delta-file delta_views_sample.json
```

---

## 4-Step Wizard Web Interface Features

- **Step 1 (Domain Selection)**: Interactive cards for SAP modules (`SD`, `FI`, `MM`, `CO`, `PP`, `PM`) with sample CDS views.
- **Step 2 (Entities & Objective)**: Dynamic entity tag input, pre-configured suggestion chips, and sample user story templates.
- **Step 3 (Field Selection)**: Domain-aware recommended field tags (click-to-add), chip removals, and free-form field input.
- **Step 4 (Review & Submit)**: Form summary, one-click evaluation trigger with active pulsing animation.
- **Dynamic Results**:
  - Colored architectural status badge: **`REUSE` (Green)**, **`EXTEND` (Amber)**, **`CREATE` (Blue)**.
  - Cosine similarity score meter and target CDS view info.
  - LLM Clean Core architectural justification.
  - Syntax-highlighted ABAP Core Data Services DDL editor with **Copy to Clipboard** and **Export (.asddls)** buttons.
  - Top 3 retrieved FAISS candidate view cards with field/association metrics.
  - Live backend health checking with automatic demo fallback if offline.

---

## API Testing & Examples

### Health Check (`GET /health`)
```bash
curl http://localhost:8000/health
```

### Evaluate CDS Requirements (`POST /api/evaluate-cds`)
```bash
curl -X POST http://localhost:8000/api/evaluate-cds \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "SD",
    "business_objective": "Track outstanding sales orders with customer delivery address, total amount, and creation date",
    "entities": ["SalesOrder", "Customer"],
    "fields": ["SalesOrder", "TotalNetAmount", "CustomerName", "CityName", "CreationDate"]
  }'
```

---

## SAP Authorization Requirements
For production SAP integration, the technical user must have the following authorization objects:
- `S_ADT_RES` (ABAP Development Tools Resource Access)
- `S_RFC` (for RFC/ADT function modules like `SADT_REST_RFC_ENDPOINT`)
- Active ICF nodes in transaction `SICF`:
  - `/default_host/sap/bc/adt`
