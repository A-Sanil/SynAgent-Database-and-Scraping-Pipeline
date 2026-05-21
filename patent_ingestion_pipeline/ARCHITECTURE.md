# Patent Ingestion Pipeline — Architecture & Capabilities

**Date:** May 2026  
**Project:** SynAgent Patent Ingestion Pipeline  
**Status:** MVP Ready for Local Use + Savio Deployment

---

## Executive Summary

The **Patent Ingestion Pipeline** is a modular system designed to collect patent documents from the web, extract synthesis reactions and chemical data using a local LLM (Qwen on Savio), validate chemical structures, and provide a human-in-the-loop interface for review and correction. It bridges web scraping, chemistry NER, LLM-powered extraction, and an interactive review UI.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Patent Ingestion Pipeline                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐        ┌──────────────┐       ┌────────────────┐
│  Scrapling  │───────▶│  Raw Docs    │──────▶│  PDF Collector │
│  (web)      │        │  (SQLite)    │       │  (pdfplumber)  │
└─────────────┘        └──────────────┘       └────────────────┘
                              │
                              │
                              ▼
                      ┌──────────────────┐
                      │  Parse Queue     │
                      │  (async workers) │
                      └──────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │  Qwen LLM Parser │
                      │  (vLLM endpoint) │
                      │  (on Savio)      │
                      └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐         ┌──────────┐
   │ Chem    │          │ PubChem  │         │ Database │
   │ NER     │──────────▶ Lookup   │────────▶│ (FTS5,   │
   │(OPSIN)  │ (name→    (SMILES   │        │ Semantic)│
   │(RDKit)  │  SMILES)  canonical)│         └──────────┘
   └─────────┘          └──────────┘              │
                                                  │
                                    ┌─────────────▼────────────┐
                                    │  Web UI (FastAPI)        │
                                    │  ┌──────────────────────┐ │
                                    │  │ Patent Review        │ │
                                    │  │ Batch Edit Modal     │ │
                                    │  │ Active Learning Logs │ │
                                    │  └──────────────────────┘ │
                                    │  http://127.0.0.1:8001    │
                                    └────────────────────────────┘
```

---

## Core Components

### 1. **Data Collection (Scrapling + PDF Collector)**
- **Scrapling** (v0.4.8): lightweight async web scraper
- **PDF Collector** (`collector_pdf.py`): downloads PDFs, extracts text via `pdfplumber`, tables via `camelot`
- **CLI:** `python -m patent_pipeline.cli collect <URL>` or `collect_many <urls_file>`
- **Output:** Raw documents stored in SQLite with URL, fetched date, raw text, and metadata

### 2. **Parse Queue & Worker**
- **Parse Queue Table:** tracks raw documents awaiting LLM parsing
- **Worker Loop:** polls queue, processes one item at a time, updates status (pending → running → done/failed)
- **CLI:** `python -m patent_pipeline.cli run_worker --base-url http://... --model qwen`
- **Async-ready:** can scale to multiple workers with Redis or multiprocess

### 3. **LLM Parser (Qwen via vLLM)**
- **Engine:** Qwen 27B running on Savio in Apptainer via vLLM
- **Endpoint:** OpenAI-compatible `/v1/chat/completions` API (HTTP)
- **Output:** structured JSON with patent ID, title, abstract, reactions (SMILES, yields, temps, catalysts, etc.)
- **Multi-pass verification:**
  1. LLM extracts initial structured data
  2. Chemistry NER augments with extracted yields, temperatures, SMILES heuristics
  3. SMILES canonicalization via RDKit + PubChem fallback
  4. Confidence scoring (0.0–1.0) based on RDKit validity + PubChem agreement
  5. Low-confidence reactions (<0.6) flagged for human review

### 4. **Chemistry NER & Validation** (`chem_ner.py`, `pubchem.py`)
- **OPSIN:** chemical name → SMILES conversion (optional, fallback to PubChem)
- **RDKit:** SMILES canonicalization and validity (optional)
- **PubChem REST API:** name/SMILES → canonical SMILES + hazard metadata
- **Heuristic Extraction:**
  - Yields: regex `yield[s]? ... (\d+[.\d]?)%`
  - Temperatures: regex `(\d+)°?C`
  - SMILES-like tokens: validated and normalized
- **Output:** confidence score + canonical SMILES per reaction

### 5. **Database (SQLite + FTS5)** (`database.py`)
**Tables:**
- `patents`: patent_id, title, abstract, source_url, publication_date, inventors, assignee, domain_tags, target_terms, reviewed, raw_text, metadata
- `reactions`: reaction_id, patent_id, reaction_smarts, reactant_smiles_json, **product_smiles**, **confidence**, yield_percent, temperature, solvent, catalyst, time_hours, mechanism_text, notes, metadata
- `raw_documents`: source_url, source_type, title, fetched_at, content_type, raw_text, raw_html, metadata_json
- `parse_queue`: id, raw_document_id, status (pending/running/done/failed), attempts, created_at, updated_at
- `active_learning`: id, reaction_id, patent_id, field, old_value, new_value, user, created_at

**Indexing:**
- FTS5 virtual tables (`patents_fts`, `reactions_fts`) for full-text search
- Optional semantic indexing (sentence-transformers + FAISS) for similarity search
- Simple indexes on patent_id, product_smiles, source_url

### 6. **Web UI (FastAPI + Jinja2 + AJAX)** (`webui.py`, `templates/`, `static/`)
**Endpoints:**
- `GET /` — review queue (list patents, sorted by reviewed status)
- `GET /patent/<patent_id>` — view single patent, reactions with modal edit button
- `POST /patent/<patent_id>/update` — save patent title/abstract
- `POST /api/reaction/update` — AJAX: update reaction SMILES/yield/notes + re-index
- `GET /batch_review` — list low-confidence reactions (<0.6) for batch correction
- `POST /api/active_learning` — log human corrections (field, old_value, new_value, user)
- `GET /search?q=...` — full-text search patents and reactions
- `POST /enqueue_all` — add all raw documents to parse queue

**UI Features:**
- **Dark theme** (Gemini/Claude-like design, violet + cyan accents)
- **Modal edit UI:** click "Edit" on a reaction to open a modal, change SMILES/notes, save via AJAX
- **Batch review:** dedicated page for low-confidence items with modal correction form
- **Active learning logging:** every correction (SMILES fix, notes edit, etc.) is logged with user + timestamp
- **Search integration:** search by keyword across patent text and reaction SMILES
- **No authentication:** open for local use (env var `PATENT_UI_DISABLE_AUTH=1` to force, or auto-allows loopback)

**Styling:**
- Responsive grid layout, glassmorphic cards
- CSS at `/static/styles.css`, JS at `/static/app.js`
- Modal backdrop for edit dialogs

---

## Key Features & Capabilities

### ✅ **Implemented**

1. **Collection**
   - Scrapling-based web scraping of patent URLs
   - PDF download + text extraction (pdfplumber) + table extraction (camelot)
   - Metadata tracking (URL, fetch date, source type)

2. **Structured Extraction**
   - LLM-powered JSON extraction (reactions, SMILES, conditions)
   - Multi-pass validation pipeline (LLM → NER → verify → score)
   - Confidence scoring (0.0–1.0 per reaction)

3. **Chemistry Validation**
   - SMILES canonicalization (RDKit if available, fallback to PubChem)
   - RDKit SMILES validity check
   - PubChem name/SMILES lookup for cross-checking
   - Chemical entity extraction (yields, temperatures, solvents, catalysts)

4. **Database & Indexing**
   - SQLite with FTS5 for keyword search
   - Parse queue for async worker orchestration
   - Confidence scores stored per reaction
   - Active learning correction logs

5. **Human-in-the-Loop Review**
   - Web UI for patent/reaction review
   - Modal editing with AJAX save
   - Batch review page for low-confidence items
   - Correction logging for future retraining

6. **Savio Integration**
   - Qwen LLM running in Apptainer via vLLM on Savio GPU node
   - OpenAI-compatible API endpoint (HTTP)
   - Slurm job submission (job 34257910 built successfully)
   - SSH tunnel support for remote API calls

### 🔄 **Partially Implemented / Future Work**

1. **Multi-worker scaling:** Currently one worker at a time; can extend with multiprocessing or Redis queue
2. **Advanced PubChem enrichment:** hazard codes, CAS lookup, supplier data (scaffolding in place)
3. **Semantic search:** FAISS + sentence-transformers (hooks present, not fully wired)
4. **Automated retraining:** active learning logs stored; retraining pipeline not yet implemented
5. **RDKit/OPSIN on Windows:** optional dependencies; graceful fallback to PubChem
6. **Advanced error handling:** currently basic try/except; could add detailed validation reports per field

---

## Usage Guide

### 1. **Setup**

```bash
cd patent_ingestion_pipeline

# Install dependencies
python -m pip install fastapi uvicorn jinja2 httpx requests scrapling

# Create .env for local model or Savio tunnel
echo "PATENT_LLM_BASE_URL=http://127.0.0.1:8000" > .env
echo "PATENT_LLM_MODEL=qwen" >> .env
# If Savio tunnel: PATENT_LLM_BASE_URL=http://<tunnel-host>:8000

# Initialize database
python -m patent_pipeline.cli init_db --db-path data/patent_pipeline.db
```

### 2. **Collect Patents**

```bash
# Single URL
python -m patent_pipeline.cli collect "https://example.com/patent/xyz"

# Batch from file (one URL per line)
echo "https://example.com/patent/1" > urls.txt
echo "https://example.com/patent/2" >> urls.txt
python -m patent_pipeline.cli collect_many urls.txt

# Add to parse queue
python -m patent_pipeline.cli enqueue_all
```

### 3. **Parse (Run Worker)**

```bash
# Terminal 1: Start web UI
$env:PYTHONPATH = "src"
python -m uvicorn patent_pipeline.webui:app --host 127.0.0.1 --port 8001 --reload

# Terminal 2: Run worker (polls queue, calls Qwen)
python -m patent_pipeline.cli run_worker \
  --base-url http://127.0.0.1:8000 \
  --model qwen \
  --interval 2.0
```

### 4. **Review & Edit**

- Open `http://127.0.0.1:8001/` in browser
- Click patent title to view reactions
- Click "Edit" on a reaction to open modal
- Change SMILES/notes, click "Save"
- Corrections are logged to `active_learning` table

### 5. **Batch Review Low-Confidence**

```
http://127.0.0.1:8001/batch_review
```
- Lists reactions with confidence < 0.6
- Use modal to bulk-correct, log corrections

### 6. **Search**

```
http://127.0.0.1:8001/search?q=yield
```
- Full-text search across patents and reactions

---

## Deployment on Savio

### **Step 1: Ensure vLLM is Running**

```bash
# SSH to Savio login node, then:
sbatch ~/run_qwen.slurm
# Monitor: tail -f slurm-<job_id>.out
```

### **Step 2: Open SSH Tunnel**

```bash
# Local machine:
ssh -L 8000:compute_node:8000 <savio_username>@savio.lbl.gov
```

### **Step 3: Update .env**

```bash
PATENT_LLM_BASE_URL=http://127.0.0.1:8000
PATENT_LLM_MODEL=qwen
PATENT_LLM_API_KEY=  # leave empty if no auth
```

### **Step 4: Run Worker on Savio**

```bash
# Option A: Run locally, call remote vLLM
cd patent_ingestion_pipeline
python -m patent_pipeline.cli run_worker --base-url http://127.0.0.1:8000 --model qwen

# Option B: Run on Savio login node (direct, no tunnel)
python -m patent_pipeline.cli run_worker --base-url http://<compute_node>:8000 --model qwen
```

---

## Current Status

### ✅ **Working**
- Web UI loads without auth errors (fixed 401 by removing HTTPBasic dependency)
- Batch review page and modal editing are functional
- CLI commands compile and import correctly
- Database initialization and FTS indexing work
- Scrapling 0.4.8 is installed and ready
- Chemistry NER (yields, temperatures, SMILES extraction) functional
- vLLM Slurm job created and Apptainer image building (job 34257910)

### ⚠️ **Partial**
- RDKit not installed on Windows (optional; falls back to PubChem)
- OPSIN not installed (optional; falls back to PubChem)
- PubChem API calls require network (local Savio nodes may be isolated)
- Qwen LLM endpoint not yet confirmed reachable (awaiting vLLM job completion)

### 🔴 **Not Yet Tested**
- End-to-end parse from raw document to stored patent (need vLLM endpoint)
- Multi-worker scaling
- Semantic search indexing

---

## Testing Checklist

- [x] UI root endpoint returns 200
- [x] Batch review endpoint returns 200
- [x] Browser loads review queue
- [x] Scrapling imports and works
- [x] Database initialization
- [x] Chemistry NER extraction (yields)
- [ ] Full parse job (need vLLM)
- [ ] Active learning logging end-to-end
- [ ] Semantic search
- [ ] Multi-worker orchestration

---

## Files & Structure

```
patent_ingestion_pipeline/
├── src/patent_pipeline/
│   ├── __init__.py
│   ├── __main__.py
│   ├── models.py              # RawDocument, PatentRecord, ReactionRecord
│   ├── database.py            # SQLite + FTS5 + queue
│   ├── pipeline.py            # IngestionPipeline orchestrator
│   ├── llm_parser.py          # QwenLLMParser + multi-pass validation
│   ├── chem_ner.py            # Chemistry extraction + validation
│   ├── pubchem.py             # PubChem REST API helpers
│   ├── collector_pdf.py       # PDF download + text/table extraction
│   ├── cli.py                 # Typer CLI commands
│   ├── webui.py               # FastAPI web server
│   ├── static/
│   │   ├── styles.css         # Dark theme + modal UI
│   │   └── app.js             # AJAX modal handling
│   ├── templates/
│   │   ├── index.html         # Review queue
│   │   ├── patent.html        # Patent detail + reactions
│   │   ├── batch_review.html  # Low-confidence batch editor
│   │   └── search.html        # Search results
│   └── agents/
│       └── (future: orchestration agents)
├── data/
│   └── patent_pipeline.db     # SQLite database
├── pyproject.toml
├── .env.example
├── .env                       # Local (not in git)
└── run_qwen.slurm            # Savio vLLM job submission
```

---

## Next Steps (Priority Order)

1. **Confirm vLLM endpoint is reachable** (monitor Savio job 34257910)
2. **Run end-to-end test:** collect 1 URL → enqueue → parse → review in UI
3. **Install optional chem deps locally** (RDKit, OPSIN) for offline validation
4. **Add test suite** (pytest with fixtures for DB, mocked vLLM)
5. **Scale worker orchestration** (Redis queue or multiprocessing)
6. **Implement active learning retraining** (log corrections → retrain → boost confidence)
7. **Add semantic search** (build FAISS index on demand)
8. **Export corrections for upstream SynAgent** (active learning logs → CSV/JSON)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SQLite + FTS5** | Lightweight, no external DB service, FTS5 for keyword search, portable |
| **Async queue + single worker** | Simplifies local dev, easy to scale; can swap queue for Redis later |
| **PubChem fallback** | No vendor lock, public API, free |
| **Multi-pass LLM** | LLM → NER → verify → score catches hallucinations + boosts confidence |
| **Active learning logs** | Enables future retraining + feedback loops |
| **Modal UI** | Reduces context switching, AJAX keeps workflow smooth |
| **FastAPI + Jinja2** | Fast, lightweight, no ORM overhead |
| **vLLM + Qwen on Savio** | Free GPU, large model capacity, local control |

---

## Questions & Support

- **"Where do I set PATENT_LLM_BASE_URL?"** → Create `.env` in `patent_ingestion_pipeline/` directory
- **"How do I access the UI remotely?"** → Use SSH tunnel: `ssh -L 8001:127.0.0.1:8001 ...` or expose FastAPI to 0.0.0.0 (not recommended for untrusted networks)
- **"Can I run the worker on Savio instead of locally?"** → Yes, but the database needs to be shared (NFS) or you need to sync results back
- **"What if RDKit/OPSIN aren't installed?"** → Pipeline falls back to heuristics and PubChem; results are less confident but still valid
- **"How do I export the active learning logs?"** → Query the `active_learning` table; export as CSV/JSON for retraining

---

**End of Document**
