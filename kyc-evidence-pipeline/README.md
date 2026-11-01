# KYC evidence preparation pipeline

A runnable **Python + LangGraph proof of concept** for preparing auditable KYC/KYB evidence from documents. It combines structure-aware parent/child chunks, document-wide extraction, exact source pointers, a coverage ledger, basic reconciliation, and a checkpointed human-review gate. It is **not a KYC decision engine** and does not establish that a document or case is complete.

> **Sensitive-data warning:** This project stores document text, extracted facts, review decisions, model outputs, and graph checkpoints in **unencrypted local SQLite files**. The original source file is not copied into the evidence store: retain it separately in an approved, immutable evidence repository. Use only synthetic data until appropriate security, retention, access-control, and model-provider controls have been established.

## What is implemented

```text
                                 ┌─ Query path: BM25 / optional embeddings
                                 │               → child hits + parent context
Document → parse + classify → structure/chunk → index
                                                  ↓
                 Whole-document extraction → source-pointer checks
                                                  ↓
                      reconcile against current case → human interrupt if needed
                                                  ↓
                         coverage ledger → transactional canonical evidence store
```

- **Two distinct paths:** retrieval searches the source chunks; extraction independently processes every parsed chunk against the applicable document-family contract. Retrieval results are not treated as proof of completeness.
- **Ten declarative document-family profiles** and a separate illustrative policy/coverage contract. Three starter examples cover a financial report, registry extract, and adverse-media article; the remaining profiles are templates, not fully tuned specialist extractors.
- **Structure and provenance:** sections, paragraphs, table rows and headers, linked Markdown footnotes, page numbers and PDF block/table coordinates where available. Quotes carry a document SHA-256, normalized segment SHA-256, exact Unicode-code-point offsets, source locator, and representation type. Table reconstruction is identified as such.
- **Atomic facts, evidence and findings** are separate. Every extracted claim remains `PENDING_HUMAN`; exact quote matching and optional model source-support checks do not establish external truth or legal identity.
- **Coverage and gaps:** section inventory, processed/failed chunks, category status and explicit unresolved issues. Available statuses are `FOUND`, `NOT_FOUND_AFTER_REVIEW`, `NOT_APPLICABLE`, `NOT_REVIEWED`, `EXTRACTION_FAILED`, and `CONFLICTED`. A negative automated review is not proof of absence. Parsing gaps (un-OCR'd images, inferred tables, thin pages) do not void what was extracted elsewhere: each category records `source_fully_parsed` and `parse_gap_pages`, cannot be `NOT_FOUND_AFTER_REVIEW` while gaps remain, and a material category with gaps stays an unresolved requirement. `EXTRACTION_FAILED` is reserved for failed chunks, an unparseable document, or open claim-level issues in that category.
- **Entity layer:** links each fact's subject (and owned or controlled party) to an entity using the document's own definitions, SEC cover-page and registry identity, and identifier-first matching across the case (see below).
- **Ownership graph:** builds who-holds-what across the case from the linked facts, with stakes as low–high ranges, look-through interests through chains, control from voting rights or majority stakes, and checks for holdings above 100% and cycles (see below).
- **Conservative reconciliation:** matching on resolved entities (text only where unresolved), same/overlapping-date ownership conflicts, duplicate-like source claims, and older ownership snapshots followed by disposals. Different dated stakes are not automatically contradictions. No current percentage is inferred from a partial disposal.
- **Human intervention:** a LangGraph interrupt includes the issues, related facts and source quotes. Review decisions are persisted with self-reported reviewer attribution. Deferring an issue does not resolve it; resolving a possible conflict preserves all source claims and does not automatically verify either claim.
- **Checkpoints and retry:** graph stages are checkpointed in SQLite; extraction results are cached per chunk. A retry of identical source bytes and the same configuration reuses successful units and retries failed ones. An optimistic case-revision check prevents persisting conclusions based on a case that changed during processing or review.
- **Offline by default:** BM25 retrieval and intentionally narrow rule extraction require no API key and make no network calls. Optional local neural embeddings run on your machine after downloading model weights; optional OpenAI-compatible extraction, material-fact source-support checks and embeddings require explicit data-sharing opt-in. A model-request budget and optional PDF page scope mark remaining chunks as not reviewed.

## Requirements and quick start

Python 3.11 or newer. Tested with Python 3.13. The example files and reviewer identities are entirely fictional.

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
pytest
./scripts/run_demo.sh
python scripts/evaluate.py
# Re-runs the offline pipeline on two genuine annual-report PDFs (~1–2 minutes):
python scripts/evaluate_public_pdfs.py
# Optional local embedding/hybrid-retrieval evaluation (downloads model weights on first run):
pip install -e '.[local-embeddings]'
python scripts/evaluate_public_pdfs.py --embeddings local --output-dir real_evaluation/local_embeddings_run
```

The demo creates a unique synthetic case under `results/`, deliberately **defers** a stale-ownership issue, exports `results/evidence-package.json`, and writes a retrieval result to `results/query.json`. The output remains `REVIEW_REQUIRED`.

`evaluate.py` checks two tiny, exhaustively labeled synthetic documents: one with facts on separate pages and a table/footnote, and another with contradictory ownership figures. It reports fact-level recall, an unsupported-claim rate against that narrow gold set, review-trigger detection, and table/footnote linkage. These results are a plumbing check, **not a performance estimate for real documents**. The tests additionally generate a 180-page synthetic document with evidence on pages 12, 47, 103 and 161.

See [REAL_PDF_EVALUATION.md](REAL_PDF_EVALUATION.md) for the measured results and the important failed model run.

`evaluate_public_pdfs.py` processes the **actual 150-page Berkshire Hathaway and 132-page Disney 2024 annual reports** in `examples/public_reports/`. The publisher URLs and SHA-256 hashes are recorded in `sources.json`. It reports retrieval and extraction against eight selected, source-checked examples and exports both evidence packages plus a blank `fact_review_template.csv`. This is a tiny, non-exhaustive, non-independently-adjudicated sample; it does not measure whole-document fact recall or claim-level unsupported rate. Complete the CSV with an independent reviewer and run `python scripts/score_reviews.py path/to/fact_review_template.csv` to score labeled claims. Automated benchmark deferrals are explicitly identified as non-human and resolve nothing.

## Manual workflow

Unless overridden, the evidence database is `kyc_data/evidence.sqlite` and the LangGraph checkpoint database is `kyc_data/checkpoints.sqlite`. The files are created with owner-only permissions where supported, but **they are not encrypted**. Keep both databases together: the evidence DB alone cannot resume a paused graph.

### 1. Inspect the available profiles

```bash
kyc-pipeline profiles
```

Auto-classification looks for simple profile signals and fails closed on an ambiguous/no-signal document. Prefer an explicitly selected profile:

```bash
kyc-pipeline ingest examples/synthetic_annual_report.txt \
  --case-id acme-demo --profile financial_annual_report > pause.json
```

The synthetic annual report pauses on `STALE_OWNERSHIP_AFTER_DISPOSAL`: the reported 65% stake is older than the disposal and no later stake is provided. `pause.json` includes a `run_id`, the issue IDs, related facts and their source quotes. A run without a material issue goes directly to `PERSISTED`, but can still be `REVIEW_REQUIRED` because required categories have not been adequately reviewed.

### 2. Review and resume

In a real workflow an authorized analyst would inspect the original documents, the citations and policy before recording a decision. For the **fictional demo only**, this command writes a deferral for each issue:

```bash
python - <<'PY'
import json
p = json.load(open('pause.json'))
assert p['status'] == 'PAUSED_FOR_REVIEW'
decision = {
    'reviewer_id': 'SYNTHETIC_DEMO_ONLY',
    'decisions': [
        {'issue_id': i['issue_id'], 'action': 'defer',
         'reason': 'Synthetic example: obtain a current ownership record.'}
        for i in p['review_request']['issues']
    ]
}
with open('decision.json', 'w') as f:
    json.dump(decision, f, indent=2)
PY
RUN_ID=$(python -c "import json; print(json.load(open('pause.json'))['run_id'])")
kyc-pipeline resume --run-id "$RUN_ID" --decision decision.json
```

Each pending high-severity issue needs exactly one decision. `defer` leaves it open. For a `POSSIBLE_OWNERSHIP_CONFLICT`, the request also permits `resolve` with a nonempty reason and exactly one `selected_fact_ids` value copied from that issue. All conflicting claims remain stored; reviewer identity is **not authenticated** in this POC. Parse failures, unsupported source claims and stale ownership can only be deferred here; obtain better evidence or reprocess an improved source rather than declaring them resolved without support.

Check a paused run without consuming it:

```bash
kyc-pipeline status --run-id "$RUN_ID"
```

### 3. Export, search, and inspect citations

```bash
kyc-pipeline report --case-id acme-demo --output evidence-package.json
kyc-pipeline ownership --case-id acme-demo --format mermaid --output ownership.mmd
kyc-pipeline query --case-id acme-demo --query 'Who controls Birch Ltd?' --top-k 5
# Copy a segment_id from an evidence record:
kyc-pipeline segment --case-id acme-demo --segment-id 'seg_...'
```

The package contains documents, run/configuration hashes, evidence, facts, entities, the ownership graph, findings, coverage, section inventory, issue/review records and an audit-event list. Retrieval also returns individually anchored `parent_context_units`; the combined parent text alone is not a citation. A `segment` provides the exact normalized text against which an evidence record's `[start:end]` offsets can be checked. For a PDF, inspect the original page and coordinates as well; extracted text and reconstructed tables can differ from the visual source.

Add more documents to the same case by calling `ingest` again with the same case ID. Reconciliation compares current extracted claims across documents by **case entity** where the entity layer linked them, and by normalized text otherwise. A match on name alone is not proof that two documents mean the same legal entity; facts persisted before the entity layer existed only match by text.

### 4. Retry failed units

```bash
kyc-pipeline retry --run-id "$RUN_ID"
```

A retry starts a new run, reuses cached successful extraction chunks when their source, policy, profile, model, endpoint and extraction-cache version match, and retries failed chunks. It re-runs source checking and reconciliation; embeddings are rebuilt if enabled. The source file must still be available and have identical bytes (or supply `--file` pointing to the identical file). For OCR or changed source bytes, use a **new** `ingest` instead. Reprocessing the original document after adding a newer ownership record can reassess a previously deferred stale-ownership finding; review remains necessary.

If another document is committed to a case while a graph is running or paused, the optimistic revision check refuses to persist the stale result. Retry the affected run so it reconciles against the new case state.

## Optional local semantic retrieval

No model API or document-upload consent is needed for local embeddings:

```bash
pip install -e '.[local-embeddings]'
kyc-pipeline ingest examples/public_reports/berkshire_2024_annual_report.pdf \
  --case-id brk-local --profile financial_annual_report --embeddings local
# Resume a paused run with a human decision, or defer it as in the section above.
kyc-pipeline query --case-id brk-local --embeddings local \
  --query 'What ownership interest did Berkshire report in Kraft Heinz at the end of 2024?'
```

First use downloads the approximately 67 MB MIT-licensed `BAAI/bge-small-en-v1.5` ONNX model via FastEmbed/Hugging Face. Document and query text are embedded locally, not transmitted to an embedding API. Pre-cache model weights before operating on an air-gapped or sensitive environment; the package manager and model downloader access the network during setup. Index vectors and checkpoints contain sensitive information too. Local neural retrieval is available for the tested English documents; it is not a semantic fact verifier. Vectors are stored separately from LangGraph state to avoid repeatedly checkpointing large vector arrays; an incomplete run's vectors are not exposed by canonical retrieval. The local model uses small inference batches to limit memory use.

## Optional model extraction and embeddings

Only use a model endpoint after confirming that sending the specific documents and personal data to it is permitted by your organization's policies and the provider's contractual/security terms. The opt-in applies to local OpenAI-compatible endpoints too.

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='your-model-supporting-json-object-mode'
# Optional; default is https://api.openai.com/v1
export OPENAI_BASE_URL='https://your-approved-openai-compatible-endpoint/v1'
# Optional, only for dense search:
export OPENAI_EMBEDDING_MODEL='your-embedding-model'

kyc-pipeline ingest examples/synthetic_annual_report.txt \
  --case-id model-demo --profile financial_annual_report \
  --extractor openai --embeddings openai --verify-material \
  --model-batch-size 12 --max-model-requests 20 \
  --allow-remote-sensitive-data
```

Model extraction groups chunks into bounded requests. The default `--max-model-requests 20` is a cost/safety cap; chunks beyond that limit are `NOT_REVIEWED`. To attempt a full large report, explicitly set an appropriate higher budget after estimating cost and capacity. `--extract-pages 45` (or `13,38,45-47`) intentionally limits extraction to selected physical pages; all other parsed chunks remain unreviewed, even when a category has a `FOUND` claim. Use `--model-batch-size`, `--model-batch-max-chars`, and `--model-request-interval` for provider limits. `--model-max-tokens` and `--model-reasoning-effort` are optional, provider-specific parameters. The extraction-request budget does not include optional embeddings or per-fact verification calls.

`--verify-material` makes an additional model call for each material candidate claim to check whether its **cited text** supports it. It does not check external truth, sanctions status or identity. A `SUPPORTED` verdict is still provisional; `UNSUPPORTED`, `UNCERTAIN`, errors, or missing literal fields create review issues. Model output must match a strict JSON schema and cite exact source text. Truncated or schema-invalid output marks the chunk as failed. A claim whose citation is not in the source is discarded and raises `CANDIDATE_CITATION_REJECTED` for that claim's category only; `retry` re-extracts those chunks rather than reusing them. Document content is treated as untrusted data, but prompt-injection resistance is not guaranteed.

For hybrid query-time retrieval, the question is also sent to the configured embedding endpoint:

```bash
kyc-pipeline query --case-id model-demo --query 'Who controls Birch Ltd?' \
  --embeddings openai --allow-remote-sensitive-data
```

Use the same embedding model as indexing. Missing, failed, incompatible or mismatched embeddings fall back to lexical search with a warning. The endpoint must implement OpenAI-style `/chat/completions` with JSON-object mode and/or `/embeddings`; no live provider is needed for the test suite. A retry that uses a model requires `--allow-remote-sensitive-data` again and the same endpoint. Model providers can change behavior under a stable model name; cached responses are not an independent validation.

## Optional Azure Document Intelligence parser

`--parser azure-di` replaces the local parsers with Azure AI Document Intelligence `prebuilt-layout` (REST API `2024-11-30`). Use it for scans, photographed documents, images and Office files, which the local parsers cannot read. The whole file is sent to your Azure resource, so the same `--allow-remote-sensitive-data` opt-in applies.

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT='https://<your-resource>.cognitiveservices.azure.com'
export AZURE_DOCUMENT_INTELLIGENCE_KEY='...'

kyc-pipeline ingest registry_extract_scan.pdf --case-id case-42 \
  --parser azure-di --allow-remote-sensitive-data
```

The adapter (`azure_di.py`) maps Azure's output onto the same segments as the local parsers, so chunking, extraction, citation checks and coverage are unchanged:

- Section nesting follows Azure's `sections` tree; `title` and `sectionHeading` paragraphs become headings. Page headers, footers and page numbers become `page_furniture`; `footnote` paragraphs become footnotes.
- Tables are rebuilt from cells, so a cell's text is not repeated as loose paragraphs. Multi-row column headers are kept as the header context for every row.
- Coordinates are converted from inches to PDF points, so they match the local PDF parser. Locators look like `page:3/di-paragraph:17` and `page:3/di-table:2`.
- Each segment records its lowest OCR word confidence (`metadata.min_word_confidence`).

Citations now match Azure's recognized text, so an OCR misread becomes the "source". These are raised as parse issues on their pages and therefore qualify coverage like any other parse gap:

- `OCR_LOW_CONFIDENCE`: a word below 0.8 confidence.
- `HANDWRITTEN_CONTENT`: handwriting on the page.
- `DI_FIGURE_NOT_INTERPRETED`: the text inside a chart or picture was read, but not the chart or picture itself.
- `DI_TABLE_MERGED_CELLS`: a table body cell spans rows or columns.
- `DI_PAGE_NO_TEXT`: no text found on the page.

Checkbox states appear in the text as Azure's `:selected:` / `:unselected:` markers.

The raw `analyzeResult` is cached in the evidence database, keyed by the file's SHA-256, model, API version and endpoint. A `retry` therefore reuses the same analysis, segments and extraction cache, without calling Azure or needing the remote opt-in again. A retry must use the same endpoint. The key is only ever sent to the configured endpoint; an operation URL on any other host is refused. Errors report the HTTP status or Azure error code, never document text.

Not yet supported: Entra ID (token) authentication, add-on features such as key-value pairs, barcodes or high-resolution OCR, markdown output, and loading a pre-computed `analyzeResult` (for example from a Databricks job). The test suite uses a hand-built `analyzeResult` and a mocked HTTP transport; **it has not been run against a live Azure resource**.

## Entity layer

`entities.py` decides which entity each fact is about, using only cited source text. Each document is resolved first:

- **Identity facts.** Registry fields (`Legal name:`, `Registration number:`, `Jurisdiction:`) and the SEC cover page (legal name, Commission file number, state of incorporation, I.R.S. employer identification number) are extracted as ordinary `legal_identity` facts and become the entity's names, identifiers and jurisdiction. When the cover-page name is not in the text (for example, printed as an image), the facts use the subject `UNRESOLVED_REGISTRANT`.
- **The document's own definitions.** `Berkshire Hathaway Inc. ("Berkshire")`, `... and subsidiaries (the "Company")` and `the terms "we," "us," or "our" refer to ...` link a short name or pronoun to its full name, and each link cites the defining sentence. A definition naming "X and its subsidiaries" points to a **group**, which is a separate entity from X. "X or a subsidiary" also resolves to X's group. A named definition beats a generic one ("the parent company and the subsidiaries"). When a generic and a named definition define the same term, an unnamed registrant takes the named parent's name; this is how Disney's cover page, whose name is an image, is resolved.
- **What is not linked.** A term defined for two different entities raises `DEFINED_TERM_AMBIGUOUS` (medium), and facts using it are left as plain mentions. Former names (`formerly known as`) are not treated as aliases. Names are bounded heuristically: a definition must follow a name ending in a legal form (Inc., Corp., LLC, Ltd, plc …), unless it defines a pronoun.

Each fact gets:

- `entity_links`: the subject's and object's entity, the basis (`DEFINED_TERM`, `GROUP`, `REGISTRANT_COVER`, `LEGAL_NAME`, `MENTION`) and the evidence for the link;
- `entity_resolution`:
  - `RESOLVED`: the subject is a named entity with an identifier, legal name or definition;
  - `RESOLVED_TO_GROUP`: the subject is a group;
  - `TEXT_MATCH_ONLY`: the subject is a plain mention;
  - `UNRESOLVED`: the subject is a pronoun or placeholder with no definition.

A group subject raises `ENTITY_RESOLVED_TO_GROUP`, which is high severity for material categories, because the direct legal holder or actor is still not stated. It replaces `ENTITY_IDENTITY_UNRESOLVED` for subjects the document itself resolves.

Across the case, document entities are merged:

- **On a shared identifier.** SEC file number, EIN and LEI each identify an entity on their own. A registration number must also match on jurisdiction.
- **On the same legal name,** with legal forms normalized (Inc./Incorporated, Ltd/Limited, a leading "The" dropped), but only when no identifier of the same scheme or a known jurisdiction contradicts it. Otherwise the entities are kept apart and `ENTITY_NAME_COLLISION` (medium) is raised.
- **Conflicts inside one entity.** One entity with two values for the same identifier raises `ENTITY_IDENTIFIER_CONFLICT` (high).

The report's `entities` list shows each case entity with its names, identifiers, documents and `match_basis` (`IDENTIFIER`, `LEGAL_NAME`, `SINGLE_DOCUMENT` or `GROUP_OF_PARENT`). `document_entities` shows the per-document records, and every fact carries `case_entity_ids`. Ownership conflicts compare case entities. Figures that agree after rounding (28% vs 28.2%, or "approximately 27%" vs 27.2%) are recorded as `OWNERSHIP_FIGURES_DIFFER_BY_ROUNDING` findings, not conflicts.

Not yet covered:

- people (no dates of birth or nationality matching);
- definitions scoped to one section (Berkshire's Item 1 uses "Company" for the parent, but its auditor's report uses it for the group);
- former names and successor entities;
- lookups against GLEIF, Companies House or SEC EDGAR;
- the model extractor producing cover-page facts (only the local rules do).

## Profiles and policy

`src/kyc_pipeline/configs/document_profiles.yaml` defines family, classification signals and chunk sizes; `src/kyc_pipeline/configs/kyc_policy.yaml` defines illustrative category guidance and materiality per family. They are separate because document processing and bank policy are different concerns. Replace them with approved configurations, for example:

```bash
kyc-pipeline ingest my-report.pdf --case-id case-42 \
  --profile financial_annual_report \
  --profiles-file approved-profiles.yaml --policy-file approved-policy.yaml
```

For a specific family, an explicit policy requirement may be written as `{category: ownership, applicable: false, reason: 'documented rationale'}`. `NOT_APPLICABLE` requires that reason; it is not inferred from missing evidence. The included policy is **illustrative only**, not a regulatory rule or a bank-approved policy.

The strategy names in the ten profiles are configuration labels for the intended document treatment; this POC uses one shared generic structure/table engine with profile-specific window sizes. It does not yet implement distinct clause-, transaction-, or chronology-aware parsers for every profile, and field groupings depend on the parsed record/paragraph boundaries.

The local rules intentionally recognize only a small set of literal patterns: simple registry fields, some ownership percentages and repurchases, contractual voting rights, selected legal-proceeding phrases, simple consolidated revenue rows, and a basic allegation phrase. They do not cover ordinary variations comprehensively. They do **not** perform an exhaustive semantic review. A category with no matched rule is `NOT_REVIEWED`, not `NOT_FOUND_AFTER_REVIEW`. A `FOUND` record only means a candidate was found; it does not imply the category is exhaustive or the fact is verified.

## Ownership graph

`ownership.py` builds the case's ownership and control graph from verified facts and case entities. It is rebuilt from the stored case on every `report` and `ownership` export, so it reflects every document and review decision. During ingestion it runs only to raise issues for review.

- **Edges.** One `EQUITY` edge per holder and held entity, one `VOTING_RIGHTS` edge per reported voting-rights claim, and a `GROUP_PARENT` edge from a parent to its "and subsidiaries" group. Each edge lists its facts, the documents they come from, and any open issues on them.
- **Stakes are ranges.** The latest reported period is current; earlier dated figures are listed as `superseded_fact_ids`, and undated figures stay in the range. Differing current figures give a range (`low`–`high`), not a choice. An open `POSSIBLE_OWNERSHIP_CONFLICT` marks the edge `CONFLICTED`. When a reviewer resolves the conflict, only the selected figure is used. A disposal after the latest stake (`STALE_AFTER_DISPOSAL`), or one that cannot be ordered (`CHRONOLOGY_UNRESOLVED`), lowers the bottom of the range to 0.
- **Groups.** A group's holding counts as its parent's consolidated interest (`ATTRIBUTED_FROM_GROUP`). It is combined with the parent's own figure, never added to it, because the group's stake includes the parent's. If the parent alone holds more than the whole group, the check raises `GROUP_STAKE_BELOW_PARENT_STAKE` (medium).
- **Look-through interests.** Stakes are multiplied along every chain of up to `max_chain_depth` holdings and added across chains. If A holds 60% of B, B holds 50% of C, and A holds 10% of C directly, A's interest in C is 40%. Each interest lists its paths and whether it meets the reporting threshold (`YES`, `POSSIBLE` when only the top of the range does, or `NO`). Flags carry through: `APPROXIMATE`, `UNDATED`, `MIXED_DATES`, `OPEN_ISSUES`, `UNRESOLVED_PARTY` (a party matched by text only) and the edge statuses.
- **Control.** Reported voting rights, or a stake whose whole range is above `control_threshold_percent`, gives `REPORTED` control. If only the top of the range is above it, control is `POSSIBLE`. Reported control links chain (`CONTROL_CHAIN`, `DERIVED`).
- **Checks.** Distinct holders reported to own more than 100% of one entity in the same or an unknown period raise `OWNERSHIP_TOTAL_EXCEEDS_100` (high). Holders that hold one another are not added, because one figure may include the other. A cycle of holdings raises `OWNERSHIP_CYCLE` (medium); look-through figures skip the circular part.

The thresholds (25% reporting, 50% control) and the chain depth are illustrative settings under `ownership_graph` in `kyc_policy.yaml`. `identified_entities_without_reported_owners` lists identified entities, such as registrants, for which no owner was found. That is a gap in the documents, not evidence that the entity has no owners. No natural persons are extracted, so the top of a chain is never a beneficial owner. `kyc-pipeline ownership --format mermaid` draws the edges as a Mermaid flowchart for reading; use the JSON for analysis.

## Input formats and known gaps

- UTF-8 `.txt`/`.md`, static `.html`, `.csv`, and text-based `.pdf` up to 50 MiB. TXT form feeds create supplied page anchors; HTML/CSV do not have invented page numbers. No URL fetching or JavaScript execution. With `--parser azure-di`: PDF, PNG, JPEG, TIFF, BMP, HEIF, DOCX, XLSX, PPTX and HTML (see above).
- PDF text blocks, coordinates and detected tables use PyMuPDF. PDF headings are inferred from font size and weight relative to the document's body text (plus `PART`/`ITEM` numbering); text repeated in the top or bottom margin of three or more pages is recorded as `page_furniture`, not a heading. PDFs with no size or weight variation fall back to capitalised-line headings. Heading inference is heuristic: some headings will be missed and some bold labels outside detected tables may still open sections. Empty pages with no detected text, images, drawings or annotations are recorded as structurally blank; pages containing images are flagged because the local parser does not OCR (use `--parser azure-di`). Borderless tables are inferred from text geometry only when heuristic alignment is detected, and always flagged for visual review. Not every embedded image, complex/merged table, cross-page table, multi-column order, footnote or cross-reference will be detected. HTML scripts, images, embedded sources, merged cells and orphan text are flagged when recognized. Treat parsing issues as coverage failures and visually review original sources.
- Inline Markdown footnote markers are linked to matching footnotes in the **same parsed section**. General footnote resolution, complex table semantics, document amendments, and semantic cross-reference navigation are not complete.
- Entity links come from the document's own definitions and printed identifiers, not from an authoritative register; a merge on legal name alone can join two different entities that share a name and lack identifiers.
- The pipeline does not perform authoritative registry checks, live sanctions/PEP screening, identity verification, beneficial-owner identification (the ownership graph has no natural persons), source-of-funds verification, complete contradiction resolution, or policy-driven customer acceptance decisions. Allegations are not findings of guilt.
- Literal quote and number checks are not semantic validation. Model source-support checks can also be wrong or manipulated. Automated `NOT_FOUND_AFTER_REVIEW` means only that the parsed chunks received a successful model pass; it is not proof of absence.
- SQLite is suitable only for a single-host POC. The evidence store and graph checkpoints are separate databases, so their commits are not one distributed transaction; writes are designed to be idempotent and use a case revision check. There is no authentication, authorization, encryption, secrets manager, immutable/WORM audit log, redaction, retention/deletion workflow, scalable job queue, or production observability. Human reviewer IDs are self-reported. Add these before handling real KYC information.
- All document content in the evidence database, extraction cache, checkpoint database, exports, and review files is sensitive. Control access to all copies, including SQLite WAL/SHM files and backups. External API calls send source text when opted in; configure provider logging and retention appropriately.

## Project layout

```text
src/kyc_pipeline/
  configs/              # processing profiles and separate illustrative policy
  parsing.py            # format adapters and parse-gap tracking
  azure_di.py           # opt-in Azure Document Intelligence layout client and segment mapping
  entities.py           # defined terms, cover-page/registry identity, document and case entity resolution
  chunking.py           # sections, child windows, tables, footnote context
  extraction.py         # local rules and opt-in batched OpenAI-compatible provider
  evidence.py           # exact spans, atomic claims, material source checks
  reconciliation.py     # conservative claims, date and conflict checks
  ownership.py          # ownership/control graph, look-through interests, graph checks
  coverage.py           # coverage ledger and section manifest
  graph.py              # LangGraph orchestration and human interrupt
  store.py              # local SQLite records, cache, audit and case revisions
  retrieval.py          # BM25 + optional embedding search, parent context
  cli.py                # ingest, status, resume, retry, report, ownership, query, segment
examples/               # fictional fixtures and two pinned public-report PDFs
real_evaluation/         # source-checked sample targets and recorded test outputs
scripts/                # demo, public-PDF evaluation and reviewer scoring
tests/                  # unit, graph, provider-mock, real-PDF and CLI tests
```
