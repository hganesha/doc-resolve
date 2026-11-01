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
- **Ten declarative document-family profiles** and a separate illustrative policy/coverage contract. Each profile's settings change how it is chunked (where long text may be split, whether field records stay whole) and what surrounding text the model sees. Starter examples cover a financial report, registry extracts, a licence register entry, an adverse-media article, and a bank statement, source-of-funds declaration and completion statement; the remaining profiles are templates, not fully tuned specialist extractors.
- **Typed extraction schemas:** each category lists the predicates it accepts and the fields each one needs (object, value type and unit, date, claim type). Off-schema predicates are rejected, and claims that don't fit are flagged. Single-valued attributes, such as legal name, jurisdiction and identifiers, are checked for conflicts across the case (see below).
- **Structure and provenance:** sections, paragraphs, table rows and headers, linked Markdown footnotes, page numbers and PDF block/table coordinates where available. Quotes carry a document SHA-256, normalized segment SHA-256, exact Unicode-code-point offsets, source locator, and representation type. Table reconstruction is identified as such.
- **Atomic facts, evidence and findings** are separate. Every extracted claim remains `PENDING_HUMAN`; exact quote matching and optional model source-support checks do not establish external truth or legal identity.
- **Coverage and gaps:** section inventory, processed/failed chunks, category status and explicit unresolved issues. Available statuses are `FOUND`, `NOT_FOUND_AFTER_REVIEW`, `NOT_APPLICABLE`, `NOT_REVIEWED`, `EXTRACTION_FAILED`, and `CONFLICTED`. A negative automated review is not proof of absence. Parsing gaps (un-OCR'd images, inferred tables, thin pages) do not void what was extracted elsewhere: each category records `source_fully_parsed` and `parse_gap_pages`, cannot be `NOT_FOUND_AFTER_REVIEW` while gaps remain, and a material category with gaps stays an unresolved requirement. `EXTRACTION_FAILED` is reserved for failed chunks, an unparseable document, or open claim-level issues in that category.
- **Registry and licence records:** registry extracts and licence register entries yield legal name, identifiers, status, legal form, incorporation date, registered address, former names, directors and secretary, and a licence's type, status, number, issuer, dates and restrictions. LEI check digits are verified, and adverse company or licence statuses and passed expiry dates are raised for review. Opt-in GLEIF and Companies House lookups compare each LEI's and UK company number's public record with the documents, including officers and people with significant control (see below).
- **Source of funds:** bank statements yield the account holder, bank, opening and closing balances, and each row's credit, debit and running balance; the printed balances are checked against the arithmetic. Declarations and sale completion statements yield the declared source and amount, the asset sold and the net proceeds, which are looked for among the credits on the same person's statements (see below).
- **Entity layer:** links each fact's subject (and owned or controlled party) to an entity using the document's own definitions, SEC cover-page and registry identity, and identifier-first matching across the case (see below).
- **Ownership graph:** builds who-holds-what across the case from the linked facts, with stakes as low–high ranges, look-through interests through chains, control from voting rights or majority stakes, checks for holdings above 100% and cycles, and a beneficial-owner report per company that sets the owners derived from holdings beside declared owners and register PSCs (see below).
- **Conservative reconciliation:** matching on resolved entities (text only where unresolved), same/overlapping-date ownership conflicts, conflicting single-valued attributes such as two legal names for one entity, duplicate-like source claims, and older ownership snapshots followed by disposals. Different dated stakes are not automatically contradictions. No current percentage is inferred from a partial disposal.
- **Human intervention:** a LangGraph interrupt includes the issues, related facts and source quotes. Review decisions are persisted with self-reported reviewer attribution. Deferring an issue does not resolve it; resolving a possible conflict preserves all source claims and does not automatically verify either claim.
- **Checkpoints and retry:** graph stages are checkpointed in SQLite; extraction results are cached per chunk. A retry of identical source bytes and the same configuration reuses successful units and retries failed ones. An optimistic case-revision check prevents persisting conclusions based on a case that changed during processing or review.
- **Offline by default:** BM25 retrieval and intentionally narrow rule extraction require no API key and make no network calls. Optional local neural embeddings run on your machine after downloading model weights; optional model extraction, material-fact source-support checks and embeddings (OpenAI, Azure OpenAI or an OpenAI-compatible server, with strict structured output and retries) require explicit data-sharing opt-in. A model-request budget and optional PDF page scope mark remaining chunks as not reviewed.

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

Each pending high-severity issue needs exactly one decision. `defer` leaves it open. For a `POSSIBLE_OWNERSHIP_CONFLICT` or `POSSIBLE_ATTRIBUTE_CONFLICT`, the request also permits `resolve` with a nonempty reason and exactly one `selected_fact_ids` value copied from that issue. All conflicting claims remain stored; reviewer identity is **not authenticated** in this POC. Parse failures, unsupported source claims and stale ownership can only be deferred here; obtain better evidence or reprocess an improved source rather than declaring them resolved without support.

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
export OPENAI_MODEL='your-model-supporting-structured-outputs'
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

### How model calls behave

- **Strict output.** Requests use JSON-schema structured output (`response_format` type `json_schema`, `strict: true`), built from the same pydantic models that validate the reply. Each batch's schema also limits `chunk_id` to the chunks sent and `category` to the applicable categories. Length and pattern limits are not sent (strict mode does not accept all of them) but are still enforced when the reply is validated. For a compatible server without `json_schema` support, pass `--model-output json-object`; the schema then goes in the prompt instead.
- **Retries.** Rate limits (429), timeouts, connection errors and 408/409/5xx responses are retried up to `--model-max-retries` times (default 4) with exponential backoff and jitter, honouring `retry-after-ms` and `Retry-After` up to 60 seconds. A 429 for exhausted quota and other 4xx errors fail at once. `--model-timeout` (default 120 seconds) bounds each response.
- **Failures say why.** Errors name the HTTP status and the provider's error code and parameter, for example `Provider returned HTTP 400 (unsupported_parameter: temperature)`, plus the provider's request ID in the extraction log (`provider_request_id`). Response bodies and document text are never included. Truncated output, a content-filter block, a model refusal and schema-invalid output each fail closed with their own `error_code`.
- **Truncated or filtered batches are split.** If a batch is truncated or blocked by a content filter, it is halved and each half retried, down to single chunks, so one long or filtered chunk fails alone. Split requests count against `--max-model-requests`.
- **Usage.** Each extraction unit records its request's ID, attempts and token counts (`model_call`); the run summary totals them in `model_usage`.
- **Reasoning models.** With `--model-reasoning-effort`, `temperature` is not sent. `--model-max-tokens` is sent as `max_completion_tokens` to OpenAI and Azure and as `max_tokens` to other servers.

### Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'
export AZURE_OPENAI_API_KEY='...'          # or AZURE_OPENAI_AD_TOKEN for an Entra ID token
export AZURE_OPENAI_DEPLOYMENT='your-chat-deployment'
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT='your-embedding-deployment'   # only for --embeddings openai
# Optional; default 2024-10-21 (the first GA version with strict structured output)
export AZURE_OPENAI_API_VERSION='2024-10-21'

kyc-pipeline ingest examples/synthetic_annual_report.txt \
  --case-id azure-demo --profile financial_annual_report \
  --extractor openai --model-provider azure-openai \
  --allow-remote-sensitive-data
```

`--model-provider azure-openai` calls `/openai/deployments/{deployment}/chat/completions?api-version=...` with the `api-key` header. `--model` and `--embedding-model` then name deployments and override the environment variables. The `openai` choices of `--extractor` and `--embeddings` mean "the configured model provider". An Entra ID token is used as given and not refreshed, so a long run needs a token that outlives it. Azure's content filters may block passages in adverse-media, litigation or sanctions documents; those chunks are reported as `EXTRACTION_FAILED` with `error_code` `content_filter`, never silently dropped. Ask your Azure administrator about a filter configuration suited to this content.

`--verify-material` makes an additional model call for each material candidate claim to check whether its **cited text** supports it. It does not check external truth, sanctions status or identity. A `SUPPORTED` verdict is still provisional; `UNSUPPORTED`, `UNCERTAIN`, errors, or missing literal fields create review issues. Model output must match a strict JSON schema and cite exact source text. Truncated or schema-invalid output marks the chunk as failed. A claim whose citation is not in the source is discarded and raises `CANDIDATE_CITATION_REJECTED` for that claim's category only; `retry` re-extracts those chunks rather than reusing them. Document content is treated as untrusted data, but prompt-injection resistance is not guaranteed.

For hybrid query-time retrieval, the question is also sent to the configured embedding endpoint:

```bash
kyc-pipeline query --case-id model-demo --query 'Who controls Birch Ltd?' \
  --embeddings openai --allow-remote-sensitive-data
```

Use the same embedding model as indexing. Missing, failed, incompatible or mismatched embeddings fall back to lexical search with a warning. The endpoint must implement OpenAI-style `/chat/completions` with JSON-schema (or, with `--model-output json-object`, JSON-object) output and/or `/embeddings`; no live provider is needed for the test suite. A retry that uses a model requires `--allow-remote-sensitive-data` again and the same provider and endpoint (for Azure, the same API version); it reuses the original run's output mode, timeout and retry settings. Model providers can change behavior under a stable model name; cached responses are not an independent validation.

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

- people beyond their names: a person is a director, a declared beneficial owner or a holder the source calls an individual, matched across documents by name without titles; no dates of birth or nationality, so two people with the same name are merged;
- definitions scoped to one section (Berkshire's Item 1 uses "Company" for the parent, but its auditor's report uses it for the group);
- former names and successor entities;
- entities resolved from register records: GLEIF and Companies House records are compared with the documents, not merged into the entity; no SEC EDGAR lookup;
- the model extractor producing cover-page facts (only the local rules do).

## Profiles and policy

Three files configure the pipeline, kept apart because document processing, bank policy and fact definitions are different concerns:

- `configs/document_profiles.yaml`: family, classification signals, chunk sizes and chunking settings per profile;
- `configs/kyc_policy.yaml`: illustrative category guidance and materiality per family, plus ownership graph settings;
- `configs/extraction_schemas.yaml`: the typed predicates each category accepts, reusable across policies.

Replace them with approved configurations, for example:

```bash
kyc-pipeline ingest my-report.pdf --case-id case-42 \
  --profile financial_annual_report \
  --profiles-file approved-profiles.yaml --policy-file approved-policy.yaml \
  --schemas-file approved-schemas.yaml
```

`kyc-pipeline profiles` lists each profile with its settings, requirements and predicates. Each run records the SHA-256 of all three files, and a change to any of them invalidates the extraction cache.

For a specific family, an explicit policy requirement may be written as `{category: ownership, applicable: false, reason: 'documented rationale'}`. `NOT_APPLICABLE` requires that reason; it is not inferred from missing evidence. The included policy is **illustrative only**, not a regulatory rule or a bank-approved policy.

### Chunking settings

A profile's `strategy` is a label. Three settings change behaviour:

| Setting | Values | Effect |
|---|---|---|
| `boundary` | `whitespace`, `sentence`, `clause` | Where a segment longer than `max_chars` may be split. `sentence` splits at sentence ends and starts each overlap at a sentence. `clause` splits first before numbered clauses (`1.`, `(a)`, `2.1`, `Section 3`). `whitespace` is the old behaviour. |
| `keep_records_whole` | `true`, `false` | Keeps a field record (two or more `Label: value` lines making up at least half its lines) in one chunk, up to four times `max_chars`. A registry's registration number then stays with its legal name instead of falling back to `UNRESOLVED_REGISTRY_SUBJECT`. A larger record is split and flagged `OVERSIZED_FIELD_RECORD_SPLIT`. |
| `context_unit` | `parent_section`, `document_header`, `none` | Text attached to each chunk as `section_context`. `parent_section` gives the section's opening paragraph and the paragraph before the chunk. `document_header` gives the document's first three paragraphs (title, caption, publisher, date). |

The model sees `section_context` beside each chunk so it can resolve "we", "the Company", units and titles. It must still quote claims from the chunk itself. If it replaces a pronoun with a name from the context, it must cite that text in `supporting_refs`, which are checked against the exact source like any other quote. Local rules ignore the context.

These settings only choose split points and context; they are not separate clause-, transaction- or chronology-aware parsers, and they work within the segments the parser produced. Most PDF text blocks are shorter than `max_chars`, so on the two annual reports only a few blocks split differently.

### Typed extraction schemas

`extraction_schemas.yaml` declares predicates per category. Each one sets whether an object is required, the value type (`percent`, `amount`, `count`, `text` or none), whether a value or date is required, and optionally the allowed `date_roles` and `claim_types`, an `evidence_terms` pattern the cited text must match, `single_valued`, and `magnitude` (the value is an unsigned size and the predicate gives the direction, so a debit printed as `-250.00` is a `made_payment` of 250 and still passes the literal check).

- **Typed and open categories.** A category with declared predicates is typed: a candidate using any other predicate is discarded and raises `CANDIDATE_SCHEMA_REJECTED`, which marks that category `EXTRACTION_FAILED`. Categories with none (identity fields, address, sanctions, prior decisions, discrepancies) are open: their facts are kept with `schema_check.status: UNTYPED`.
- **Field checks.** Each fact records `schema_check` (`CONFORMS`, `VIOLATIONS` or `UNTYPED`). A violation, such as a stake given in dollars, revenue without a period, or an accusation recorded as a finding, raises `FACT_SCHEMA_VIOLATION`. A numeric string such as `"27.5"` is converted to a number where the schema expects one. `evidence_terms` replaces the earlier hard-coded wording checks (`PREDICATE_NOT_LITERAL_IN_EVIDENCE`).
- **The model sees the schema.** Each category is sent with its predicates and field rules, replacing the four predicate names that were hard-coded in the prompt.
- **Single-valued attributes.** Legal name, registration number, jurisdiction, legal status, legal form, registered address, LEI, SEC file number and EIN are `single_valued`. A licence's status is `single_valued: per_object`: one status per licence, so a firm can hold one licence that is suspended and another that is authorised. Two facts for the same entity with different values, in overlapping or undated periods, raise `POSSIBLE_ATTRIBUTE_CONFLICT` (high), which a reviewer can resolve by selecting one fact. Values are compared after normalizing case, punctuation and legal forms, so `BERKSHIRE HATHAWAY INC.` and `Berkshire Hathaway Incorporated` agree, and so do `12-3456789` and `123456789`. Facts whose subject is unresolved are not compared. A conflict may be a real rename or re-domiciling, or a sign that two entities were wrongly merged.

The schemas are illustrative. They cover the categories of the four priority families (financial statements, ownership and control, regulatory and legal, adverse media) plus registry identity, licensing, transactions and wealth and funds.

The local rules intentionally recognize only a small set of literal patterns: registry and licence `Label: value` records (see below), bank-statement headers and transaction tables, source-of-funds declarations and sale completion statements, shareholder-register and beneficial-owner tables (see below), some ownership percentages and repurchases, contractual voting rights, selected legal-proceeding phrases, simple consolidated revenue rows, and a basic allegation phrase. They do not cover ordinary variations comprehensively. They do **not** perform an exhaustive semantic review. A category with no matched rule is `NOT_REVIEWED`, not `NOT_FOUND_AFTER_REVIEW`. A `FOUND` record only means a candidate was found; it does not imply the category is exhaustive or the fact is verified.

## Registry and licence records

The rules read `Label: value` lines in registry extracts and licence register entries; [synthetic_registry_extract.txt](examples/synthetic_registry_extract.txt) and [synthetic_licence_register.txt](examples/synthetic_licence_register.txt) show the recognized labels.

- **Subject.** The record's `Company name`, `Legal name`, `Firm name` or `Licensee` line names the subject, and each fact cites that line as support. Without one, the subject is `UNRESOLVED_REGISTRY_SUBJECT` and directors are not extracted.
- **Dates.** A `Date of extract`, `As at` or `Register date` line dates status, address, legal form, officers and licence facts (`date_role: as_of`) and is cited as support. Two dated extracts with different statuses are then a change over time, not a conflict. Incorporation, grant and expiry dates are events.
- **People.** `Directors: Jane Smith (appointed 2019-01-01); Omar Haddad` gives one `is_director_of` fact per person, with the note dropped; `Company secretary: Lena Park` gives `is_officer_of` with the role as its value. People are linked by name only.
- **Licences.** In a record with a licence line, `Status:` is the licence's status and becomes the value of `holds_license`; grant and expiry dates name the licence they belong to. `Restrictions:` or `Conditions:` give one fact per item (separated by `;`).
- **Checks on the facts.** These run on every document, offline:
  - `LEI_CHECK_DIGITS_INVALID` (high): the LEI fails its ISO 17442 check digits. It is then not used to identify or merge entities.
  - `REGISTRY_STATUS_ADVERSE` (medium): a company status such as dissolved, struck off, in liquidation or administration.
  - `LICENSE_STATUS_ADVERSE` (medium): a licence status such as suspended, revoked, cancelled, lapsed or restricted.
  - `LICENSE_EXPIRY_PASSED` (medium): the stated expiry date is before the date of the run.

Only single-line values are read: an address spread over several lines, or a registry exported as a two-column table, is not captured by the rules (the model extractor can still read it).

### GLEIF LEI check

```bash
kyc-pipeline ingest examples/synthetic_registry_extract.txt --case-id registry-demo --registry-check gleif
```

`--registry-check gleif` looks up each LEI that the document names in the public [GLEIF API](https://api.gleif.org) and compares the record with what the case's documents say about that entity. Only the LEI is sent and no key is needed, but a lookup still tells GLEIF which entity is of interest, so it is off by default. Records are stored in the evidence database and reused for 24 hours, so a retry sees the same record. Each check is kept in the run summary and in the case report's `registry_checks`, with the record, when it was fetched and a link to its GLEIF page.

The example's LEI is synthetic (its check digits are valid but GLEIF has no record), so this command pauses with `REGISTRY_RECORD_NOT_FOUND`; a document naming a real LEI shows the comparisons.

| Field | Compared with | Issue if different |
|---|---|---|
| Legal name | GLEIF legal name, then its previous and other names | `REGISTRY_NAME_MISMATCH` (high); `REGISTRY_NAME_IS_OTHER_NAME` (medium) when the documents use a previous name |
| Former names | GLEIF other names | reported as `NOT_IN_REGISTRY`, no issue |
| Jurisdiction | GLEIF jurisdiction code (`Delaware` → `US-DE`, `England and Wales` → `GB`) | `REGISTRY_JURISDICTION_MISMATCH` (high); wording not in the built-in list is `NOT_COMPARED` |
| Registration number | GLEIF `registeredAs`, ignoring punctuation and leading zeros | `REGISTRY_NUMBER_MISMATCH` (medium); `MATCH_IGNORING_PREFIX` when only a letter prefix differs (`C0806592` and `806592`) |
| Incorporation date | GLEIF entity creation date | `REGISTRY_INCORPORATION_DATE_MISMATCH` (medium) |
| Entity and LEI status | `ACTIVE`; `ISSUED` | `REGISTRY_ENTITY_INACTIVE`, `LEI_NOT_CURRENT` (medium) |

An LEI that GLEIF does not know raises `REGISTRY_RECORD_NOT_FOUND` (high); a failed lookup raises `REGISTRY_LOOKUP_FAILED` (medium), so a check never passes silently. GLEIF data is reported by the entity itself and validated by its LEI issuer to varying degrees (the record's `corroboration_level`), so a match is not proof of identity and a mismatch may be a stale record. Other national company registers (US state registers among them) and regulators' licence registers are not looked up.

## Source of funds

```bash
kyc-pipeline ingest examples/synthetic_source_of_funds_declaration.txt --case-id sof
kyc-pipeline ingest examples/synthetic_completion_statement.txt --case-id sof
kyc-pipeline ingest examples/synthetic_bank_statement.txt --case-id sof
```

- **Bank statements.** The header's `Account holder:`, `Bank:`, `Account number:`, `Currency:`, `Statement period:` and `Opening/Closing balance:` lines give `holds_account_at` and two `reported_account_balance` facts, dated with the period's start and end. Each row of the transaction table gives a `received_payment` or `made_payment` (the description is the object) and a running `reported_account_balance`. Separate money-in and money-out columns and a single signed amount column (`-1,000.00` or `(1,850.00)`) are both read. Each row cites its own line, the table header, and the statement's holder, currency and account lines, so every field is literally supported.
- **Balance arithmetic.** Within one statement, each printed balance must equal the previous balance plus the row's credits minus its debits, and the last must equal the closing balance (to half a cent). Rows are taken in document order and reversed when the statement is newest first. A break raises `STATEMENT_BALANCE_BREAK` (high when transactions are material to the profile) with the expected and printed figures; a wrong closing figure raises `STATEMENT_CLOSING_BALANCE_MISMATCH`. A break may be a misread or missing row, or an altered statement, so it goes to review rather than being judged. A statement that adds up gives a `STATEMENT_BALANCES_RECONCILE` finding. That checks the arithmetic, not that the statement is genuine.
- **Declarations and completion statements.** A declaration gives `declared_source_of_funds` and `declared_source_of_wealth` (with the expected amount, as of the declaration date). A completion statement gives `sold_asset` (sale price) and `received_sale_proceeds` (net proceeds, dated on completion).
- **Expected inflows.** Each declared amount and net-proceeds figure is looked for among the credits on the same person's statements in the case, in the same currency. An exact match, or one within 1% (fees, rounding), gives an `EXPECTED_FUNDS_RECEIVED` finding listing the credits. If the person has statements in the case but none matches, `EXPECTED_FUNDS_NOT_FOUND` (medium) is raised. The check runs whenever a document adds to either side, so the documents can arrive in any order.

Limits:

- Amounts are read only with comma thousands separators and a dot decimal (`1,250.00`); `1.250,00` is not read. Dates must be unambiguous: written months (`2 March 2026`) or ISO dates; `02/03/2026` is not read. A row whose date is not read is skipped, so the next printed balance breaks and the statement goes to review rather than passing with a missing row.
- Statement rows are extracted only when the statement names its account holder; without one the transactions category stays `NOT_REVIEWED`.
- People are matched by name, ignoring titles and case (`Ms Jane Smith` is `JANE SMITH`) but not initials (`J Smith` is a different person). A joint account or a company account in a person's name is not handled.
- A matching credit shows that a payment of that size arrived, not that it came from the declared source; payments split across several credits, or arriving in another account, are not matched. Unusual credits (size, round amounts, unexplained third parties) are not flagged.

### Companies House check

```bash
export COMPANIES_HOUSE_API_KEY=...   # free, from the Companies House developer hub
kyc-pipeline ingest examples/synthetic_uk_registry_extract.txt --case-id uk-demo --registry-check companies-house
```

`--registry-check companies-house` looks up each company number that the document gives for a UK company in the [Companies House public data API](https://developer.company-information.service.gov.uk/): the company profile, its officers (including resigned ones) and its people with significant control (PSCs). Only the company number is sent. A company is looked up only when the documents give it a UK jurisdiction (`England and Wales`, `Scotland`, `Northern Ireland`, `United Kingdom`); a number with no stated jurisdiction is listed as `NOT_LOOKED_UP` and never sent, since it may belong to another country's register. Numbers are normalized to Companies House format (`9876543` → `09876543`, `SC 123456` → `SC123456`). `--registry-check` can be repeated to run GLEIF and Companies House together. Records are cached for 24 hours as for GLEIF, and each check is kept in `registry_checks` with a link to the company's public page.

The example's company number is synthetic, so with a real key this command pauses with `REGISTRY_RECORD_NOT_FOUND`; the tests run every comparison against a mocked API.

| Compared | Against | Raised when they differ |
|---|---|---|
| Legal name | Company name, then previous names | `REGISTRY_NAME_MISMATCH` (high); `REGISTRY_NAME_IS_OTHER_NAME` (medium) for a previous name |
| Former names | Previous company names | reported as `NOT_IN_REGISTRY`, no issue |
| Jurisdiction | `england-wales`, `scotland`, `northern-ireland` | `REGISTRY_JURISDICTION_MISMATCH` (high); `United Kingdom` alone is `NOT_COMPARED` |
| Status | Company status | `REGISTRY_ENTITY_INACTIVE` whatever the documents say: high when dissolved, removed or closed, medium otherwise (liquidation, administration, ...) |
| Legal form | Company type (`ltd`, `plc`, `llp`, ...) with common wording for each | `REGISTRY_LEGAL_FORM_MISMATCH` (medium); an unlisted type is `NOT_COMPARED` |
| Incorporation date | Date of creation | `REGISTRY_INCORPORATION_DATE_MISMATCH` (medium) |
| Registered office | Postcode only | `REGISTRY_ADDRESS_MISMATCH` (medium); the office may have moved since the document's date |
| Directors and secretary | Appointments, by name, as of the document's date | `REGISTRY_OFFICER_NOT_FOUND`; `REGISTRY_OFFICER_RESIGNED` when they resigned before that date (both medium). Resigned since is reported as `RESIGNED_SINCE`, no issue |
| Reported stakes (`owns_equity_in` this company) | Current PSCs' shareholding bands (more than 25% up to 50%, and so on) | `REGISTRY_PSC_BAND_MISMATCH` when outside the band; `REGISTRY_PSC_MISSING` for a stake above 25% whose holder is not a current PSC (both medium) |

Register flags (insolvency history, registered office in dispute or undeliverable, overdue accounts or confirmation statement) raise `REGISTRY_RECORD_FLAGGED` (medium). Current officers the documents don't name are listed under `registry_only`, not raised: a document may name only some of them. The record's current PSCs are also added to the ownership graph and its beneficial-owner report (see below).

Officer names are matched on first name and surname, with every name part in the document also in the register's name: `Jane Smith` matches `SMITH, Jane Elizabeth`, but `J Smith` and `Jane Smyth` do not. A missing match may be a spelling difference. A PSC holder is matched by legal name or, for a person, the same rule; a stake whose holder the documents name only as "we" or "the Company" will not match. Companies House records what companies file and verifies little of it, so a match is not proof of identity or ownership. Missing API key, rate limits and failed requests raise `REGISTRY_LOOKUP_FAILED` (medium); no check passes silently.

## Ownership graph

`ownership.py` builds the case's ownership and control graph from verified facts and case entities. It is rebuilt from the stored case on every `report` and `ownership` export, so it reflects every document and review decision. During ingestion it runs only to raise issues for review.

- **Edges.** One `EQUITY` edge per holder and held entity, one `VOTING_RIGHTS` edge per reported voting-rights claim, a `GROUP_PARENT` edge from a parent to its "and subsidiaries" group, and a `REGISTER_PSC` edge for each current person with significant control from a Companies House check. Each edge lists its facts, the documents they come from, and any open issues on them.
- **Stakes are ranges.** The latest reported period is current; earlier dated figures are listed as `superseded_fact_ids`, and undated figures stay in the range. Differing current figures give a range (`low`–`high`), not a choice. An open `POSSIBLE_OWNERSHIP_CONFLICT` marks the edge `CONFLICTED`. When a reviewer resolves the conflict, only the selected figure is used. A disposal after the latest stake (`STALE_AFTER_DISPOSAL`), or one that cannot be ordered (`CHRONOLOGY_UNRESOLVED`), lowers the bottom of the range to 0.
- **Groups.** A group's holding counts as its parent's consolidated interest (`ATTRIBUTED_FROM_GROUP`). It is combined with the parent's own figure, never added to it, because the group's stake includes the parent's. If the parent alone holds more than the whole group, the check raises `GROUP_STAKE_BELOW_PARENT_STAKE` (medium).
- **Look-through interests.** Stakes are multiplied along every chain of up to `max_chain_depth` holdings and added across chains. If A holds 60% of B, B holds 50% of C, and A holds 10% of C directly, A's interest in C is 40%. Each interest lists its paths and whether it meets the reporting threshold (`YES`, `POSSIBLE` when only the top of the range does, or `NO`). Flags carry through: `APPROXIMATE`, `UNDATED`, `MIXED_DATES`, `OPEN_ISSUES`, `UNRESOLVED_PARTY` (a party matched by text only) and the edge statuses.
- **Control.** Voting rights reported without a figure, or a stake or voting percentage whose whole range is above `control_threshold_percent`, gives `REPORTED` control; 40% of the votes is not control. If only the top of the range is above the threshold, control is `POSSIBLE`. A register PSC with the right to appoint the board, significant influence or control, or more than 50% of the votes gives `REGISTER_PSC` control. Reported control links chain (`CONTROL_CHAIN`, `DERIVED`).
- **Register bands.** A PSC's shareholding band (more than 25% up to 50%, and so on) is used as a stake only where the documents report none for that holder and company, flagged `REGISTER_BAND`; where both exist the documents' figure is used and the Companies House check compares the two.
- **Checks.** Distinct holders reported to own more than 100% of one entity in the same or an unknown period raise `OWNERSHIP_TOTAL_EXCEEDS_100` (high). Holders that hold one another are not added, because one figure may include the other. A cycle of holdings raises `OWNERSHIP_CYCLE` (medium); look-through figures skip the circular part.

The thresholds (25% reporting, 50% control) and the chain depth are illustrative settings under `ownership_graph` in `kyc_policy.yaml`. `identified_entities_without_reported_owners` lists identified entities, such as registrants, for which no owner was found. That is a gap in the documents, not evidence that the entity has no owners. `kyc-pipeline ownership --format mermaid` draws the edges as a Mermaid flowchart for reading, with people as rounded nodes and register PSCs as dotted edges; use the JSON for analysis.

### Beneficial owners

```bash
kyc-pipeline ingest examples/synthetic_beneficial_ownership_declaration.txt --case-id ubo-demo
kyc-pipeline ingest examples/synthetic_shareholder_register.txt --case-id ubo-demo
kyc-pipeline ownership --case-id ubo-demo
```

The rules read two kinds of table, in the same section as the company's `Company name:` record (cited with each row, with the record's `Date of declaration:` or `Register date:` as the as-of date):

- a **shareholder register** (`Shareholder`, `Shares held (%)`, optional `Type` and `Voting rights (%)` columns) gives `owns_equity_in` and `has_voting_rights_over`; a `Type` of `Individual` marks the holder as a person;
- a **beneficial-owner table** (a `Beneficial owner` column or a `Nature of control` column) gives `is_beneficial_owner_of`: a *declared* ultimate owner and interest, which is never treated as a direct stake.

People are entities of type `person`: directors, declared beneficial owners and holders typed `Individual`, unless the name is an organization's. They are matched across documents by name without titles (`Ms Jane Smith` is `Jane Smith`, `match_basis: PERSON_NAME`) and to register PSCs by first name and surname.

`beneficial_owners` lists, for each organization with reported owners, the people who own or control it, with three sources side by side rather than merged into one figure:

- `HOLDINGS`: a look-through interest at or above the reporting threshold (`interest`, `meets_threshold`);
- `CONTROL`: reported, derived or possible control, with its basis and path;
- `DECLARED` and `REGISTER_PSC`: what a beneficial-ownership declaration and the Companies House register say.

The company's `status` is `IDENTIFIED` when the holdings, control or register identify someone, `DECLARED_ONLY` when only a declaration names owners, `POSSIBLE` when only the top of a range reaches the threshold, and `NOT_IDENTIFIED` otherwise. A declared owner nothing else supports is flagged `NOT_CORROBORATED`. `chain_ends_at` lists organizations holding at least the threshold whose own owners the case doesn't show: the beneficial owners may be behind them. `direct_holdings_percent` is how much of the company the reported direct stakes account for. Where no owner is identified, `senior_managing_officials` lists the directors the documents name and the register's current directors, the usual fallback.

Where a declaration exists for the company, two issues compare it with the rest:

- `UBO_NOT_DECLARED` (high): the holdings (at or above the threshold), reported or derived control, or a register PSC identifies a person the declaration doesn't name.
- `UBO_DECLARED_PERCENT_DIFFERS` (medium): a declared percentage is outside the look-through interest (allowing 0.5 points) or the register band.

In the demo, Omar Haddad holds 60% of Aster Holdings, which holds 60% of Birchwood Trading: a 36% interest, and control through Aster's majority. Both match the declaration, so nothing is raised. Remove him from the declaration and ingesting Aster's register raises `UBO_NOT_DECLARED`.

Limits: only tables in the company's own section are read, and only plain percentages (a band such as `25-50%` is skipped). People have no dates of birth or nationality yet, so two people with the same name are one person. A trust, nominee or a holder the source does not call an individual is an organization, so its beneficiaries are not reached. A listed company's owners are not expected to be reported, so annual-report investees show `NOT_IDENTIFIED` with the registrant at the top of the chain. A beneficial owner here is a lead built from the documents and register, not a verified identification.

## Input formats and known gaps

- UTF-8 `.txt`/`.md`, static `.html`, `.csv`, and text-based `.pdf` up to 50 MiB. TXT form feeds create supplied page anchors; HTML/CSV do not have invented page numbers. No URL fetching or JavaScript execution. With `--parser azure-di`: PDF, PNG, JPEG, TIFF, BMP, HEIF, DOCX, XLSX, PPTX and HTML (see above).
- PDF text blocks, coordinates and detected tables use PyMuPDF. PDF headings are inferred from font size and weight relative to the document's body text (plus `PART`/`ITEM` numbering); text repeated in the top or bottom margin of three or more pages is recorded as `page_furniture`, not a heading. PDFs with no size or weight variation fall back to capitalised-line headings. Heading inference is heuristic: some headings will be missed and some bold labels outside detected tables may still open sections. Empty pages with no detected text, images, drawings or annotations are recorded as structurally blank; pages containing images are flagged because the local parser does not OCR (use `--parser azure-di`). Borderless tables are inferred from text geometry only when heuristic alignment is detected, and always flagged for visual review. Not every embedded image, complex/merged table, cross-page table, multi-column order, footnote or cross-reference will be detected. HTML scripts, images, embedded sources, merged cells and orphan text are flagged when recognized. Treat parsing issues as coverage failures and visually review original sources.
- Inline Markdown footnote markers are linked to matching footnotes in the **same parsed section**. General footnote resolution, complex table semantics, document amendments, and semantic cross-reference navigation are not complete.
- Entity links come from the document's own definitions and printed identifiers, not from an authoritative register; a merge on legal name alone can join two different entities that share a name and lack identifiers.
- Apart from the opt-in GLEIF and Companies House lookups, the pipeline does not perform authoritative registry or licence-register checks, live sanctions/PEP screening, identity verification, verified beneficial-owner identification (people are matched by name only, and the report is built from what the documents and register state), source-of-funds verification (statement arithmetic and amount matching are not proof of origin), complete contradiction resolution, or policy-driven customer acceptance decisions. Allegations are not findings of guilt.
- Literal quote and number checks are not semantic validation. Model source-support checks can also be wrong or manipulated. Automated `NOT_FOUND_AFTER_REVIEW` means only that the parsed chunks received a successful model pass; it is not proof of absence.
- SQLite is suitable only for a single-host POC. The evidence store and graph checkpoints are separate databases, so their commits are not one distributed transaction; writes are designed to be idempotent and use a case revision check. There is no authentication, authorization, encryption, secrets manager, immutable/WORM audit log, redaction, retention/deletion workflow, scalable job queue, or production observability. Human reviewer IDs are self-reported. Add these before handling real KYC information.
- All document content in the evidence database, extraction cache, checkpoint database, exports, and review files is sensitive. Control access to all copies, including SQLite WAL/SHM files and backups. External API calls send source text when opted in; configure provider logging and retention appropriately.

## Project layout

```text
src/kyc_pipeline/
  configs/              # processing profiles, illustrative policy, typed extraction schemas
  schemas.py            # typed predicate schema validation and per-fact checks
  parsing.py            # format adapters and parse-gap tracking
  azure_di.py           # opt-in Azure Document Intelligence layout client and segment mapping
  entities.py           # defined terms, cover-page/registry identity, document and case entity resolution
  chunking.py           # sections, boundary-aware windows, field records, tables, footnote and section context
  extraction.py         # local rules and opt-in batched OpenAI / Azure OpenAI client (strict output, retries)
  evidence.py           # exact spans, atomic claims, material source checks
  reconciliation.py     # conservative claims, date and conflict checks
  registry.py           # LEI check digits, status and expiry checks, opt-in GLEIF and Companies House lookups
  funds.py              # statement balance arithmetic and declared-amount matching
  ownership.py          # ownership/control graph, look-through interests, register PSCs, beneficial owners, graph checks
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
