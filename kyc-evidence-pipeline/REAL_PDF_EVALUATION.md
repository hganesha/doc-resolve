# Real annual-report PDF evaluation

**Run date:** 1 November 2026. **Status:** proof of concept; not a KYC decision engine. The PDFs and model output are public, but the pipeline stores ordinary case data and vectors in unencrypted SQLite: do not use it for confidential KYC data without appropriate controls.

## Sources and scope

| Report | Physical PDF pages | Official source | SHA-256 |
|---|---:|---|---|
| Berkshire Hathaway 2024 annual report | 150 | [Publisher PDF](https://www.berkshirehathaway.com/2024ar/2024ar.pdf) | `6aa0a501680c50a57e0a1a2e841b46800e1fed828a8a7889d83d8eae7953e4bb` |
| The Walt Disney Company fiscal 2024 annual report | 132 | [Publisher PDF](https://thewaltdisneycompany.com/app/uploads/2025/01/2024-Annual-Report.pdf) | `8887cce1a4b609edc9f0b6aac241a1a097fa611cdaf46a84906e5c105eea66fd` |

These are historical reports, not current information about either company. They are packaged under `examples/public_reports/` with retrieval date and source metadata in `sources.json`.

The eight labeled examples in `real_evaluation/gold_sample.json` were selected and checked against the sources **after initial development and debugging**. They are neither held out nor an exhaustive inventory of facts; some features were specifically developed to handle them. No inference about whole-document recall or accuracy on other documents is justified. Page references below are physical PDF pages.

## End-to-end offline rules run

Both reports were parsed, segmented, indexed, passed through local narrow rules, verified against normalized source text, sent through LangGraph's review interrupt, and persisted. The benchmark **automatically deferred** all issues using an explicitly non-human identifier; no person approved or resolved them.

| | Berkshire | Disney |
|---|---:|---:|
| Parsed child chunks | 3,615 | 3,218 |
| Chunks processed by rules | 3,615 | 3,218 |
| Provisional extracted facts | 15 | 17 |
| Evidence spans matching stored normalized text and offsets (including definition sentences cited by entity links) | 121/121 | 24/24 |
| Final case status | REVIEW_REQUIRED | REVIEW_REQUIRED |

An earlier, pre-improvement pass found only one provisional fact across both reports and matched none of the eight selected examples. The latest rule pass matched **7 of 8 selected source claims** in their labeled fields and produced a **partial** match for Disney 2024 revenue: `91,361` million and reporting year are captured, but the legal reporting subject remains `UNRESOLVED_REPORTING_ENTITY`. The exact-match count does not validate every extracted claim and is not overall document recall.

Examples that the final run captures include Berkshire's **approximately 27%** of Kraft Heinz and **28%** of Occidental as of year-end 2024, with the holder preserved as **“Berkshire or a subsidiary”**; 2024 consolidated revenue of **$371,433 million** with separate table/title/header references; BHE's **5.85%** repurchase; ESPN's **generally 80%** ownership; and the **August 2024 preliminary injunction** relating to Venu Sports. Do not resolve a direct owner, materiality, wrongdoing or current legal outcome from these statements alone.

Coverage now distinguishes categories. Berkshire: `legal_identity`, `ownership`, `financial_condition`, `legal_regulatory` and `material_changes` are `FOUND`; `control` and `related_parties` are `NOT_REVIEWED`. Disney: `legal_identity` and `legal_regulatory` are `FOUND`; `ownership` and `financial_condition` are `EXTRACTION_FAILED` because specific claims carry open issues (an `and`-joined owned entity; revenue whose reporting entity is not in the cited text); the remaining categories are `NOT_REVIEWED`. Every record lists the pages with parsing gaps (Berkshire p.139; Disney pp. 3, 38–43 and others with inferred tables or un-OCR'd images) and sets `source_fully_parsed: false`, which keeps all material categories in the package's unresolved requirement gaps. Earlier runs marked **every** category `EXTRACTION_FAILED` whenever any page had a parse gap; that made the ledger uninformative and was changed. A rules pass through every parsed chunk is **not a semantic review of every fact**.

**Entities.** The four extra Berkshire facts and three extra Disney facts are cover-page identity: Commission file number, state of incorporation and employer identification number (and, for Berkshire, the legal name). Disney's cover-page name is an image that was not OCR'd. Its registrant is still identified as The Walt Disney Company, because the report's generic definition ("the terms “Company”, “we” … refer collectively to the parent company and the subsidiaries") and the auditor's named definition ("The Walt Disney Company and its subsidiaries (the “Company”)") define the same term. In Berkshire, "Berkshire", "BERKSHIRE HATHAWAY INC." and "Berkshire Hathaway Inc." resolve to one entity carrying both identifiers. "we", "Berkshire or a subsidiary" and "Berkshire and its subsidiaries" resolve to that entity's group, as does "the Company" in Disney. These 16 group-level claims raise `ENTITY_RESOLVED_TO_GROUP`, because neither report names the direct holder.

Two Berkshire terms are left unlinked as `DEFINED_TERM_AMBIGUOUS`, and both are genuine:
- "Company" means Berkshire Hathaway Inc. alone in Item 1, but the group in the auditor's report.
- The report spells the same business both "Marmon Holdings, Inc." and "Marmon Holding, Inc.".

Matching on entities exposed one comparison that text matching never made: "28% of Occidental" (p.45) against "we owned … 28.2%" (p.104). It is recorded as `OWNERSHIP_FIGURES_DIFFER_BY_ROUNDING`, not as a conflict. Disney's two consolidated-revenue claims remain `UNRESOLVED`: the rule did not capture a reporting entity from the table.

**Ownership graph.** Each export now includes the case ownership graph. Berkshire has three holdings, all attributed from "we" or "Berkshire or a subsidiary" to Berkshire Hathaway Inc.: Occidental 28.0–28.2%, Kraft Heinz 27.2% and American Express 21.6%, all as of 2024-12-31. Kraft Heinz is reported three times (27.2% by Berkshire, 27.2% and "approximately 27%" by the group). These are combined as one interest of 27.2%, not added. Disney has five undated holdings by "the Company", attributed to The Walt Disney Company: ESPN 80% ("generally") and four National Geographic items at 73%. Each is above 50%, so control is recorded, but every one carries `OPEN_ISSUES`. Two of the five owned parties are business descriptions, not entities ("FX and National Geographic", "National Geographic magazine and online business"). Neither report states who owns the registrant, so both registrants are listed under `identified_entities_without_reported_owners`. No graph check fired on either report.

**Profiles and typed schemas.** Both reports now run with sentence boundaries, parent-section context and typed schemas. The output is unchanged: the same 15 Berkshire and 17 Disney facts, the same issues and the same coverage. All 32 facts conform to their predicate schemas, and no attribute conflict was raised. Both reports agree with themselves on legal name, jurisdiction and identifiers. Chunk counts are unchanged (3,615 and 3,218) because PDF text blocks are mostly shorter than the chunk size. A few long blocks now split at sentence ends, which changed the lower-ranked pages returned for one Disney retrieval query; every scored rank is unchanged. The benefit of section context is for the model extractor, which was not re-run.

**Registry and licence records.** The new registry and licence rules add nothing on either report: neither has `Label: value` registry records, and both still give the same facts, issues and coverage. The GLEIF check is off by default and was not used for these runs.

**Source of funds.** The bank-statement, declaration and completion-statement rules add nothing on either report: neither has an account-holder header or a sale record, so both still give the same 15 and 17 facts. The transactions and wealth-and-funds categories are not part of the financial-statements family, so these rules do not run on annual reports.

**Companies House check.** Not run for these reports: both registrants are US companies, so neither would be looked up. Registry checks are off by default.

**Beneficial owners.** Facts, issues and coverage are unchanged on both reports. The shareholder-register and beneficial-owner table rules do not fire on them, and no person is extracted. The new `beneficial_owners` section lists each investee as `NOT_IDENTIFIED`, with the registrant where its stake reaches 25%: Occidental and Kraft Heinz under Berkshire Hathaway Inc., and the five Disney holdings under The Walt Disney Company. American Express lists no chain end because Berkshire's 21.6% is below the threshold. Neither report says who owns the registrant, which is expected for a listed company.

**Identity documents, sanctions and PEP, prior KYC files, adverse media.** Facts, issues, coverage and findings are unchanged on both reports (rules cache version 11). The identity, screening-report and prior-file rules belong to families that annual reports are not classified as. The new outcome-stage sentences and enforcement-notice records run only for the adverse-media, litigation and licensing families, so the reports' `legal_regulatory` facts come from the same rules as before. No screening list was supplied. Run times fell (Berkshire 23.0 s to 15.8 s, Disney 16.5 s to 11.4 s), but these are single runs on a laptop and not a benchmark.

Section structure is inferred from font size and weight, with repeated page headers and footers recorded as `page_furniture` rather than headings. The Berkshire report now yields 105 sections (previously 54), including Parts, Items 1–15 and all 27 notes to the financial statements; previously signatures, table labels and running headers such as `/S/ CHARLOTTE GUYMAN` and `MANUFACTURING:` opened top-level sections. Disney yields 126 sections following PART → ITEM → subsection.

Berkshire p.139 and 26 Disney pages include geometry-inferred, borderless tables. Disney PDF pages 3, 129 and 132 include images that were **not OCR'd**. The PDF parser inferred a few structurally blank pages from the absence of text/images/drawings/annotations; visual review is still needed. Offsets refer to normalized stored text, not raw PDF byte offsets. Reconstructed table separators and row order must be checked against the original pages.

## Retrieval evaluated separately

Eight natural-language questions were evaluated against the same eight source-selected examples. The final lexical run reported:

- **5/8** selected passages found in the top five child chunks.
- **6/8** selected passages found in the top five results after expanding individually anchored parent context.
- **7/8** with parent context when manually labeled alternative passages were also allowed. For example, Berkshire p.142 supplies a more precise 27.2% figure for Kraft Heinz, and p.104 gives 28.2% for Occidental. These are relevant alternatives, not identical claims to the approximate p.45 values. (Before section detection was corrected this was 8/8: the Occidental hit on p.104 sits in Note 4 and the 28.2% passage in Note 5, which previously fell inside one oversized section, so parent context happened to include both.)

This is a tiny, post-development sample. Alternative relevant evidence can also exist outside the labels, and retrieval quality does not establish extraction correctness. Each `parent_context_unit` carries its own source/page anchor; the concatenated parent text is not itself one citation.

<!-- LOCAL_EMBEDDING_RESULTS -->

## Real document-wide model extraction

A separate run sent only the **public Berkshire PDF's extracted text** to the anonymous OpenAI-compatible `openai-fast` endpoint at `text.pollinations.ai`. It used up to 250 chunks/45,000 characters per request, a 32-second minimum interval, a 40-request budget, and an 8,000-token response limit. This is a rate-limited third-party service, not an approved KYC processor or a representative test of other models. Never send confidential material there without explicit authorization and appropriate contractual/security controls.

That run used the earlier JSON-object output mode, a single retry and no batch splitting. The client now defaults to strict JSON-schema output with backoff retries, and halves truncated batches; repeating the run against a server without `json_schema` support needs `--model-output json-object`. It has not been repeated, so the figures below describe the earlier client.

The actual run took approximately **735 seconds and 21 extraction requests**:

| Result | Count |
|---|---:|
| Total parsed chunks attempted | 3,615 |
| Chunks whose model response completed and passed basic candidate processing | 2,876 |
| Failed chunks | 739 |
| — provider HTTP 400 | 494 |
| — output truncated | 237 |
| — at least one candidate citation did not match normalized source | 8 |
| Provisional claims retained | 20 |
| Original retained claims with a literal-support warning | 18 |
| Selected Berkshire target claims matched exactly | 0/5 |

A completed model response is **not** evidence of exhaustive extraction: the model missed all five labeled Berkshire targets under the strict fact-field test. For example, it found the 27% number but named “Berkshire Hathaway Inc.” as the holder where the citation said “Berkshire or a subsidiary,” and it omitted “approximately.” Under the coverage rules in force for that run, every category remained `EXTRACTION_FAILED`; all material issues were deferred, and the case remains `REVIEW_REQUIRED`.

### Citation-support check on the 20 model claims

A **non-independent assistant source check, not human adjudication**, reviewed the cited span(s) of all 20 retained claims. It labeled **18 unsupported by their cited spans** and **2 uncertain**. Among the 18 given a supported/unsupported label, the unsupported fraction is **18/18**; the two uncertain claims are excluded. This is a result for one weak model, one configuration and one document only, not a general model error rate. “Unsupported by the cited span” means the quoted evidence does not establish every claim field; it does not prove the underlying assertion false.

Two examples show why exact text matching is not semantic verification:

1. A quoted **$26.8 billion of IRS payments** was labeled `reported_revenue`.
2. A quoted **$3.9 billion acquisition cost** was attached as the value of an ownership relationship.

Neither example triggered a literal-support warning in the original model run. The validator was subsequently strengthened with narrow predicate, missing-object and ownership-unit checks. A **retrospective revalidation of the cached candidates**, without another model call or a new human review, flagged these and all 20 retained claims for at least one reason. Warnings can be false positives or miss other errors; they do not make this pipeline production-ready.

Artifacts: `real_evaluation/results/berkshire_full_model_evidence.json`, `berkshire_model_assistant_citation_review.csv`, and `berkshire_model_revalidation.json`. The original, public-data-only model evidence database is in `real_evaluation/model_archive/`; the review CSV explicitly identifies the reviewer as an assistant, not a person. A separate blank analyst template is generated for the rules facts; no independent unsupported rate for those facts has been measured.

## Reproduce and inspect

```bash
pip install -e '.[test]'
pytest -q
python scripts/evaluate_public_pdfs.py --output-dir real_evaluation/reproducible_run
# Optional local neural retrieval (model weights are downloaded on first use):
pip install -e '.[local-embeddings]'
python scripts/evaluate_public_pdfs.py --embeddings local --output-dir real_evaluation/local_embeddings_run
# Score the recorded non-human model citation review:
python scripts/score_reviews.py real_evaluation/results/berkshire_model_assistant_citation_review.csv
# Re-run only current heuristic validation on cached model candidates, without a model call:
python scripts/revalidate_model_cache.py \
  --db real_evaluation/model_archive/berkshire_model_evidence.sqlite \
  --case-id public-berkshire-model-full --output /tmp/model-revalidation.json
```

For an independent human assessment of the rules output, have a reviewer inspect the original pages and complete `real_evaluation/reproducible_run/fact_review_template.csv` with `SUPPORTED`, `UNSUPPORTED` or `UNCERTAIN`, an identifier, and notes, then pass the file to `scripts/score_reviews.py`. This does not alter or approve canonical evidence. The sample is not independently adjudicated by default.

## Unresolved safety and engineering limitations

- Source match is not external truth, legal-entity identity verification, semantic entailment, sanctions clearance, absence of adverse information, or a decision to onboard.
- OCR, comprehensive layout/footnote reconstruction, document-family validation beyond the small POC, stronger independently assessed models, robust numerical/temporal reconciliation, and fresh sources are still required.
- All extracted facts remain `PENDING_HUMAN`; self-reported reviewer identity is not authenticated. PDF/report-derived evidence may be outdated.
- SQLite, checkpoints, vectors and archived run data are plaintext; no authentication, encryption, retention policy, tenant isolation or secure deployment is provided.
- The network-enabled model and local model weights are third-party software/services; evaluate their licenses, privacy terms, location, quality and availability before use.
