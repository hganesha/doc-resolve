#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
# A unique case avoids mixing separate demo runs. These records are synthetic.
CASE_ID="synthetic-$(date +%s)-$$"
DB="results/evidence.sqlite"
CHECKPOINT="results/checkpoints.sqlite"
python -m kyc_pipeline ingest examples/synthetic_annual_report.txt --case-id "$CASE_ID" --profile financial_annual_report --db "$DB" --checkpoint-db "$CHECKPOINT" > results/pause.json
python - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('results/pause.json').read_text())
if p['status'] != 'PAUSED_FOR_REVIEW':
    raise SystemExit('Expected the synthetic stale-ownership review gate')
d={'reviewer_id':'SYNTHETIC_DEMO_ONLY','decisions':[{'issue_id':i['issue_id'],'action':'defer','reason':'Synthetic demonstration: obtain a current ownership record.'} for i in p['review_request']['issues']]}
Path('results/demo-decision.json').write_text(json.dumps(d,indent=2))
print('Paused on:', ', '.join(i['code'] for i in p['review_request']['issues']))
PY
RUN_ID="$(python -c "import json; print(json.load(open('results/pause.json'))['run_id'])")"
# This example deliberately defers rather than pretending to resolve the issue.
python -m kyc_pipeline resume --run-id "$RUN_ID" --decision results/demo-decision.json --db "$DB" --checkpoint-db "$CHECKPOINT" > results/resume.json
python -m kyc_pipeline report --case-id "$CASE_ID" --db "$DB" --output results/evidence-package.json
python -m kyc_pipeline query --case-id "$CASE_ID" --query 'Who controls Birch Ltd?' --db "$DB" > results/query.json
printf 'Case: %s\nEvidence package: results/evidence-package.json\nRetrieval result: results/query.json\n' "$CASE_ID"
