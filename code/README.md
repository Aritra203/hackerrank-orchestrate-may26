# Support Triage Agent

Deterministic terminal-based support triage agent for the HackerRank Orchestrate challenge.

## What it does

- Reads tickets from `support_tickets/support_tickets.csv`
- Classifies the request type and risk level
- Retrieves relevant evidence from the local `data/` corpus only
- Decides whether to reply or escalate
- Writes results to `support_tickets/output.csv`

## Run

From the repository root:

```bash
python code/main.py
```

Optional arguments:

```bash
python code/main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/output.csv
```

## Design notes

- Uses only the shipped local corpus.
- No web requests or external APIs are used for support answers.
- Retrieval and routing are deterministic and based on local text scoring plus safety heuristics.
