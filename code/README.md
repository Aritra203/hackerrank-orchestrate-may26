# Support Triage Agent

Deterministic terminal-based support triage agent for the HackerRank Orchestrate challenge.

## Setup

This project uses only the Python standard library.

1. Open a terminal in the repository root.
2. If you want an isolated environment, create and activate a virtual environment.
3. No packages need to be installed.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, use:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Run

From the repository root:

```bash
python code/main.py
```

To test the sample input explicitly:

```bash
python code/main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/output.csv
```

The agent always reads from the local corpus in `data/` and writes predictions to `support_tickets/output.csv`.

## Approach Overview

The solution is split into small, deterministic modules:

- `code/retriever.py` loads every markdown article from `data/`, strips front matter and markup, chunks the text, and ranks the best article for each ticket using local term-frequency / inverse-document-frequency style scoring.
- `code/classifier.py` infers the company, request type, product area hint, and escalation risk with explicit keyword rules.
- `code/decision.py` turns the classifier output plus retrieved evidence into the final CSV fields and applies conservative escalation logic for billing, fraud, and account-access cases.
- `code/agent.py` orchestrates the pipeline for each CSV row and writes the final output file.
- `code/main.py` provides the terminal entry point.

## Safety and Grounding

- Answers are grounded only in the shipped support corpus.
- The agent does not call external APIs or browse the web.
- Ambiguous, risky, or unsupported tickets are escalated instead of guessed.

## Output

The generated CSV contains these columns:

- `status`
- `product_area`
- `response`
- `justification`
- `request_type`
