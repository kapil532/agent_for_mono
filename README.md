# MonoceptGPT

Tiny local LLM with a ChatGPT-style UI, plus a Jira assistant panel.

## Setup

```bash
pip3 install -r requirements.txt
cp teams.json.example teams.json
```

Create a `.env` file:

```
JIRA_DOMAIN=your-domain.atlassian.net
JIRA_EMAIL=you@company.com
JIRA_TOKEN=your_api_token
# Optional
JIRA_PRODUCTION_FILTER_ID=
JIRA_PRODUCTION_DASHBOARD_URL=
```

Get an API token: https://id.atlassian.com/manage-profile/security/api-tokens

## Run

```bash
python3 app.py
```

Open http://localhost:8080

## Files

- `app.py` — Flask server + web UI + inference
- `kapil_llm.py` — tiny transformer model definition
- `jira_client.py` — Jira REST API wrapper
- `teams_store.py` — team members persistence
- `model.pt` — trained weights (not committed; retrain if missing)
- `teams.json` — your team roster (not committed; see `.example`)
