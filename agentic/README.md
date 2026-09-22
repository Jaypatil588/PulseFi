# PulseFi heartbeat agent

The worker continuously tails `runtime/live_predictions.csv`. Each row goes
through the deterministic guard in `policy.py`. The real configured Groq model
runs on alert transitions and a bounded periodic cadence. A deterministic merge
keeps the guard as a severity floor, so Groq may escalate but cannot downgrade
it. Groq also supplies trend, forecast, evidence, feedback, suggestion, and
memory.

Run the worker:

```bash
python -m agentic worker
```

If `GROQ_API_KEY` is absent, the worker remains active in visibly labeled
`deterministic-guard-only` mode. It never fabricates an LLM response.

Without ESP32 hardware, start that worker with an honest source label:

```bash
python -m agentic worker --source generated_stream
```

Then write schema-accurate input in another terminal:

```bash
python -m agentic stream
```

The generator replaces only unavailable ESP32/LSTM input. It never creates
agent responses, memories, episodes, or alerts.

Open the persisted-state dashboard:

```bash
streamlit run agentic/dashboard.py
```

SQLite stores:

- `observations`: bounded working memory
- `features`: deterministic guard inputs and evidence
- `episodes`: open, escalated, recovering, and resolved events
- `semantic_memory`: personal baseline/profile and latest agent memory
- `procedural_memory`: the fixed guarded loop procedure
- `model_memory`: the upstream LSTM as parametric memory
- `decisions`: LLM and final states, output, usage, and API identity
- `actions`: durable in-app messages and acknowledgement
- `integration_outbox`: future MCP/export events, not a claimed integration
- `delivery_attempts`: real optional transport attempts and retry outcomes
- `stream_events`: redacted malformed-row and incompatible-schema records
- `traces`: latency, tokens, states, outcome, confidence, and errors

Groq configuration is loaded from `.env` or the process environment:

```dotenv
GROQ_API_KEY=replace-with-your-key
PULSEFI_AGENT_MODEL=openai/gpt-oss-120b
PULSEFI_AGENT_BASE_URL=https://api.groq.com/openai/v1
PULSEFI_LLM_INTERVAL_SECONDS=30
PULSEFI_ALERT_EXPORT_JSONL=
```

Set `PULSEFI_ALERT_EXPORT_JSONL` to a path to activate durable local JSONL
delivery with bounded exponential retries. Leave it blank to retain events as
pending future-integration wiring.

This is a research prototype and not a medical device.
