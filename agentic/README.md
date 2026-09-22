# PulseFi heartbeat agent

This worker continuously tails `runtime/live_predictions.csv`. For every new
LSTM inference row, it sends a bounded context to an OpenAI-compatible model
and stores the validated response and memories in SQLite.

It does not run without credentials:

```bash
export OPENAI_API_KEY="your-new-key"
export PULSEFI_AGENT_MODEL="gpt-4o-mini"
python -m agentic worker
```

Optional:

```bash
export PULSEFI_AGENT_BASE_URL="https://api.openai.com/v1"
python -m agentic worker \
  --csv runtime/live_predictions.csv \
  --db runtime/pulsefi_agent.db
```

The offline harness does not call an external model:

```bash
python -m agentic harness
```

Database memories:

- `observations`: bounded context source (working memory)
- `episodes`: periods the model labels `watch` or `urgent` (episodic memory)
- `semantic_memory`: the latest model-generated memory summary
- `procedural_memory`: the fixed agent-loop procedure
- `model_memory`: metadata describing the upstream LSTM as parametric memory
- `decisions`: predictions, feedback, suggestions, confidence, and model name

This is a research prototype and not a medical device.
