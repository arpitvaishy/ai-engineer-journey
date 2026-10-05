# ai-engineer-journey

My path to an AI Engineer role — built project by project, committed in public.
This repo is both my portfolio and my lab notebook.

## Progress

- [x] **Phase 0** — setup, repo, first LLM call (`first_call.py`)
- [ ] **Phase 1** — LLM foundations, prompting & structured output
- [ ] **Phase 2** — RAG foundations
- [ ] **Phase 3** — Advanced RAG + evaluation
- [ ] **Phase 4** — Agents & tool use
- [ ] **Phase 5** — Deployment
- [ ] **Phase 6** — Interview prep & portfolio

---

## Phase 1 — Project 1: multi-provider streaming chatbot

**Goal:** build a small abstraction so the app can talk to any LLM provider
through one interface, stream responses, and survive transient errors.

### Files

| File | What it is |
|------|-----------|
| `llm.py` | The provider abstraction — `GroqProvider` (done) + `OllamaProvider` (your TODO) |
| `chat.py` | The CLI chatbot. Uses the abstraction + streaming + retry. Never imports a provider directly. |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for your secrets |

### Setup

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

Make sure `.env` exists with your `GROQ_API_KEY` (see Phase 0).

### Your assignment

1. **Finish `OllamaProvider.stream` in `llm.py`.** The hint is in the file.
   Install Ollama and pull a model first:
   ```bash
   ollama pull llama3.2
   ```
2. **Run both providers and compare:**
   ```bash
   python chat.py groq
   python chat.py ollama
   ```
   Same interface, same behaviour — that is the point of the abstraction.
3. **Stretch (this is the resume-worthy bit):** add automatic **failover** —
   if Groq raises (e.g. rate limit), fall back to Ollama automatically. Think
   about where that logic belongs: inside the provider, or above it?

### Commit when done

```bash
git add .
git commit -m "Phase 1: multi-provider chatbot with streaming and retry"
git push
```

---

## Concepts this week

- Tokens, context window, temperature — what they actually control
- Streaming vs blocking calls, and why perceived latency matters
- Retries with exponential backoff (the `tenacity` decorator in `chat.py`)
- Why an abstraction layer beats `if provider == "groq": ...` scattered everywhere

## Optional deeper reading

From *AI Engineering from Scratch* (`github.com/rohitg00/ai-engineering-from-scratch`):

- `phases/11-llm-engineering/01-prompt-engineering`
- `phases/11-llm-engineering/02-few-shot-cot`

## Notes to future me

_(Add three lines after each session: what I built, what broke, what I'd do
differently. These become interview answers.)_
