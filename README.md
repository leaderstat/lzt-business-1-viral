# lzt-business-1-viral

Smart Gate — early detection of emerging trends: cheap statistical change-point detection
(EWMA / CUSUM) plus a semantic gate powered by **Qwen3** served through **Ollama** as an
inference/runtime layer.

**Documentation is in Russian:**

* [Readme-ru.md](Readme-ru.md) — what it does, quick start, configuration
* [docs/architecture.md](docs/architecture.md) — architecture and the reasoning behind it
* [Report.md](Report.md) — Sprint 01 results and decisions
* [Backlog.md](Backlog.md) — Sprint 02 scope and technical debt

```bash
pip install -e ".[dev]"
ollama serve & ollama pull qwen3:0.6b
python -m smartgate doctor
python -m smartgate dataset --n 200 --out artifacts/dataset.jsonl
python -m smartgate sweep   --dataset artifacts/dataset.jsonl
```
