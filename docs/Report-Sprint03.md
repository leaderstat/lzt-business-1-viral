# Report — Sprint 03 (external-context foundation)

## Result

The real-world corpus builder now enriches every eligible Wikipedia topic with an
independent article description from the Wikimedia Core REST API. The response is cached
on disk and processed one topic at a time. The gate prompt explicitly separates semantic
background from numeric pageview evidence and tells Qwen that a description is not proof
of growth.

This fixes the implementation defect identified in Sprint 02: `context` previously held
only a title and statistics derived from the same pageview series. It did not test the
semantic-gate hypothesis.

## Reproduction and regression coverage

The regression tests cover:

1. the exact external-context endpoint and URL encoding;
2. disk caching and graceful fallback when a description is unavailable;
3. one external-context request per accepted corpus sample;
4. context provenance stored with each sample;
5. explicit semantic/numeric evidence sections in the shared ablation prompt.

Run locally:

```bash
ruff check src tests experiments examples
pytest -m unit --timeout=120
pytest -m real_world --timeout=120
```

## Resource policy

Corpus building remains sequential. Each pageview series and description is handled for
one topic, written through the existing corpus artifact flow, and released before the next
topic. HTTP responses are cached on disk. No model, multiprocessing pool, or full response
collection is added.

## Scientific limit

This change supplies real semantic evidence, but it does **not** claim that Qwen improves
quality. The required dev/test ablation must be rerun on a rebuilt corpus with live Ollama.
The Core API description is fetched at corpus-build time; it is not a historical snapshot,
so `meta.context_available_at` records that limitation explicitly. S3-01 stays partially
open until the ablation is measured, and S3-02/S3-03 remain open.
