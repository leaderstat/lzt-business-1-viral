"""Smart Gate — evidence-aware trend detection package.

Layers (see docs/architecture.md):

* ``dataset``   — reproducible synthetic signal corpus with labelled change points;
* ``detectors`` — cheap statistical baselines (EWMA, CUSUM, static threshold);
* ``llm_gate``  — semantic gate on top of the statistical trigger, served by Qwen3 via Ollama;
* ``metrics``   — evaluation (P/R/F1, ROC-AUC, PR-AUC, Precision@K, Recall@K, Lead Time, FPR);
* ``pipeline``  — orchestration producing reproducible artifacts.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
