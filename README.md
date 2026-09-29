# ЛЦТ Задача Бизнес-1 концепт Модель вирального спроса, развитие предложения Sber-500

Smart Gate — early detection of emerging trends: cheap statistical change-point detection
(EWMA / CUSUM) plus a semantic gate powered by **Qwen3** served through **Ollama** as an
inference/runtime layer.

**Documentation is in Russian:**

* [Readme-ru.md](Readme-ru.md) — what it does, quick start, configuration
* [docs/architecture.md](docs/architecture.md) — architecture and the reasoning behind it
* [Report.md](Report.md) — Sprint 01 results and decisions
* [Backlog.md](Backlog.md) — Sprint 02 scope and technical debt

[![CI](https://github.com/leaderstat/lzt-business-1-viral/actions/workflows/ci.yml/badge.svg)](https://github.com/leaderstat/lzt-business-1-viral/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Sprint](https://img.shields.io/badge/sprint-01-orange)

## Smart Gate — Sprint 01

Baseline-контур раннего обнаружения зарождающихся трендов: дешёвая статистическая детекция разладки + семантический гейт на Qwen3, который обслуживается Ollama как inference/runtime-слой.

**Область Sprint 01.** Веса модели не обучаются. Ни LoRA, ни QLoRA, ни SFT, ни DPO, ни GRPO, ни full fine-tuning — согласно RULE 4 брифа спринта: сначала baseline-датасет и evaluation, только потом веса. Ollama здесь — не training framework (RULE 3).

---

## О проекте

Smart Gate — сервис автоматизированного сбора и анализа зарождающихся трендов (слабых сигналов) в научно-технологических отраслях. Разрабатывается в рамках хакатона Газпромбанк.Тех.

**Продуктовая рамка (полная версия):**
`Данные → Слабый сигнал → Подтверждение → Прогноз → Влияние → Банковская возможность → Продукт`

**Целевая архитектура (Sprint 02+):**
- **YandexGPT + GigaChat** — два аналитических контура (AI Consensus)
- **Signal Engine** — собственные правила, скоринг, кросс-проверка
- Фильтрация мейнстрима, карточки сигналов, Confidence Score
- ТОП-15 зарождающихся трендов, дашборд, API

**Что реализовано в Sprint 01 (этот репозиторий):**
Честный baseline-контур раннего обнаружения. Дешёвая статистическая детекция разладки (EWMA / CUSUM) + семантический гейт на Qwen3 через Ollama. Веса не обучаются — сначала baseline и evaluation, потом модели.

**Почему baseline на Qwen3, а не сразу YandexGPT + GigaChat:**
Sprint 01 решает задачу воспроизводимости и измерения. Qwen3 локально через Ollama даёт нулевую стоимость инференса, полный контроль над режимом и воспроизводимость результатов. Это позволяет зафиксировать метрики baseline, которые в Sprint 02 будут сравниваться с YandexGPT + GigaChat.

**Связь с презентацией:**

| Презентация | Sprint 01 (репозиторий) |
|---|---|
| AI Consensus (YandexGPT + GigaChat) | Qwen3 через Ollama — один контур, baseline |
| Signal Engine | Статистические детекторы EWMA / CUSUM + правила гейта |
| Интерпретируемость | Метрики, lead time, gate_stats, fallback_calls |
| Фильтрация мейнстрима | Семантический гейт отсекает hard negatives |
| Работа на опережение | Lead time — продаваемый результат |

---

## 1. Что умеет этот репозиторий

Sprint 01 — это **baseline-контур**: статистика + семантический гейт. Он не заменяет целевую архитектуру (YandexGPT + GigaChat + Signal Engine), а задаёт точку отсчёта для неё. Всё, что ниже — работает уже сейчас.

| Возможность | Команда | Модуль |
|---|---|---|
| Проверить runtime: сервер, список моделей, точные характеристики модели, зафиксированный режим инференса | `smartgate doctor` | `ollama_client.py`, `cli.py` |
| Сгенерировать воспроизводимый размеченный корпус с известными точками разладки | `smartgate dataset` | `dataset.py` |
| Сравнить статистические детекторы между собой (без LLM, миллисекунды) | `smartgate sweep` | `detectors.py`, `pipeline.py` |
| Прогнать полный контур «детектор → Qwen3-гейт → метрики» и сохранить артефакт | `smartgate run` | `pipeline.py`, `llm_gate.py` |
| Отрисовать пример детекции с lead time | `python examples/plot_detection.py` | `examples/` |

**Дополнительно:**

- **REST API, а не shell.** Qwen3 вызывается только через документированные эндпоинты `http://localhost:11434/api/*` (`/api/version`, `/api/tags`, `/api/show`, `/api/chat`). В коде нет ни одного вызова бинарника `ollama` из Python.
- **Ноль рантайм-зависимостей.** Только стандартная библиотека Python ≥ 3.9.
- **Детерминизм.** Один и тот же `--seed` + зафиксированный режим инференса (`think=false`, `temperature=0`, `top_p=1`, `seed=42`) → тот же результат.
- **Честная деградация.** Если сервер Ollama недоступен или ответ не распарсился, гейт переходит на детерминированную эвристику, и это видно в отчёте (`gate_stats.fallback_calls`), а не замалчивается.

---

## 2. Архитектура

Полная схема и обоснование решений: [docs/architecture.md](docs/architecture.md).

**Ключевой принцип разделения:** статистический слой задаёт потолок recall, семантический слой торгует часть этого recall на precision. Гейт умеет только отклонять кандидатов, поэтому recall никогда не поднимется выше того, что дал дешёвый слой. Но и не бесплатно: в измерениях Sprint 01 recall упал с 1.00 до 0.72, а precision вырос с 0.41 до 0.72 (см. [Report.md](Report.md)).

### Примеры детекции

**Зарождающийся тренд** — детектор срабатывает за 20 дней до того, как тема становится очевидно вирусной; это и есть продаваемый lead time:

![emerging trend](examples/emerging_trend.svg)

**Тяжёлый ложноположительный класс** — рост от одной платной кампании, который потом затухает. В момент решения он статистически неотличим от тренда, различить его можно только семантически (нет органического распространения, 80% упоминаний от 5 аккаунтов):

![hard negative](examples/hard_negative.svg)

**Простой ложноположительный класс** — разовый всплеск:

![one-off spike](examples/one_off_spike.svg)

---

## 3. Быстрый старт

### 3.1 Установка

```bash
git clone https://github.com/leaderstat/lzt-business-1-viral.git
cd lzt-business-1-viral
pip install -e ".[dev]"
```

### 3.2 Ollama + Qwen3

```bash
# 1. Установка (официальная инструкция: https://docs.ollama.com/linux)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Запуск сервера
ollama serve &

# 3. Модель. Целевая для продакшена — qwen3:14b,
#    для смоука на ноутбуке/CI достаточно qwen3:0.6b.
ollama pull qwen3:0.6b

# 4. Проверка контура
python -m smartgate doctor --model qwen3:0.6b
```

**Ожидаемый вывод `doctor`:**

```
endpoint: http://localhost:11434/api
server version: 0.34.0
models: qwen3:0.6b
model: qwen3:0.6b
  family: qwen3  params: 751.63M
  quantization: Q4_K_M
  format: gguf
  frozen inference mode: think=False options={'temperature': 0.0, 'top_p': 1.0, 'seed': 42, 'num_ctx': 4096, 'num_predict': 256}
```

### 3.3 Прогон

```bash
# Корпус
python -m smartgate dataset --n 200 --out artifacts/dataset.jsonl

# Только статистика — сравнение детекторов (секунды, LLM не нужна)
python -m smartgate sweep --dataset artifacts/dataset.jsonl --top-k 20

# Полный контур со статистикой + Qwen3
SMARTGATE_TIMEOUT=600 python -m smartgate run \
    --dataset artifacts/dataset.jsonl \
    --detector cusum --top-k 10 \
    --out artifacts/run.json
```

---

## 4. Конфигурация

Переменные окружения (все имеют разумные значения по умолчанию):

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Адрес сервера Ollama |
| `SMARTGATE_MODEL` | `qwen3:0.6b` | Тег модели; целевая продакшен-модель — `qwen3:14b` |
| `SMARTGATE_NUM_CTX` | `4096` | Размер контекста |
| `SMARTGATE_NUM_PREDICT` | `256` | Лимит генерации |
| `SMARTGATE_TIMEOUT` | `120` | Таймаут HTTP-запроса, сек. На CPU ставьте 600 |
| `SMARTGATE_TRACE` | — | `1` включает подробный лог HTTP-обмена с Ollama |

Гиперпараметры детекторов (`--ewma-alpha`, `--ewma-k`, `--cusum-k`, `--cusum-h`, `--warmup`, `--decision-horizon`) — это экспериментальные точки, а не истина (RULE 2). Их подбор экспериментом стоит в бэклоге как S2-05.

---

## 5. Тесты

```bash
pytest -m "not integration"          # быстрые, без Ollama — используются в CI
pytest -m integration                # требуют живого сервера Ollama с моделью
ruff check src tests                 # линт
```

Unit-тесты поднимают настоящий HTTP-сервер, который говорит на подмножестве Ollama REST API. Это проверяет транспортный слой целиком, включая точный JSON, который уходит на сервер (в частности, что режим инференса действительно зафиксирован), а не замоканную функцию.

CI (`.github/workflows/ci.yml`) состоит из двух джоб: линт + unit-тесты на Python 3.9 и 3.12, и отдельная джоба, которая ставит настоящую Ollama, тянет `qwen3:0.6b` и гоняет integration-тесты против живой модели.

---

## 6. Результаты

**Ключевые результаты Sprint 01:**

| Метрика | Значение | Комментарий |
|---|---|---|
| Precision | 0.72 | Вырос с 0.41 после семантического гейта |
| Recall | 0.72 | Упал с 1.00 — плата за precision |
| Lead Time | 20 дней | Продаваемый результат: сигнал раньше тренда |
| Fallback calls | видны в отчёте | Честная деградация, не замалчивается |

**Связь с презентацией:**
Эти цифры — baseline для Sprint 02, где к Qwen3 добавятся YandexGPT и GigaChat (AI Consensus). Сравнение метрик baseline и AI Consensus покажет, даёт ли вторая модель прирост precision без потери recall.

Полный разбор — [Report.md](Report.md). План следующего спринта и технический долг — [Backlog.md](Backlog.md).

---

## 7. Структура репозитория

```
src/smartgate/
  config.py          конфигурация; зафиксированный режим инференса
  ollama_client.py   REST-клиент к Ollama (stdlib, без зависимостей)
  dataset.py         генератор размеченного корпуса с точками разладки
  detectors.py       EWMA / CUSUM / threshold по формулам NIST
  llm_gate.py        семантический гейт на Qwen3 + детерминированный fallback
  metrics.py         P/R/F1, ROC-AUC, PR-AUC, Precision@K, Recall@K, Lead Time, FPR
  pipeline.py        оркестрация и артефакты прогона
  cli.py             doctor / dataset / sweep / run
tests/               unit (фейковый HTTP-сервер Ollama) + integration (живая Qwen3)
examples/            рендер примеров детекции в SVG
docs/architecture.md схема и обоснование архитектурных решений
artifacts/           воспроизводимые результаты прогонов
```

---

## 8. Связь с целевой архитектурой

Sprint 01 реализует **нижний слой** целевой архитектуры:

```
[Sprint 01 — реализовано]
Статистические детекторы (EWMA / CUSUM)
        ↓
Семантический гейт (Qwen3 через Ollama)
        ↓
Метрики: P/R/F1, Lead Time, Precision@K, Recall@K

[Sprint 02+ — в бэклоге]
YandexGPT + GigaChat (AI Consensus)
        ↓
Signal Engine: правила, скоринг, кросс-проверка
        ↓
Фильтрация мейнстрима, Confidence Score
        ↓
Карточка сигнала, ТОП-15, дашборд, API
```

**Что это даёт:**
- Sprint 01 даёт **измеримый baseline**: precision 0.72, recall 0.72, lead time 20 дней
- Sprint 02 добавит **AI Consensus** и сравнит метрики с baseline
- Sprint 03+ — продуктовый слой: карточки, дашборд, API

**Почему это правильно:**
Нельзя строить AI Consensus на непроверенном baseline. Сначала нужно понять, что даёт дешёвая статистика, и только потом наращивать семантику и LLM. Это соответствует RULE 4 брифа спринта.

---

## 9. Источники

Материалы, которые реально использованы в коде (RULE 1 — официальная документация выше блогов):

- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) — эндпоинты `/api/chat`, `/api/tags`, `/api/show`, structured outputs.
- [Ollama Linux install](https://docs.ollama.com/linux), [Ollama Docker](https://docs.ollama.com/docker) — развёртывание runtime.
- [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) — `FROM` / `PARAMETER` / `SYSTEM` / `TEMPLATE` / `ADAPTER`; понадобится в Sprint 02 для подключения LoRA-адаптера.
- [Qwen3 (GitHub)](https://github.com/QwenLM/Qwen3) и [Qwen3 Quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html) — chat template, thinking / non-thinking режимы, параметры генерации.
- [Qwen3-14B (Hugging Face)](https://huggingface.co/Qwen/Qwen3-14B) — точный ID модели и требования к инференсу.
- [Qwen3 Speed Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html) — форма таблицы BF16 / FP8 / AWQ-INT4.
- [NIST e-Handbook 6.3.2.3 CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm) и [6.3.2.4 EWMA](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm) — формулы детекторов.
- [scikit-learn — Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) — определения метрик (значения сверены в тестах).
- [ruptures](https://centre-borelli.github.io/ruptures-docs/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Axolotl Qwen3](https://docs.axolotl.ai/docs/models/qwen3.html), [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), [PEFT](https://huggingface.co/docs/peft/index) — изучено для Sprint 02, в коде Sprint 01 не используется.

```bash
pip install -e ".[dev]"
ollama serve & ollama pull qwen3:0.6b
python -m smartgate doctor
python -m smartgate dataset --n 200 --out artifacts/dataset.jsonl
python -m smartgate sweep   --dataset artifacts/dataset.jsonl
```
