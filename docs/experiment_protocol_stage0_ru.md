# Протокол эксперимента

## 1. Исследовательский вопрос

Как prompt engineering, retrieval-augmented generation и QLoRA fine-tuning по отдельности и совместно влияют на качество, безопасность и groundedness медицинского question answering на Qwen2.5-3B-Instruct?

## 2. Экспериментальная схема

| ID | Модель | Prompt | RAG |
|---|---|---|---|
| A | Base Qwen | Исходный | Нет |
| B | Base Qwen | Улучшенный | Нет |
| C | Base Qwen | Улучшенный | Да |
| D | QLoRA Qwen | Улучшенный | Нет |
| E | QLoRA Qwen | Улучшенный | Да |

Интерпретация сравнений:

- **A → B** — эффект prompt engineering;
- **B → C** — эффект RAG на base-модели;
- **B → D** — эффект QLoRA без RAG;
- **C → E** — эффект QLoRA при фиксированном RAG;
- **D → E** — эффект RAG на QLoRA-модели.

Сравнение **A → E** показывает разницу между исходной и полной системой, но не позволяет определить вклад отдельных компонентов.

## 3. Данные и защита от leakage

- **Train** — supervised fine-tuning.
- **Development / validation** — prompt selection, retrieval design, chunking, top-k, thresholds, reranker, query rewriting, QLoRA hyperparameters и generation settings.
- **Debug set** — небольшой фиксированный subset development data для быстрых sanity checks и отладки pipeline.
- **Frozen test** — только финальная оценка после завершения разработки.

Использование final test для выбора prompt, thresholds, top-k, chunk size, reranker, LoRA-параметров или других решений считается test leakage.

Перед финальным split должны быть проверены exact duplicates и near-duplicate contamination.

## 4. Контролируемые условия

В сопоставимых end-to-end экспериментах сохраняются одинаковыми:

- backbone Qwen2.5-3B-Instruct;
- quantization setup;
- test questions;
- generation settings;
- максимальный generation budget;
- критерии и процедура оценки.

При ablation-сравнении изменяется только исследуемый компонент.

## 5. Retrieval evaluation

Retrieval оценивается на отдельном gold benchmark:

```text
query
↓
relevant article / chunk
```

Основные метрики:

- Recall@1;
- Recall@3;
- Recall@5;
- MRR;
- Hit Rate.

Основные retrieval ablations:

1. TF-IDF;
2. Dense BGE;
3. Dense + reranker;
4. Dense + sparse;
5. Hybrid + reranker;
6. Hybrid + query rewriting + reranker.

Цель — определить, какие компоненты действительно улучшают поиск релевантной информации.

## 6. End-to-end evaluation

Все варианты A–E оцениваются на одном frozen test set.

Основные критерии:

- relevance;
- factual correctness;
- completeness;
- clarity;
- safety;
- hallucinations;
- unsupported claims.

Для RAG дополнительно:

- groundedness;
- использование retrieved evidence;
- provenance.

## 7. Диагностика RAG-ошибок

Плохой ответ RAG-системы может быть связан с разными причинами:

1. нужной информации нет в knowledge base;
2. retriever не нашёл релевантный материал;
3. reranker ухудшил ranking;
4. нужный chunk не попал в final context;
5. context был обрезан;
6. LLM не использовала evidence;
7. LLM добавила unsupported information;
8. ответ оказался небезопасным или неполным.

Поэтому retrieval quality и generation quality анализируются отдельно.

## 8. Критерий финальной архитектуры

Word/char TF-IDF, hybrid retrieval, reranker, query rewriting, thresholds и safety fallback рассматриваются как экспериментальные компоненты.

Компонент остаётся в финальной системе только при измеримом положительном вкладе в качество или надёжность.
