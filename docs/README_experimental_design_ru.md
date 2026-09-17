# Дизайн эксперимента

## Исследовательский вопрос

Как prompt engineering, RAG и QLoRA по отдельности и в комбинации влияют на качество, безопасность и groundedness медицинских ответов Qwen2.5-3B-Instruct?

## Экспериментальные варианты

| ID | Модель | Prompt | RAG | Что измеряет |
|---|---|---|---|---|
| A | Base Qwen | Исходный | Нет | Исходный baseline |
| B | Base Qwen | Улучшенный | Нет | Вклад prompt engineering |
| C | Base Qwen | Улучшенный | Да | Вклад RAG для base-модели |
| D | QLoRA Qwen | Улучшенный | Нет | Вклад fine-tuning без RAG |
| E | QLoRA Qwen | Улучшенный | Да | Комбинированная система |

Основные ablation-сравнения:

- **A vs B** — влияние prompt engineering;
- **B vs C** — влияние RAG на base-модель;
- **B vs D** — влияние QLoRA без retrieval;
- **C vs E** — влияние QLoRA при наличии RAG;
- **D vs E** — влияние RAG на QLoRA-модель.

## Протокол оценки

Train используется для supervised fine-tuning, development set — для выбора prompt, retrieval-конфигурации и гиперпараметров, а frozen test — только для финальной оценки.

Все end-to-end варианты сравниваются на одних и тех же test questions с одинаковыми generation settings.

Retrieval оценивается отдельно от generation. Для retrieval используются Recall@k, MRR и Hit Rate. Для end-to-end оценки учитываются factual correctness, relevance, completeness, safety, hallucinations и unsupported claims; для RAG дополнительно оценивается groundedness.

Финальная архитектура включает только компоненты, вклад которых подтверждён экспериментально.
