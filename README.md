# Медицинский LLM-ассистент: QLoRA, RAG и Semantic Search

Экспериментальный LLM-пайплайн для ответов на медицинские вопросы на базе `Qwen/Qwen2.5-3B-Instruct`. Проект объединяет Prompt Engineering, QLoRA fine-tuning через PEFT, RAG по клиническим рекомендациям и dense semantic search на BGE embeddings. Качество компонентов оценивается отдельно: проверяются retrieval, влияние system prompt, обоснованность RAG-ответов найденными источниками и итоговые конфигурации на фиксированной тестовой выборке.

## Tech Stack

**LLM / Fine-tuning:** Python, PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, Qwen2.5-3B-Instruct, QLoRA

**RAG / Retrieval:** SentenceTransformers, BAAI/bge-base-en-v1.5, NumPy dense search, FAISS `IndexFlatIP`, PyMuPDF

**Data / Evaluation:** pandas, NumPy, scikit-learn, Hugging Face Datasets, Jupyter

## Что реализовано

- Очистка медицинского QA-датасета, удаление точных и семантических дубликатов, формирование `train/dev/test/debug` выборок с проверкой пересечений.
- Сравнение исходного и улучшенного system prompt через слепую оценку на фиксированной dev-выборке.
- Supervised fine-tuning `Qwen2.5-3B-Instruct` с QLoRA и PEFT, расчет loss только по токенам ответа модели и выбор checkpoint с приоритетом медицинской безопасности.
- Подготовка базы знаний из 8 клинических рекомендаций WHO, CDC и VA/DoD с сохранением источника, раздела и страницы.
- Chunking документов, построение BGE embeddings и dense semantic retrieval по cosine similarity.
- RAG с передачей найденных фрагментов в контекст LLM и ссылками на использованные источники.
- Эксперименты с TF-IDF, NumPy search, FAISS, cross-encoder reranking и порогом similarity.
- Финальное слепое сравнение базовой модели, Prompt Engineering, RAG и QLoRA на фиксированной тестовой выборке из 300 вопросов.

## Архитектура и pipeline

Проект состоит из нескольких связанных экспериментальных веток.

QA-датасет проходит проверку качества, нормализацию, удаление дубликатов и разделение на train, dev и test. Dev-выборка используется для Prompt Engineering и выбора QLoRA checkpoint, test остается фиксированной до финального сравнения.

Для RAG отдельно формируется база знаний из PDF с клиническими рекомендациями. Документы парсятся с сохранением metadata, разбиваются на chunks и кодируются моделью `BAAI/bge-base-en-v1.5`. Для пользовательского запроса строится embedding, после чего выполняется dense similarity search по embeddings документов. Три наиболее релевантных фрагмента передаются `Qwen2.5-3B-Instruct` как дополнительный контекст.

Финальный этап сравнивает несколько фиксированных конфигураций модели на одной test-выборке.

## LLM Fine-tuning

Базовая модель для inference и fine-tuning - `Qwen/Qwen2.5-3B-Instruct`, загружаемая через Hugging Face Transformers.

Fine-tuning выполняется с QLoRA через PEFT:

- базовая модель загружается в 4-bit;
- используется NF4 quantization с double quantization;
- обучаются только LoRA adapters, полные веса базовой модели не обновляются;
- LoRA применяется к `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`;
- `r=16`, `alpha=32`, `dropout=0.05`;
- число обучаемых параметров составляет 29 933 568, около 0.96% параметров модели;
- gradient checkpointing используется для снижения потребления памяти.

Обучающие примеры имеют chat format с ролями `system`, `user` и `assistant`. Loss рассчитывается только по токенам ответа `assistant`. Токены system prompt и пользовательского сообщения получают label `-100`.

Перед fine-tuning из train split удаляются 3 699 примеров, которые эвристика определила как высокорисковые прямые инструкции по назначению лекарств. Примеры длиннее 768 токенов исключаются вместо обрезания. После фильтрации для QLoRA остается 104 772 обучающих примера.

Основные параметры обучения:

| Параметр | Значение |
| --- | --- |
| `max_steps` | 5000 |
| Effective batch size | 4 |
| Learning rate | `2e-4` |
| Scheduler | cosine |
| Warmup | 150 steps |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 768 |
| Интервал сохранения checkpoint | 1000 steps |

Checkpoints с 1000 по 5000 шаг сравниваются на фиксированных 50 вопросах из dev-выборки без RAG. Сначала минимизируется число критических нарушений медицинской безопасности, затем учитываются метрики качества. По этому правилу выбран `checkpoint-1000`.

Сохраняется PEFT adapter, а не объединенная копия полной модели. Во время эксперимента обучение прерывалось и затем продолжалось из сохраненного checkpoint.

## RAG и Semantic Search

База знаний строится из 8 PDF-документов с клиническими рекомендациями:

- CDC STI Treatment Guidelines;
- WHO guideline for pharmacological treatment of hypertension;
- VA/DoD guidelines по asthma, CKD, low back pain, major depressive disorder, pregnancy и type 2 diabetes.

Документы обрабатываются с помощью PyMuPDF. Для разных источников используются отдельные правила парсинга. Для фрагментов сохраняются `document_id`, источник, название документа, год, `section_path`, `page` и `pdf_page`.

После парсинга получен 1 251 структурированный фрагмент документа.

Chunking выполняется токенизатором `BAAI/bge-base-en-v1.5` внутри разделов документа. Целевой размер chunk составляет 384 токена, overlap - 64 токена. После разбиения база знаний содержит 2 135 chunks.

Для каждого chunk строится нормализованный embedding с помощью `BAAI/bge-base-en-v1.5`. При построении embedding к тексту добавляются название документа и путь к разделу. Для запроса используется BGE retrieval prefix.

Запрос кодируется той же embedding-моделью. Затем его нормализованный embedding сравнивается с embeddings документов через dot product, который для нормализованных векторов соответствует cosine similarity. Результаты сортируются по similarity score, после чего в RAG-контекст передаются top-3 chunks вместе с информацией об источнике, разделе и странице. В ответе модель должна указывать ссылки вида `[Source N]`.

Embeddings сохраняются в `embeddings.npy`, chunks - в `chunks.parquet`, параметры retrieval - в `config.json`. Отдельная vector database в текущей версии проекта не используется.

### Качество retrieval

Для retrieval подготовлен фиксированный benchmark из 47 вопросов.

| Метрика | BGE dense retrieval | TF-IDF baseline |
| --- | ---: | ---: |
| Hit@1 | 0.830 | 0.340 |
| Hit@3 | 0.936 | 0.596 |
| Hit@5 | 0.936 | 0.702 |
| MRR@10 | 0.882 | 0.509 |
| Document Hit@1 | 1.000 | 0.936 |

Так как `Hit@3` и `Hit@5` для BGE совпали, в основном RAG используется `top_k=3`.

FAISS `IndexFlatIP` протестирован на тех же нормализованных BGE embeddings. Для всех 47 benchmark-вопросов top-10 результатов FAISS полностью совпал с NumPy implementation. В основном RAG v1 используется NumPy dense search.

Отдельно проверены два cross-encoder reranker:

- `cross-encoder/ms-marco-MiniLM-L6-v2`;
- `ncbi/MedCPT-Cross-Encoder`.

Оба варианта ухудшили retrieval-метрики на фиксированном benchmark, поэтому reranking не включен в итоговый pipeline.

## Prompt Engineering

Исходный system prompt из датасета сравнивается с более строгой медицинской версией.

Улучшенный prompt задает правила для работы с неопределенностью, разделения подтвержденных фактов и предположений, ограничения неподтвержденных медицинских утверждений, осторожности при рекомендациях лекарств и рекомендаций обратиться за срочной помощью при потенциально опасных симптомах.

Prompt настраивался только на `debug` subset из 30 примеров. После фиксации версии было проведено слепое сравнение на 50 примерах из dev-выборки, которые не использовались при настройке prompt. Улучшенный prompt победил в 36 случаях, исходный - в 14.

Для RAG используется дополнительный набор инструкций: медицинские утверждения должны опираться на найденные фрагменты клинических рекомендаций и сопровождаться ссылками `[Source N]`. Если найденных данных недостаточно, модель должна явно указать это в ответе.

Few-shot prompting и Chain-of-Thought в проекте не используются.

Также был проверен экспериментальный LLM-based механизм оценки достаточности найденного контекста. Он не был включен в основной pipeline, так как оказался слишком консервативным на development benchmark.

## Данные

Для обучения используется датасет `Doctor-HealthCare-100k` с медицинскими парами `input/output`. Исходный CSV не хранится в репозитории из-за размера и загружается отдельно с [Kaggle](https://www.kaggle.com/datasets/divyanshu2000/doctor-healthcare-100k).

Исходный датасет содержит 112 156 примеров.

Перед разделением данных выполняются нормализация текста, удаление точных дубликатов и поиск семантических near-duplicates. Для поиска семантически близких вопросов используются `BAAI/bge-small-en-v1.5` и FAISS `IndexFlatIP`. Пары с cosine similarity `>= 0.97` удаляются до разделения выборки.

После очистки остается 110 325 примеров.

| Выборка | Размер | Назначение |
| --- | ---: | --- |
| Train | 109 025 | QLoRA training |
| Dev | 1 000 | разработка и выбор модели |
| Test | 300 | фиксированная финальная оценка |
| Debug | 30 | настройка prompt, подвыборка из dev |

Файлы `train.csv`, `dev.csv`, `test.csv` и `debug.csv` создаются локально при подготовке данных и не хранятся в Git.

Для RAG отдельно используется корпус из 8 PDF с клиническими рекомендациями. Test set не используется для настройки prompt, retrieval или выбора QLoRA checkpoint.

## Оценка качества

Оценка разделена по компонентам. Наличие RAG или fine-tuning само по себе не считается улучшением.

### Retrieval

Retrieval оценивается на benchmark из 47 вопросов с помощью Hit@1, Hit@3, Hit@5, MRR@10 и Document Hit@1. Дополнительно сравниваются TF-IDF baseline, NumPy и FAISS, cross-encoder reranking и similarity threshold.

### RAG

На 47 development-вопросах RAG сравнивается с конфигурацией, использующей только улучшенный prompt.

- RAG лучше в 29 случаях;
- baseline лучше в 13;
- 5 сравнений закончились ничьей.

На отдельной выборке из 19 вопросов, которые покрываются используемыми клиническими рекомендациями, RAG выиграл 17 сравнений.

Дополнительная проверка показала, что не все части сгенерированных ответов полностью подтверждаются retrieved evidence. На 8 вопросах, намеренно не покрываемых базой знаний, корректный отказ от ответа сработал в 5 случаях.

### Выбор QLoRA checkpoint

Пять checkpoints оцениваются на фиксированных 50 вопросах из dev-выборки по relevance, instruction following, unsupported claims, medical safety, overall quality и critical safety violations.

По правилу с приоритетом безопасности выбран `checkpoint-1000`.

### Финальное сравнение

На фиксированной test-выборке из 300 вопросов сравниваются пять конфигураций. Перед оценкой ответы перемешиваются.

| Вариант | Конфигурация | Средний overall quality, 0-2 | Critical safety violations |
| --- | --- | ---: | ---: |
| A | Base + original prompt | 1.367 | 21 / 300 |
| B | Base + improved prompt | 1.407 | 30 / 300 |
| C | Base + improved prompt + RAG | 1.010 | 44 / 300 |
| D | QLoRA + improved prompt | 0.453 | 68 / 300 |
| E | QLoRA + improved prompt + RAG | 0.383 | 103 / 300 |

Улучшенный system prompt немного повысил `overall_quality` относительно исходного prompt, однако одновременно выросло число критических нарушений безопасности.

Текущие конфигурации RAG и QLoRA не улучшили результаты на фиксированной test-выборке. Дополнительный RAG v2 с FAISS и similarity threshold `0.65` также не улучшил финальный результат.

## Структура проекта

```text
01_data_quality_and_split.ipynb
    Проверка качества данных, дедупликация и формирование выборок

02_baseline_and_prompting.ipynb
    Базовая Qwen-модель, эксперименты с system prompt и слепое сравнение

03_knowledge_base_and_dense_retrieval.ipynb
    Парсинг PDF, chunking, BGE embeddings и оценка retrieval

04_retrieval_experiments.ipynb
    TF-IDF, NumPy vs FAISS, reranking и similarity threshold

05_rag.ipynb
    RAG-генерация, ссылки на источники и оценка обоснованности ответов

06_qlora.ipynb
    Подготовка данных, QLoRA fine-tuning и выбор checkpoint

07_final_evaluation.ipynb
    Финальное сравнение пяти конфигураций на test-выборке

retrieval.py
    Функции загрузки retrieval-данных, кодирования запроса и NumPy dense search

requirements.txt
    Основные зависимости проекта
```

Крупные исходные данные, сгенерированные выборки, retrieval-артефакты и model checkpoints не хранятся в Git.

## Установка и запуск

Проект разрабатывался на Python 3.10.11 в отдельном окружении.

Основные зависимости зафиксированы в `requirements.txt`:

```bash
pip install -r requirements.txt
```

Для воспроизведения экспериментов требуется отдельно загрузить исходный `Doctor-HealthCare-100k` с Kaggle и PDF с клиническими рекомендациями.

Ноутбуки отражают отдельные этапы исследования, а не единый автоматизированный end-to-end скрипт. В процессе экспериментов переиспользуются сохраненные выборки, embeddings, результаты retrieval и model checkpoints.

QLoRA training выполнялся на CUDA GPU. Один из запусков обучения был продолжен из сохраненного checkpoint после прерывания. Для снижения потребления видеопамяти используются 4-bit quantization, gradient checkpointing и 8-bit optimizer.

## Ограничения

- На финальной test-выборке RAG не улучшил baseline. Одна из проблем текущей версии в том, что top-3 найденных chunks добавляются в контекст и для вопросов, которые база знаний покрывает плохо.
- QLoRA также ухудшила итоговые метрики. Подготовка обучающих данных и параметры fine-tuning требуют дополнительных экспериментов.
- База знаний состоит из 8 клинических рекомендаций, поэтому retrieval проверяется на ограниченном наборе медицинских тем.
- Эксперименты проводились с моделью на 3B параметров и в пределах доступных локальных вычислительных ресурсов. Большой перебор моделей и гиперпараметров не проводился.
- Не проводился полный поиск по `chunk_size`, `top_k`, retrieval thresholds, LoRA-параметрам и training settings. Проверялись отдельные гипотезы, которые можно было воспроизвести на доступном GPU.
