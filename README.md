# IT Service Ticket Classification

This project was completed as part of the [**Deep Learning**](https://www.ef.uns.ac.rs/ofakultetu/studijski-programi/mas-advanced-data-analytics-in-business-files/deep-learning.pdf) course (ADA16) within the [Advanced Data Analytics in Business study program](https://www.ef.uns.ac.rs/ofakultetu/studijski-programi/mas-advanced-data-analytics-in-business.php).

---

##  Content

1. Project Description
2. Data
3. Preprocessing
4. Model
5. Training
6. Results
7. Application
8. Repo structure
9. Future Work

---

## 1. Project Description

**Goal:** Automatic classification of IT requests by topic to speed up routing and processing.


**Context:** university project for the course [*Deep Learning*](https://www.ef.uns.ac.rs/ofakultetu/studijski-programi/mas-advanced-data-analytics-in-business-files/deep-learning.pdf) 


**Briefly:** Fine-tuning of the pre-trained transformer model on the ticket corpus.

---

## 2. Data

* **Source:** [Kaggle](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
* **Format:** table with columns `Document` (ticket text) and `Topic_group` (category).
* **Info (df.info()):**

  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 47837 entries, 0 to 47836
  Data columns (total 2 columns):
   0   Document      47837 non-null  object
   1   Topic_group   47837 non-null  object
  dtypes: object(2)
  memory usage: 747.6+ KB
  ```
* **Splitting the sample:**

  * Train size: **34442**
  * Validation size: **3827**
  * Test size: **9568**
* **Class balance:**
* 
  <img width="602" height="298" alt="image" src="https://github.com/user-attachments/assets/b19b6720-c013-40aa-b06d-8297fb6516a2" />

  <img width="382" height="296" alt="image" src="https://github.com/user-attachments/assets/4c1ddf76-77af-41ea-b986-0e70a040a42b" />


---

## 3. Preprocessing

* **Tokenization:** was used `AutoTokenizer("distilbert-base-uncased")` (Hugging Face).

  ```python
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
  ```


* **Working with labels:** the dictionary `label2id` and the reverse `id2label` are used (to store labels as numbers in the Dataset and return readable names during inference).
  ```python
  label2id = {
      "Hardware": 0,
      "HR Support": 1,
      "Access": 2,
      "Miscellaneous": 3,
      "Storage": 4,
      "Purchase": 5,
      "Internal Project": 6,
      "Administrative rights": 7
  }
  id2label = {v: k for k, v in label2id.items()}
  ```

  Mapping fixes the order of classes and simplifies pairing with the model (numeric labels are needed for loss/metrics).

---

## 4. Model

* **Basic architecture:** [`distilbert-base-uncased` (lightweight BERT encoder)](https://huggingface.co/distilbert/distilbert-base-uncased)
* **Libraries:** `transformers`, `torch`, `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `ipywidgets`.
* **Class:** standart `SequenceClassification` (fully connected layer at the output with `num_labels=8`).
* **Loss:** CrossEntropyLoss (built into the HF model class).

---

## 5. Training

**Final TrainingArguments used in training:**

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to=[]
)
```


* `output_dir="./results"` — папка для чекпойнтов и финальной модели.
* `evaluation_strategy="epoch"` — валидация выполняется в конце каждой эпохи; удобно мониторить поведение модели по эпохам.
* `save_strategy="epoch"` — сохраняем чекпоинт после каждой эпохи.
* `learning_rate=2e-5` — типичное значение LR для fine-tuning BERT-подобных моделей: достаточно маленькое, чтобы не «сломать» предобученные веса.
* `per_device_train_batch_size=32` — размер батча на одну GPU; при использовании 2 GPU effective batch = 32 × 2 = **64** (меньше шагов/эпоха, плавнее градиенты).
* `per_device_eval_batch_size=32` — батч для валидации (снижает время валидации при сохранённой точности).
* `num_train_epochs=3` — 2–4 эпох обычно достаточно для трансформеров; слишком много — риск переобучения.
* `weight_decay=0.01` — L2-регуляризация (AdamW) — помогает обобщению.
* `logging_dir="./logs"` — куда писать логи для tensorboard.
* `logging_strategy="epoch"` и `logging_steps=50` — логируем в конце эпох (можно менять на шаги; при logging_strategy="epoch" logging_steps не критичен).
* `load_best_model_at_end=True` — по окончании тренировки загрузить чекпоинт с наилучшей метрикой валидации (если задан metric_for_best_model).
* `report_to=[]` — отключён W&B (чтобы не блокироваться ожиданием логина).

**Optimizer / scheduler:** Trainer uses AdamW and a standard scheduler (linear warmup) by default. This is a suitable setting for fine-tuning transformers.

---

## 6. Results

**Training / Validation loss:**

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1     |
| ----- | ------------- | --------------- | -------- | --------- | ------ | ------ |
| 1     | 0.7024        | 0.4257          | 0.8560   | 0.8567    | 0.8560 | 0.8557 |
| 2     | 0.3578        | 0.3743          | 0.8712   | 0.8718    | 0.8712 | 0.8709 |
| 3     | 0.2600        | 0.3740          | 0.8725   | 0.8722    | 0.8725 | 0.8721 |

**Interpretation:**

* Train loss decreases significantly → the model learns.

**Metrics (accuracy, precision, recall, f1) on the test sample:**

| Класс                 | Precision | Recall | F1-score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
| Hardware              | 0.87      | 0.86   | 0.87     | 2724    |
| HR Support            | 0.89      | 0.90   | 0.89     | 2183    |
| Access                | 0.89      | 0.93   | 0.91     | 1425    |
| Miscellaneous         | 0.84      | 0.84   | 0.84     | 1412    |
| Storage               | 0.89      | 0.91   | 0.90     | 555     |
| Purchase              | 0.93      | 0.90   | 0.91     | 493     |
| Internal Project      | 0.88      | 0.88   | 0.88     | 424     |
| Administrative rights | 0.82      | 0.74   | 0.78     | 352     |


**Confusion matrix**

<img width="592" height="491" alt="Снимок экрана 2025-10-03 142635" src="https://github.com/user-attachments/assets/1f65185f-7309-4ff6-850d-cb07ba08a3b9" />


---

## 7. Application

**Jupyter-виджет** (ipywidgets) for interactive ticket entry and receiving a prediction directly on your laptop.

<img width="1516" height="654" alt="Снимок экрана 2025-10-03 133425" src="https://github.com/user-attachments/assets/fbde0b2a-885a-4545-94eb-74452681bc8b" />



## 8. Repository structure
```
├── data/               # Исходные датасеты (raw, processed)
├── notebooks/          # Jupyter ноутбуки (EDA, обучение, inference)
├── src/                # Скрипты: train.py, inference.py, service.py
├── models/             # Сохранённые модели / ticket_classifier/
├── results/            # логи/чекпойнты / tensorboard
├── figures/            # графики (loss/metrics/confusion)
└── README.md
```

---

## 9. Future work

* Prototype API (FastAPI) and/or Streamlit UI.
* Think about class imbalance: weighted loss, oversampling, focal loss.
* Document the pipeline as a Docker image (for presentation/portfolio).
