# Notebooks

This section contains Jupyter notebooks used for **exploring, preprocessing, and modeling**.

## Note

* The main notebook with model training is [here](https://github.com/Lunalykaya/IT-Service-Ticket-Classification/blob/main/notebooks/data-tokenization-and-model%20(1).ipynb).


## [`data-overview`](https://github.com/Lunalykaya/IT-Service-Ticket-Classification/blob/main/notebooks/data-overview.ipynb)

This notebook performs an **initial exploration of the dataset**:

**Key steps:**

* Importing libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`)
* Loading the dataset and checking for missing/duplicate values
* Counting unique ticket categories and visualizing their distribution
* Visualizing ticket lengths (mean, median, min, max)
* Displaying example tickets per category

**Purpose:**
Provides a **general understanding of the dataset**, its size, category distribution, and typical ticket content.

---

##  [`ticket-class-complete-eda-baseline-lr-dt (1)`](https://github.com/Lunalykaya/IT-Service-Ticket-Classification/blob/main/notebooks/ticket-class-complete-eda-baseline-lr-dt%20(1).ipynb)

This notebook focuses on **preprocessing, feature extraction, and baseline models**:

**Key steps:**

* Downloading the latest dataset version using `kagglehub`
* Cleaning text:

  * Lowercasing, removing numbers and punctuation
  * Removing HTML tags
  * Tokenization
  * Removing stopwords (including custom stopwords)
  * Expanding contractions (`don't → do not`)
  * Lemmatization
* Exploratory analysis:

  * Token count per document
  * Word frequency distribution
  * Lexical diversity
  * Average word and ticket lengths
  * WordCloud visualization
* Feature extraction:

  * TF-IDF vectorization (`unigrams` and `bigrams`)
* Target encoding with `LabelEncoder`
* Baseline classification models:

  * Logistic Regression
  * Random Forest
* Evaluation using classification reports

**Purpose:**
Provides a **full end-to-end baseline pipeline** for text preprocessing and classical ML models for ticket classification.


##  [`data-tokenization-and-model (1)`](https://github.com/Lunalykaya/IT-Service-Ticket-Classification/blob/main/notebooks/data-tokenization-and-model%20(1).ipynb)

*To be added:*

* Tokenization for transformer models
* Fine-tuning or using pretrained DistilBERT model
* Model training and evaluation
