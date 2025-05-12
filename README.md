# IT-Service-Ticket-Classification
IT Service Ticket Classification (NLP) [in progress]


Project Title: Support Ticket Classification (Support Ticket Triage)
Dataset: IT Service Ticket Classification Dataset

Project Overview
This project focuses on automating the classification and routing of IT service support tickets using supervised machine learning. The objective is to assign each ticket to the appropriate department or category based on its textual content.

1. Data Loading and Initial Exploration

* The CSV file was successfully loaded.
* Preliminary inspection (head, tail, dtypes, missing values) confirmed that the structure and content were intact.
* The dataset contains 47,837 support tickets across multiple topic groups.

2. Text Preprocessing

* All text entries were cleaned by removing HTML tags, links, special characters, and extra whitespace.
* Texts were converted to lowercase.
* Tokenization was performed at the word level.
* Stopwords were removed, and lemmatization was applied to normalize the tokens.
* Manual inspection of a sample confirmed the quality of the preprocessing.

3. Vectorization

* TF-IDF vectorization was applied using unigrams and bigrams (min\_df=5, max\_df=0.8).
  

4. Model Training and Evaluation

* The dataset was split into training and test sets.
* Baseline classifier was trained: Logistic Regression.

5. Ongoing Work

* Hyperparameter tuning.
* Evaluation based on accuracy, precision, recall, and F1-score for each topic group.
* A confusion matrix generation and visualization to identify underperforming classes.
* Additional models such as Decision Trees, Random Forests, XGBoost, RNNs, and BERT are currently being implemented.
* A comparative analysis across all models will be conducted.
* The final model will be deployed via a REST API built with FastAPI, containerized with Docker, and prepared for integration into a larger IT support workflow.
