# AIT 526-001 Sentiment Analysis 

** Yasser Jaghoori, Andrej Paskalov, Yaseen Trombati

## Overview
This project processes consumer complaint narratives to measure and classify negativity using a combination of NLP techniques and machine learning models. The script cleans and tokenizes text, computes multiple negativity metrics, assigns multi-level labels, and trains both logistic regression and neural network classifiers on sentence embeddings.

## Prerequisites
- Python 3.x
- Install the following libraries:
  - `spacy`
  - `en_core_web_sm` (spaCy model)
  - `beautifulsoup4`
  - `nltk`
  - `pandas`
  - `numpy`
  - `textblob`
  - `transformers`
  - `sentence_transformers`
  - `scikit-learn`
  - `torch`
  - `wordcloud`
  - `tabulate`
  - `matplotlib`
  - `seaborn`
  - `tqdm`

## Setup
1. Install ALL required packages (example below):
   ```bash
   pip install spacy beautifulsoup4 nltk pandas numpy textblob transformers sentence-transformers scikit-learn torch wordcloud tabulate matplotlib seaborn tqdm
   ```
2. Ensure your working directory contains the input data:
   - Original complaints dataset 
   - Cleaned complaints dataset (e.g.removing duplicates and uncessesary columns)
 scores.

## Data Processing Pipeline
1. **Data Loading & Deduplication**  
   - Load the complaints dataset into a DataFrame (`df`).  
   - Create a subset: `Complaint ID` and `Consumer complaint narrative`.  
   - Remove duplicate narratives and sample N number of rows for efficient processing.

2. **Text Cleaning & Tokenization**  
   - `clean_text(text)`:  
     - Removes special characters (except `. ! ?`) and extra whitespace.  
     - Tokenizes into words and sentences using spaCy.  
     - Generate both stemmed and unstemmed tokens.

3. **Negativity Metrics**  
   - `calculate(word_tokens, sent_tokens)`:  
     - Computes negative word frequency ratio (`NegFreq`).  
     - Computes weighted negative sentence ratio (`WgtNegFreq`).  
   - `polarity_score(text)`:  
     - Uses TextBlob to calculate the average polarity of negative/neutral words.  
   - **FinBERT Analysis**  
     - Loads the `ProsusAI/finbert` model with `transformers` pipeline.  
     - `get_finbert_polarity(text)`: averages negative-confidence scores across text chunks.
   - `process_row(row)`: applies all metrics to a single complaint record.

4. **DataFrame & Labeling**  
   - Assemble results into `df_scores` with columns:  
     `Complaint ID`, `NegFreq`, `WgtNegFreq`, `NegFreqStem`, `WgtNegFreqStem`, `AvgPolarity`, `NegConfScore`, `Cleaned Complaint`.
   - Convert metrics to percentiles and compute an overall `AvgPercentile` per complaint.
   - Save Scores to Excel sheet (since the scoring process will take some time, there is no need to constantly rerun if the kernel shutsdown/restarts). 
   - **5-class labeling** with `assign_label(percentile)`:
     - `Neutral`, `Slightly Negative`, `Moderately Negative`, `Very Negative`, `Extremely Negative`. **This method will reference the Excel file, therefore, update the file with your respected file path!**
   - **3-class mapping** with `map_to_3class(label)`:
     - `Weakly Negative`, `Negative`, `Strongly Negative`.
   - Encode labels into numeric arrays (`LabelEncoded`, `LabelEncoded3`).

## Exploratory Visualization
- **Word Cloud**: Visualizes common terms in all cleaned complaints.  
- **Label Distribution**: Bar chart of complaint counts per negativity label.  
- **Metrics Table**: Console output of the first 100 complaints with all computed scores (using `tabulate`).

## Sentence Embeddings & Modeling
1. **Embeddings**  
   - Uses `SentenceTransformer('all-MiniLM-L6-v2')` to convert cleaned complaints into 384-dimensional vectors.

2. **Logistic Regression**  
   - 5-class and 3-class classifiers (`sklearn.linear_model.LogisticRegression`) on embeddings.
   - Outputs classification reports and accuracy scores.

3. **Neural Network (PyTorch)**
   - Defines a feed-forward network (`Net`) with one hidden layer and dropout.  
   - Trains separate models for 5-class and 3-class tasks over 10 epochs.  
   - Reports test accuracy, classification reports, and plots training loss & accuracy per epoch.
  
## Outputs 
- **Visualizations:** Word cloud, bar plot of label distribution, training curves. 
- **Console Reports:** Tables of metrics, classification reports for both models.  


