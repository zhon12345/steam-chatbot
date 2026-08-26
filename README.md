# SteamBot

An intent-based conversational chatbot and recommendation engine designed to help users search, evaluate, and discover games from the Steam catalog. Built using an end-to-end NLP pipeline with **Scikit-Learn**, **FlashText**, and **Streamlit**.

## Technologies Used

- **Frontend:** `streamlit`
- **Data Processing:** `pandas`, `numpy`
- **ML & NLP:** `scikit-learn`, `flashtext`, `nltk`, `rouge-score`
- **Artifact Serialization:** `joblib`, `scipy`

## 📂 Project Structure

```text
steam-chatbot/
│
├── data/
│   ├── games.json                 # Raw Steam games dataset
│   ├── intents.json               # Intent tags, patterns, and static responses
│   ├── eval_conversation.json     # Conversational evaluation benchmark
│   └── eval_data.json             # Retrieval evaluation benchmark
│
├── models/
│   ├── auxiliary_data.joblib      # Intent lookups & metadata term dictionaries
│   ├── games_data.joblib          # Cleaned pandas DataFrame
│   ├── intent_model.joblib        # Fitted Logistic Regression pipeline
│   ├── metadata_matrix.joblib     # Pre-computed metadata TF-IDF sparse matrix
│   ├── metadata_vectorizer.joblib # TF-IDF vectorizer for categories/genres/tags
│   ├── title_matrix.joblib        # Pre-computed title similarity matrix
│   └── title_vectorizer.joblib    # TF-IDF title vectorizer
│
├── app.py                         # Streamlit web application interface
├── chatbot_engine.py              # Core execution pipeline & parsing logic
├── requirements.txt               # Application dependencies
└── README.md
```

## Installation

To run the project locally, follow these steps:

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

**⚠️ Disclaimer**: Due to repository size constraints, the raw Steam Games Dataset (`games.json`) is not committed to this repository. Please download `games.json` manually from the link above and place it inside the `data/` directory before running the pipeline.

### Setup

1. Clone the repository

```bash
  git clone https://github.com/zhon12345/steam-chatbot.git
```

2. Navigate to the project directory

```bash
  cd steam-chatbot
```

3. Setup a virtual environment

```bash
python -m venv .venv

.venv\Scripts\activate
```

4. Install dependencies:

```bash
  pip install -r requirements.txt
```

5. Start the application:

```bash
  streamlit run app.py
```
