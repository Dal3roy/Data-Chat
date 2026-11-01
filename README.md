# Data-Chat

Chat with your CSV files using a local LLM. Upload a CSV, ask questions in plain English, and get the generated SQL, a result table, a chart and a short summary. Everything runs on your machine: nothing is sent to an external service.

## How it works
1. The CSV file is loaded into an in-memory DuckDB table.
2. A local model from Ollama turns your input question into a read-only SQL query using the table schema and a few sample rows.
3. The query is validated (single SELECT Only) and run. On an error, the model gets one or two retries with the error message.
4. The result is charted with Plotly and summarised by the model.

Ask for a chart type by name ("scatter plot of price vs area", "histogram of age") and the app will honour it.

## Features 
* Local and private: Ollama + DuckDB, no API keys.
* Chat history saved locally, with follow-up questions.
* Dark and light mode.
* Scatter, line, bar, histogram, box and pie charts.

## Setup

# 1. Install Ollama from https://ollama.com, then pull a model: 
ollama pull qwen2.5-coder:7b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

The app opens at http://localhost:8501. Upload a CSV and ask questions.

To use a bigger model for better SQL, change MODEL at the top of ``app.py``, e.g. qwen2.5-coder:14b

## Notes 
* Chat history is stored in chat.json. The CSV itself is not stored, so re-upload it to ask new questions.
* Small models can write incorrect SQL. The generated query is always shown so you can make sure.
