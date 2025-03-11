# web_chatbott

This is a simple Streamlit app that scrapes a webpage, extracts text, creates embeddings using OpenAI, and lets you ask questions about the content.

---

## Prerequisites

- Python 3.8 or higher  
- OpenAI API key saved in a `.env` file

---

## Setup

### 1. Create Environment
```bash
conda create -n scraper-chatbot python=3.11 -y
conda activate scraper-chatbot

pip install -r requirements.txt


OPENAI_API_KEY=your_openai_api_key_here

streamlit run app.py

### How to Use
1. Enter one or more webpage URLs.
2. Ask questions related to the scraped content.
3. Get accurate responses from the chatbot.