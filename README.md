# Curobot

Medical chatbot Flask application using Gemini 2.5 Pro

### TechStack Used

- Python
- LangChain
- Flask
- Gemini 2.5 Pro
- Pinecone

### Steps to install

1. Create virtual environment

```bash
conda create -n curenv python==3.10 -y
```

2. Activate the virtual environment

```bash
conda activate curenv
```

3. Install requirements

```bash
pip install -r requirements.txt
```

4. Create a .env file in the root directory and add Pinecone and Gemini credentials

```bash
PINECONE_API_KEY="xxxxxxxxxxxxx"
GOOGLE_API_KEY="xxxxxxxxxxxxxxx"
```

5. Store embeddings in Pinecone Vector Database

```bash
python store_index.py
```

6. Run the app and open localhost

```bash
python app.py
```