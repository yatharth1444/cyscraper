# cyscraper

AI-powered web scraping tool for extracting structured data from websites.

---

## Requirements

- Python 3.10+
- Playwright
- OpenAI or Gemini API key (recommended)

---

## Installation

```bash
git clone https://github.com/itsOwen/CyberScraper-2077.git
cd CyberScraper-2077

virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
playwright install

```
## Set API Keys

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
````


## Run

```bash
streamlit run main.py
```

Open in browser:

[http://localhost:8501](http://localhost:8501)

Enter a website URL and extract data in your preferred format.

---

## Docker (optional)

```bash
docker build -t cyscraper .

docker run -p 8501:8501 \
-e OPENAI_API_KEY="your-key" \
-e GOOGLE_API_KEY="your-key" \
cyscraper
```

---

## Security

Report vulnerabilities privately:

**Email:** [yatharthsingh1444@gmail.com](mailto:yatharthsingh1444@gmail.com)

```
::contentReference[oaicite:0]{index=0}
```
