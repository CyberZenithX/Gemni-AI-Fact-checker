# Fake News Detector AI — Streamlit + Gemini

This is a **Streamlit web application** utilizing **Google's Gemini API** to perform **fact-checking on news stories, social media posts, and assertions**. It separates factual claims from a document, verifies them with Gemini, and receives back a structured judgment with confidence values, justification, and (where applicable) evidence citations.

The goal is to provide a **local, rapid prototype** of an AI-driven fake news classifier with a simple, user-friendly UI.

---

## ✨ Features
- **Interactive Streamlit UI** – paste an article or claim and see immediate results.
- **Claim extraction** – automatically finds up to *N* claims from the input text.
- **Structured analysis** – all claims are labeled as **REAL / FAKE / UNVERIFIED**.
- **Confidence scoring** – numeric confidence (0–100) for every claim and overall verdict.
- **Evidence citations** – Gemini returns URLs/snippets if available (never hallucinating, if it is uncertain, it returns no sources).
- **JSON Output** – raw structured JSON is output for auditing or reuse.
- **Flexible controls** – choose model (e.g., `gemini-2.5-flash`), alter max claims, and set a thinking budget.
- **Fully customizable UI** - any colors can be chosen for the the theme of the web app.
- **Screencast functionality** - you can record and download your screen within the web app. (your data is only stored locally)
---

## Getting Started

### Prerequisites
- Python 3.9+
- A **Gemini API Key** from Google AI Studio (It's free)

### Installation
```bash
# Clone repo (if using git)
git clone https://github.com/CyberZenithX/Gemni-AI-Fact-checker.git
cd Gemni-AI-Fact-checker

# Install dependencies
pip install streamlit google-genai pydantic
```

### Environment Setup
Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Or, copy and paste the key directly into the **Streamlit sidebar** when running.

### Run the App
```bash
streamlit run AI-Fact-Checker.py
```

---
## Usage
1. Launch the app in your browser (Streamlit will serve a local URL).
2. Copy and paste a news article, tweet, or claim.
3. Click **Run fact-check**.
4. Check it out:
- Overall decision (REAL / FAKE / LIKELY REAL / LIKELY FAKE / UNVERIFIED)
- Decision at every level of claim with justification & evidence
- Unprocessed JSON output

---

## ⚠️ Limitations & Caveats
- Gemini may **not always have access to real-time web**. Temporally sensitive or obscure claims may be flagged **UNVERIFIED**.
- Don't rely exclusively on AI decisions when making critical decisions. Always cross-check with **independent fact-checking agencies**.
- URLs provided can be partial when Gemini cannot find credible evidence.

---

## Future Roadmap
- [ ] Integrate **Google Fact Check Tools API** for stronger proof validation.
- [ ] Add **search engine cross-referencing** (e.g., SerpAPI, NewsGuard) for better sourcing.
- [ ] Save results to a database for auditing, analysis and learning.
- [ ] Provide export capabilities (CSV / PDF report).

---

## License
This project is provided as-is under the MIT License.

---

## Acknowledgments
- [Google Gemini API](https://ai.google.dev)
- [Streamlit](https://streamlit.io)
- Fact-checking organizations like PolitiFact, Snopes, and Full Fact, whose practices inspired the structure of this tool.

---

**Summary**: This project is a **prototype AI fact-checking assistant** that combines Gemini’s reasoning power with a simple Streamlit UI to provide quick and interpretable fake news detection.
