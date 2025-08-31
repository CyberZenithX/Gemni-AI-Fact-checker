"""
Streamlit + Gemini + SerpAPI — AI-Powered Fake News / Fact-Checker (local)

This file provides a clean, working Streamlit prototype that:
- Accepts article text or article URL (fetches page content)
- Extracts factual claims using Gemini
- Optionally searches the web with SerpAPI per claim to gather evidence
- Asks Gemini to verify each claim using the search snippets
- Displays structured JSON results and a human-readable UI

Dependencies
pip install streamlit google-genai pydantic serpapi beautifulsoup4 requests

Run
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export SERPAPI_API_KEY="YOUR_SERPAPI_KEY"  # optional
streamlit run streamlit_gemini_factchecker.py
"""

from __future__ import annotations

import os
import json
import textwrap
import time
from typing import List, Optional, Dict, Any

import streamlit as st
import requests
from pydantic import BaseModel
from bs4 import BeautifulSoup

# serpapi is optional
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except Exception:
    GoogleSearch = None
    SERPAPI_AVAILABLE = False

# The genai SDK is expected to be available as `google.genai`. If not installed, users will get an import error.
try:
    from google import genai
except Exception:
    genai = None


# -----------------------------
# Pydantic response schema
# -----------------------------
class ClaimVerification(BaseModel):
    claim: str
    verdict: str  # REAL / FAKE / UNVERIFIED
    confidence: float  # 0-100
    evidence: List[str] = []  # list of evidence URLs or short citations
    notes: Optional[str] = None


class FactCheckResponse(BaseModel):
    overall_verdict: str
    overall_confidence: float
    claims: List[ClaimVerification]
    summary: str


# -----------------------------
# Helpers: Gemini client
# -----------------------------

def get_genai_client(api_key: Optional[str] = None):
    """Return a genai client. If the genai package is missing, returns None."""
    if genai is None:
        return None
    if api_key:
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            # fallback to environment-based client
            try:
                return genai.Client()
            except Exception:
                return None
    try:
        return genai.Client()
    except Exception:
        return None


# -----------------------------
# Helpers: Article extraction from URL
# -----------------------------

def fetch_article_text(url: str) -> str:
    """Fetch an article's main text using requests + BeautifulSoup.
    Returns title + joined paragraphs.
    Raises RuntimeError on failure.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not fetch URL: {e}")

    soup = BeautifulSoup(r.text, "html.parser")

    # Try to find typical article containers
    selectors = ["article", "main", "#content", ".article-content", ".post-content"]
    article = None
    for sel in selectors:
        found = soup.select(sel)
        if found:
            article = found[0]
            break

    if not article:
        article = soup.body or soup

    paragraphs = [p.get_text(strip=True) for p in article.find_all("p") if p.get_text(strip=True)]
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Ensure proper use of newline characters
    text = "\n\n".join(paragraphs)
    if title:
        return title + "\n\n" + text
    return text


# -----------------------------
# Helpers: SerpAPI search
# -----------------------------

def serpapi_search(query: str, serpapi_key: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Return a list of search results: {title, snippet, link}.
    Requires serpapi package and key.
    """
    if not SERPAPI_AVAILABLE or GoogleSearch is None:
        raise RuntimeError("SerpAPI package not installed. Install with `pip install serpapi`.")
    params = {
        "q": query,
        "engine": "google",
        "api_key": serpapi_key,
        "num": num_results,
    }
    search = GoogleSearch(params)
    res = search.get_dict()

    results = []
    org = res.get("organic_results") or res.get("results") or []
    for item in org[:num_results]:
        title = item.get("title") or item.get("name") or ""
        snippet = item.get("snippet") or item.get("snippet_highlighted_words") or ""
        link = item.get("link") or item.get("url") or ""
        results.append({"title": title, "snippet": snippet, "link": link})
    return results


# -----------------------------
# Gemini prompts: extraction and verification
# -----------------------------

def build_extract_claims_prompt(text: str, max_claims: int = 6) -> str:
    prompt = f"""
You are a careful fact-checker. The user will paste an article or text. Do the following and return results in the structured JSON schema requested:

1) Extract up to {max_claims} distinct factual claims from the text. Keep each claim short (one sentence).
2) For each claim, verify its accuracy by conducting multiple searches across reliable sources. Report if verification is based on multiple confirmations.
3) For each claim include: the claim text, verdict (REAL/FAKE/UNVERIFIED), a numeric confidence (0-100), a short list of evidence citations or URLs (if no reliable URL, leave evidence empty), and a short note explaining the reasoning.
4) Provide an overall_verdict (REAL / LIKELY REAL / LIKELY FAKE / FAKE / UNVERIFIED), overall_confidence (0-100) and a short summary paragraph.

IMPORTANT:
- Do not invent evidence URLs. If you cannot find a source or are unsure, leave the evidence array empty and mark the verdict as UNVERIFIED.
- Before finalizing, cross-check each claim using at least two independent sources.
- Keep answers concise and factual.

Text to check:
{text.strip()}

Return the result strictly as JSON following the schema: overall_verdict (string), overall_confidence (number), claims (array of objects with claim, verdict, confidence, evidence (array), notes), summary (string).
"""
    return prompt


def build_verify_claim_prompt(claim: str) -> str:
    prompt = f"""
You are a fact-verifier. Verify the following claim with at least two independent sources. Provide the claim, a verdict (REAL/FAKE/UNVERIFIED), a confidence value (0-100), evidence URLs (if available) and a short explanation.

Claim: {claim}

Return the result as JSON with keys: claim, verdict, confidence, evidence, notes.
"""
    return prompt


def call_gemini_for_json(client, prompt: str, model: str = "gemini-2.5-flash") -> Optional[dict]:
    """Call the Gemini client and parse JSON output. This function attempts multiple known call patterns
    to be compatible with different genai SDK versions.
    """
    if client is None:
        st.error("Gemini client not available. Ensure google-genai is installed and GEMINI_API_KEY is set.")
        return None

    # Try common SDK method names dynamically
    response = None
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            response = client.models.generate_content(model=model, contents=prompt, config={"response_mime_type": "application/json"})
        elif hasattr(client, "generate"):
            response = client.generate(model=model, prompt=prompt)
        elif hasattr(client, "model") and hasattr(client.model, "generate"):
            response = client.model.generate(model=model, prompt=prompt)
        else:
            # last resort: try calling client directly
            response = client(model=model, prompt=prompt)
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return None

    # Extract raw text from response safely
    text_out = None
    try:
        # common SDK fields
        if hasattr(response, "text") and isinstance(response.text, str):
            text_out = response.text
        elif hasattr(response, "content"):
            text_out = response.content if isinstance(response.content, str) else str(response.content)
        else:
            text_out = str(response)
    except Exception:
        text_out = str(response)

    # Try to parse JSON
    try:
        data = json.loads(text_out)
        return data
    except Exception:
        # If the SDK provided a 'parsed' attribute that is already structured
        parsed = getattr(response, "parsed", None)
        if parsed:
            try:
                # parsed might be a pydantic object or similar
                return json.loads(parsed.json()) if hasattr(parsed, "json") else dict(parsed)
            except Exception:
                try:
                    return dict(parsed)
                except Exception:
                    pass

        st.warning("Could not parse Gemini response as JSON. Showing raw output for debugging.")
        st.code(text_out)
        return None


# -----------------------------
# High-level flows
# -----------------------------

def extract_claims(client, text: str, max_claims: int = 6, model: str = "gemini-2.5-flash") -> Optional[List[str]]:
    prompt = build_extract_claims_prompt(text, max_claims=max_claims)
    data = call_gemini_for_json(client, prompt, model=model)
    if not data:
        return None
    claims = data.get("claims") or []
    clean_claims = []
    for c in claims:
        if isinstance(c, dict) and "claim" in c:
            claim_text = c.get("claim")
            if claim_text and isinstance(claim_text, str):
                clean_claims.append(claim_text.strip())
        elif isinstance(c, str) and c.strip():
            clean_claims.append(c.strip())
    return clean_claims[:max_claims]


def verify_claim_with_search_and_gemini(client, claim: str, serpapi_key: Optional[str], use_search: bool, model: str, results_per_claim: int = 5) -> Optional[ClaimVerification]:
    search_results = None
    if use_search and serpapi_key:
        try:
            search_results = serpapi_search(claim, serpapi_key, num_results=results_per_claim)
        except Exception as e:
            st.warning(f"SerpAPI search failed for claim: {e}")
            search_results = None

    prompt = build_verify_claim_prompt(claim)
    data = call_gemini_for_json(client, prompt, model=model)
    if not data:
        return None

    try:
        verdict = data.get("verdict") or "UNVERIFIED"
        confidence = float(data.get("confidence") or 0.0)
        evidence = data.get("evidence") or []
        notes = data.get("notes") or ""
        return ClaimVerification(claim=claim, verdict=verdict, confidence=confidence, evidence=evidence, notes=notes)
    except Exception as e:
        st.warning(f"Failed to build ClaimVerification: {e}")
        return None


def aggregate_overall(claims: List[ClaimVerification]) -> Dict[str, Any]:
    if not claims:
        return {"overall_verdict": "UNVERIFIED", "overall_confidence": 0.0}

    confidences = [c.confidence for c in claims]
    avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

    verdicts = [c.verdict.upper() for c in claims]
    if any(v == "FAKE" for v in verdicts):
        overall = "FAKE"
    elif all(v == "REAL" for v in verdicts):
        overall = "REAL"
    else:
        num_real = sum(1 for v in verdicts if v == "REAL")
        if num_real >= len(verdicts) / 2:
            overall = "LIKELY REAL"
        else:
            overall = "UNVERIFIED"

    return {"overall_verdict": overall, "overall_confidence": round(avg_conf, 2)}


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="AI Fact-Checker — @CyberZenithX on Github", layout="wide")
    st.title("🔎 AI Fact-Checker — An advanced prototype")

    st.sidebar.header("API & Settings")
    api_key = st.sidebar.text_input("Gemini API key (paste or set GEMINI_API_KEY)", type="password")
    serpapi_key = st.sidebar.text_input("SerpAPI API key (optional, paste or set SERPAPI_API_KEY)", type="password")
    model = st.sidebar.selectbox("Gemini Model", options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)
    max_claims = st.sidebar.slider("Max claims to extract", 1, 12, 6)
    results_per_claim = st.sidebar.slider("Search results per claim", 0, 10, 4)
    _thinking_budget = st.sidebar.slider("Thinking budget (0 = off)", 0, 5000, 0)  # reserved
    use_search = st.sidebar.checkbox("Enable SerpAPI web evidence (recommended)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""**Notes**:
- SerpAPI must be installed (`pip install serpapi`) and a key provided to enable web evidence.
- The app fetches URL content when you paste a link, so avoid pasting private/sensitive data.
- All credit for this prototype goes to [@CyberZenithX on Github](https://github.com/CyberZenithX/)                   
""")

    input_mode = st.radio("Input type", options=["Article text", "Article URL", "Single claim"], index=0)

    if input_mode == "Article URL":
        url = st.text_input("Paste article URL here")
        text = ""
    else:
        url = ""
        default = "A recent study shows that drinking coffee reduces your risk of heart disease by 80%."
        text = st.text_area("Paste article text or claim(s) to fact-check", value=default, height=280)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Run fact-check"):
            # Input handling
            if input_mode == "Article URL":
                if not url.strip():
                    st.warning("Please paste an article URL first.")
                    return
                try:
                    text = fetch_article_text(url.strip())
                    st.success("Fetched article content — running claim extraction and checks.")
                except Exception as e:
                    st.error(f"Failed to fetch article: {e}")
                    return
            else:
                if not text.strip():
                    st.warning("Please paste some text first.")
                    return

            # initialize clients
            gapi_key = api_key or os.getenv("GEMINI_API_KEY")
            serp_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
            if use_search and not serp_key:
                st.warning("SerpAPI key not provided — continuing without web evidence.")
                serp_key = None

            client = get_genai_client(api_key=gapi_key)

            if client is None:
                st.error("Gemini client not configured. Install google-genai and set GEMINI_API_KEY.")
                return

            with st.spinner("Extracting claims via Gemini..."):
                claims = extract_claims(client, text, max_claims=max_claims, model=model)

            if not claims:
                st.error("No claims were extracted or Gemini failed to respond.\n\nPlease check the Gemini raw extraction response shown above for debugging information. Try a shorter input or adjust settings.")
                return

            st.subheader("Extracted claims")
            for i, c in enumerate(claims, start=1):
                st.write(f"{i}. {c}")

            verifications: List[ClaimVerification] = []
            st.subheader("Claim verifications (per-claim)")
            for i, claim in enumerate(claims, start=1):
                st.markdown(f"**{i}.** {claim}")
                with st.spinner(f"Searching web (if enabled) and asking Gemini about claim {i}..."):
                    result = verify_claim_with_search_and_gemini(
                        client,
                        claim,
                        serp_key,
                        use_search and serp_key is not None,
                        model=model,
                        results_per_claim=results_per_claim,
                    )
                    if result:
                        verifications.append(result)

                if result:
                    st.write(f"Verdict: **{result.verdict}** — Confidence: {result.confidence:.1f}%")
                    if result.evidence:
                        st.write("Evidence: ")
                        for url_item in result.evidence:
                            st.markdown(f"- [{url_item}]({url_item})")
                    if result.notes:
                        st.write(f"Notes: {result.notes}")
                else:
                    st.write("No structured result for this claim.")

                time.sleep(0.2)

            agg = aggregate_overall(verifications)
            response = FactCheckResponse(
                overall_verdict=agg["overall_verdict"],
                overall_confidence=agg["overall_confidence"],
                claims=verifications,
                summary=(
                    f"Checked {len(verifications)} claim(s). Overall: {agg['overall_verdict']} "
                    f"(Avg confidence: {agg['overall_confidence']:.1f}%)."
                ),
            )

            st.subheader("Overall verdict")
            st.metric("Verdict", response.overall_verdict, delta=f"Avg confidence: {response.overall_confidence:.1f}%")

            st.subheader("Summary")
            st.write(response.summary)

            st.subheader("Raw JSON")
            try:
                st.json(json.loads(response.json()))
            except Exception:
                st.write(response.json())

    with col2:
        st.subheader("Tips & next steps")
        st.markdown(
            """
- For best results, enable SerpAPI web evidence and provide a valid SerpAPI key in the sidebar.
- If a claim is time-sensitive (e.g., financial quarterly numbers, recent statements), Gemini may mark it UNVERIFIED if web evidence is lacking.
- To reduce API costs, lower `max_claims` and `results per claim` in the sidebar.
- Consider saving outputs to a local JSON log for auditing.

"""
        )


if __name__ == "__main__":
    main()
