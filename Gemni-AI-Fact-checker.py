"""
Streamlit + Gemini — AI-Powered Fake News / Fact-Checker (local)

How it works
- User pastes article text or claim(s) into the Streamlit UI.
- The app sends a structured prompt to Gemini (Gemini API) asking it to:
  1) extract up to N claims
  2) verify each claim (REAL / FAKE / UNVERIFIED)
  3) provide confidence (0-100), short evidence snippets and URLs (if available)
  4) return a short summary and an overall verdict
- The app requests Gemini to return **JSON** structured output that matches a Pydantic schema so the Streamlit UI can render it cleanly.

Notes / Caveats
- Gemini may not have live web access in your configuration. If it cannot confidently verify a claim, it should mark it UNVERIFIED. Always treat this prototype as an assistant — verify critical claims with independent sources.
- You need a Gemini API key. Set it as the environment variable `GEMINI_API_KEY` **or** paste it into the sidebar.

Dependencies
pip install streamlit google-genai pydantic

Run
export GEMINI_API_KEY="YOUR_KEY"
streamlit run streamlit_gemini_factchecker.py

"""

import os
import json
import textwrap
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel
from google import genai


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
# Gemini client helper
# -----------------------------

def get_genai_client(api_key: Optional[str] = None, vertexai: bool = False):
    """Return a configured genai.Client. If api_key provided, will use it explicitly."""
    if api_key:
        return genai.Client(api_key=api_key, vertexai=vertexai)
    # otherwise the client will pick up GEMINI_API_KEY from environment
    return genai.Client(vertexai=vertexai)


# -----------------------------
# Prompt / call
# -----------------------------

def build_factcheck_prompt(text: str, max_claims: int = 6) -> str:
    prompt = f"""
You are a careful fact-checker. The user will paste an article or text. Do the following, and return results in the structured JSON schema requested:
    
1) Extract up to {max_claims} distinct factual claims from the text. Keep each claim short (one sentence).
2) For each claim, verify its accuracy by conducting multiple searches across reliable sources. Report if verification is based on multiple confirmations.
3) For each claim include: the claim text, verdict (REAL/FAKE/UNVERIFIED), a numeric confidence (0-100), a short list of evidence citations or URLs (if no reliable URL, leave evidence empty), and a short note explaining the reasoning.
4) Provide an overall_verdict (REAL / LIKELY REAL / LIKELY FAKE / FAKE / UNVERIFIED), overall_confidence (0-100) and a short summary paragraph.
    
IMPORTANT:
- Do not invent evidence URLs. If you cannot find a source or are unsure, leave the evidence array empty and mark the verdict as UNVERIFIED.
- Before finalizing, cross-check each claim using at least two independent sources.
- Keep answers concise and factual.

Text to check:
"""
    prompt += "\n" + textwrap.dedent(text).strip()
    prompt += "\n\nReturn the result strictly as JSON following the schema: overall_verdict (string), overall_confidence (number), claims (array of objects with claim, verdict, confidence, evidence[], notes), summary (string)."
    return prompt


def gemini_fact_check(client: genai.Client, text: str, model: str = "gemini-2.5-flash", max_claims: int = 6, thinking_budget: int = 0) -> Optional[FactCheckResponse]:
    prompt = build_factcheck_prompt(text, max_claims=max_claims)

    # request structured JSON output using the schema
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FactCheckResponse,
                "thinking_config": {"thinking_budget": thinking_budget},
            },
        )

        # `response.parsed` should contain a FactCheckResponse instance if the model followed the schema
        parsed = getattr(response, "parsed", None)
        if parsed:
            # If the SDK gives us Pydantic objects, we can return them directly
            return parsed

        # fallback: try to parse the text as JSON
        text_out = response.text
        try:
            data = json.loads(text_out)
            return FactCheckResponse(**data)
        except Exception:
            st.warning("Could not parse Gemini's response as structured JSON. Showing raw output below.")
            st.code(text_out)
            return None

    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Gemini Fact-Checker", layout="wide")
    st.title("🔎 Gemini Fact-Checker — Streamlit prototype")

    st.sidebar.header("API & Settings")
    api_key = st.sidebar.text_input("Gemini API key (paste or set GEMINI_API_KEY)", type="password")
    model = st.sidebar.selectbox("Model", options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)
    max_claims = st.sidebar.slider("Max claims to extract", 1, 12, 6)
    thinking_budget = st.sidebar.slider("Thinking budget (0 = off, higher = more compute)", 0, 5000, 0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Notes**:\n- This is a prototype. Gemini may not have web access depending upon your environment.\n- Avoid pasting private/sensitive data.")

    input_mode = st.radio("Input type", options=["Article / long text", "Single claim"], index=0)

    default_demo = (
        "A recent study shows that drinking coffee reduces your risk of heart disease by 80%."
        if input_mode == "Single claim"
        else "\n".join([
            "Vaccine X caused an increase in illness in City Y, according to a local post.",
            "Company Z lost $5 billion last quarter due to product recalls.",
            "The governor announced an immediate halt to funding for school lunches."
        ])
    )

    text = st.text_area("Paste article text, tweet(s), or claim(s) to fact-check", value=default_demo, height=280)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Run fact-check"):
            if not text.strip():
                st.warning("Please paste some text first.")
            else:
                # initialize client
                if api_key:
                    client = get_genai_client(api_key=api_key)
                else:
                    client = get_genai_client()

                with st.spinner("Contacting Gemini and running checks — this can take a few seconds..."):
                    result = gemini_fact_check(client, text, model=model, max_claims=max_claims, thinking_budget=thinking_budget)

                if result is None:
                    st.info("No structured result was returned. Check the raw output for details.")
                else:
                    # display summary
                    st.subheader("Overall verdict")
                    st.metric("Verdict", result.overall_verdict, delta=f"Confidence: {result.overall_confidence:.1f}%")

                    st.subheader("Summary")
                    st.write(result.summary)

                    st.subheader("Claims checked")
                    for i, c in enumerate(result.claims, start=1):
                        with st.expander(f"{i}. {c.claim} — {c.verdict} ({c.confidence:.1f}%)"):
                            st.write(f"**Notes:** {c.notes or '—'}")
                            if c.evidence:
                                st.write("**Evidence / links:**")
                                for url in c.evidence:
                                    st.markdown(f"- [{url}]({url})")
                            else:
                                st.write("No evidence URLs provided by the model.")

                    st.subheader("Raw JSON")
                    st.json(json.loads(result.json()))

    with col2:
        st.subheader("Next steps")
        st.markdown(
            """
- Integrate Google Fact Check Tools API for stronger proof validation.
- Add search engine cross-referencing (e.g., SerpAPI, NewsGuard) for better sourcing.
- Save results to a database for auditing, analysis and learning.
- Provide export capabilities (CSV / PDF report).

"""
        )


if __name__ == "__main__":
    main()

