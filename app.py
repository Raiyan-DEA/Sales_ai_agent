# app.py
import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from closeio_api import Client
from openai import OpenAI
from langsmith import traceable

# ========= ENV & CONFIG =========
# Load /etc/salesai/.env if present (system-wide), otherwise ./.env
ENV_PATHS = [Path("/etc/salesai/.env"), Path(".env")]
for p in ENV_PATHS:
    if p.exists():
        load_dotenv(p.as_posix())
        break

# Optional LangSmith tracing (reads from env you created)
if os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
if os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOSE_API_KEY = os.getenv("CLOSE_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Fail fast if keys are missing
missing = [k for k, v in {"OPENAI_API_KEY": OPENAI_API_KEY, "CLOSE_API_KEY": CLOSE_API_KEY}.items() if not v]
if missing:
    st.error(f"Missing required env vars: {', '.join(missing)}. "
             f"Ensure they are set in /etc/salesai/.env and restart the service.")
    st.stop()

client = Client(CLOSE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="DEA Post-Call Follow-up Assistant", page_icon="🤖", layout="wide")

# ========= UTILITIES =========
@traceable(run_type="llm")
def gpt_call(system_prompt, user_prompt, temperature=0.9, max_tokens=2000):
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

@traceable()
def load_content_excel(file_path=None):
    """
    Load the content catalog from Excel.
    Default: "AI Agent Personalized followup.xlsx" next to this file.
    You can override by setting CONTENT_FILE in /etc/salesai/.env
    """
    excel_filename = os.getenv("CONTENT_FILE", "AI Agent Personalized followup.xlsx")
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / excel_filename
    if not excel_path.exists():
        st.error(f"Excel file not found at: {excel_path}")
        st.stop()
    df = pd.read_excel(excel_path)
    return df.to_dict(orient="records")

@traceable()
def fetch_paginated_activities(lead_id):
    pagination = True
    all_activities = []
    try:
        current_date = datetime.now()
        current_start_date = current_date - timedelta(days=180)
        first_iteration = True
        offset = 0
        pages = 0
        first_iteration_date = None

        while pagination:
            activity_params = {
                "lead_id": lead_id,
                "_skip": offset,
            }
            # Switch the direction after the first page
            if first_iteration:
                activity_params["date_created__gt"] = current_start_date
            else:
                activity_params["date_created__lt"] = current_start_date

            activity_response = client.get("activity", params=activity_params)
            data = activity_response.get("data", [])
            if not data:
                break

            all_activities.extend(data)

            if first_iteration and data:
                first_iteration_date = data[0]["activity_at"]

            last_activity_date = data[-1]["activity_at"]
            if last_activity_date == first_iteration_date:
                offset += len(data)
            else:
                offset = 0
                first_iteration_date = last_activity_date

            pages += 1
            current_start_date = last_activity_date
            first_iteration = False

            if not activity_response.get("has_more", False):
                break

    except Exception as e:
        print(f"Error on lead_id={lead_id}: {str(e)}")
        raise e

    return all_activities[:10]

@traceable(run_type="llm")
def analyze_call(transcript):
    system = "You are a professional sales AI that extracts structured call insights."
    user = f"""
    Extract JSON fields:
    - lead_type_guess: 'Full Transitioner' OR 'Upgrader' OR 'Switcher' OR 'Advanced'
    - topics: list of 3-6 short keywords
    - objections: list of 0-3
    - tone: one of ["Engaged","Skeptical","Neutral","Excited"]

    Transcript:
    {transcript}

    Return JSON only.
    """
    raw = gpt_call(system, user)
    try:
        return json.loads(re.search(r"\{{.*\}}", raw, re.S).group(0))
    except Exception:
        return {"lead_type_guess": "Unknown", "topics": [], "objections": [], "tone": "Neutral"}

@traceable(run_type="llm")
def gpt_select_content(transcript, crm_data, content_catalog):
    system = """You are a senior sales enablement AI for Data Engineer Academy.
    You will read a sales call transcript, the lead's CRM info, and a content catalog.
    Pick 3–4 pieces of content that will best nurture this lead.
    Output strictly JSON list of objects with fields: topic, reason, link."""
    user = f"""
    Sales call transcript:
    {transcript}

    Lead info (from Close CRM):
    {json.dumps(crm_data, indent=2)}

    Content catalog:
    {json.dumps(content_catalog, indent=2)}

    Respond in JSON only, selecting 3–4 items that fit the lead's current goals, stage, and objections.
    """
    resp = gpt_call(system, user, temperature=0.9, max_tokens=3000)
    try:
        return json.loads(re.search(r"\[.*\]", resp, re.S).group(0))
    except Exception:
        return []

@traceable(run_type="llm")
def gpt_write_email(crm_data, insights, selected_content):
    system = """You are a concise, friendly SDR email writer for Data Engineer Academy.
            [style rules as in your original]"""
    user = f"""
    Lead info:
    {json.dumps(crm_data, indent=2)}

    Call insights:
    {json.dumps(insights, indent=2)}

    Selected content for follow-up:
    {json.dumps(selected_content, indent=2)}

    Write a personalized 2–3 paragraph follow-up email.
    Mention the content naturally (no bullet lists).
    Start with a subject line like 'Subject: ...'
    End with a clear, friendly CTA.
    Keep it under 300 words.
    Keep it quirky and human-like.
    Make sure to mention lead names and specific topics from the call. Any email/name associated with data engineer academy should not be used as lead names. If no name is found, use "there".
    Explain some of the selected content and why it's relevant based on the pain points.
    Make sure to add valid links to the content.
    No need to add formal starts like 'hope this finds you well'.
    Do not provide similar kind of content pieces.
    """
    return gpt_call(system, user, temperature=0.9, max_tokens=1000)

# ========= UI =========
st.title("🤖 DEA Post-Call Nurture Assistant")

st.subheader("🎧 Fathom Recording or Transcript")
transcript = st.text_area("Paste transcript or Fathom call summary", height=200)

st.subheader("👤 Lead Info (Close CRM)")
lead_email = st.text_input("Lead Email (for Close CRM lookup)", placeholder="lead@example.com")

if st.button("🚀 Analyze & Generate Follow-up", use_container_width=True):
    content_catalog = load_content_excel()
    with st.spinner("Analyzing call..."):
        insights = analyze_call(transcript)
    st.success("✅ Call analysis complete")

    with st.expander("📊 Call Insights Summary", expanded=True):
        st.markdown(f"""
        **Lead Type Guess:** {insights.get('lead_type_guess', 'N/A')}  
        **Tone:** {insights.get('tone', 'N/A')}  
        **Topics Discussed:**  
        🔹 {' | '.join(insights.get('topics', []) or ['None'])}

        **Objections Raised:**  
        ⚠️ {' | '.join(insights.get('objections', []) or ['None'])}
        """)

    with st.spinner("Fetching CRM info from Close..."):
        try:
            leads = client.get('lead', params={"query": f"email:{lead_email}"})
            lead_id = leads["data"][0]["id"]
            crm_data = fetch_paginated_activities(lead_id)
        except Exception:
            crm_data = None

        if crm_data:
            st.success(f"✅ CRM data found for {lead_email}.")
        else:
            st.error(f"🚨 No Close CRM data found for {lead_email}.")

    with st.spinner("Selecting best nurturing content..."):
        selected = gpt_select_content(transcript, crm_data, content_catalog)
    st.success("✅ Best nurturing content selected.")

    with st.spinner("Composing personalized follow-up email..."):
        email_text = gpt_write_email(crm_data, insights, selected)
    st.success("Follow-up email generated successfully!")
    st.markdown("### Generated Follow-up Email")
    st.write(email_text)

