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

# =========================
# 🔌 UTILITIES
# =========================
@traceable(run_type="llm", project_name="dea_post_call_assistant")
def gpt_call(system_prompt, user_prompt, temperature=0.9, max_tokens=2000):
    """Use OpenAI library to call GPT."""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

@traceable(project_name="dea_post_call_assistant")
def load_content_excel(file_path=None):
    """Read Excel directly from a fixed local path instead of upload."""
    # 👇 Set your local Excel file location here
    excel_filename = "AI Agent Personalized followup.xlsx"  # 👈 your file name
    excel_path = os.path.join(os.path.dirname(__file__), excel_filename)
    if not os.path.exists(excel_path):
        st.error(f"Excel file not found at {excel_path}")
        st.stop()
    df = pd.read_excel(excel_path)
    return df.to_dict(orient="records")

@traceable(project_name="dea_post_call_assistant")
def fetch_paginated_activities(lead_id):
    pagination = True
    all_activities = []
    try:
        current_date = datetime.now()
        current_start_date = current_date - timedelta(days=180)
        first_iteration = True  # Track if it's the first iteration
        offset = 0              # Track offset per lead
        pages = 0               # Optional: track number of pages fetched
        first_iteration_date = None

        while pagination:
            if first_iteration:
                activity_params = {
                    "lead_id": lead_id,
                    "date_created__gt": current_start_date,
                    "_skip": offset,
                }
            else:
                activity_params = {
                    "lead_id": lead_id,
                    "date_created__lt": current_start_date,
                    "_skip": offset,
                }

            activity_response = client.get("activity", params=activity_params)
            data = activity_response.get("data", [])

            if not data:
                break
            all_activities.extend(data)

            # Save first page date to compare
            if first_iteration and data:
                first_iteration_date = data[0]["activity_at"]

            # Determine if we should continue skipping or reset
            last_activity_date = data[-1]["activity_at"]
            if last_activity_date == first_iteration_date:
                # same date, continue skipping
                offset += len(data)
            else:
                # new date, reset skip
                offset = 0
                first_iteration_date = last_activity_date

            pages += 1
            current_start_date = last_activity_date
            first_iteration = False

            # Stop if API says no more
            if not activity_response.get("has_more", False):
                break

        # print(f"✅ Lead {lead_id}: fetched {len(all_activities)} activities in {pages} pages")

    except Exception as e:
        print(f"Error on lead_id={lead_id}: {str(e)}")
        raise e

    return all_activities[:10]


@traceable(run_type="llm", project_name="dea_post_call_assistant")
def analyze_call(transcript):
    """Extract insights from call transcript."""
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
        return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        return {
            "lead_type_guess": "Unknown",
            "topics": [],
            "objections": [],
            "tone": "Neutral"
        }

@traceable(run_type="llm", project_name="dea_post_call_assistant")
def gpt_select_content(transcript, crm_data, content_catalog):
    """Let GPT choose the best 3–4 pieces of content."""
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

@traceable(run_type="llm", project_name="dea_post_call_assistant")
def gpt_write_email(insights, selected_content,lead_email):
    """Compose personalized follow-up email."""
    system = """You are a concise, friendly SDR email writer for Data Engineer Academy.
            As an AI writing assistant, to ensure your output does not exhibit typical AI characteristics and feels authentically human, you must avoid certain patterns based on analysis of AI-generated text and my specific instructions. 
            Specifically, do not default to a generic, impersonal, or overly formal tone that lacks personal voice, anecdotes, or genuine emotional depth, and avoid presenting arguments in an overly balanced, formulaic structure without conveying a distinct perspective or emphasis. 
            Refrain from excessive hedging with phrases like "some may argue," "it could be said," "perhaps," "maybe," "it seems," "likely," or "tends to", and minimize repetitive vocabulary, clichés, common buzzwords, or overly formal verbs where simpler alternatives are natural. 
            Vary sentence structure and length to avoid a monotonous rhythm, consciously mixing shorter sentences with longer, more complex ones, as AI often exhibits uniformity in sentence length. Use diverse and natural transitional phrases, avoiding over-reliance on common connectors like "Moreover," "Furthermore," or "Thus," and do not use excessive signposting such as stating "In conclusion" or "To sum up" explicitly, especially in shorter texts. 
            Do not aim for perfect grammar or spelling to the extent that it sounds unnatural; incorporating minor, context-appropriate variations like contractions or correctly used common idioms can enhance authenticity, as AI often produces grammatically flawless text that can feel too perfect. 
            Avoid overly detailed or unnecessary definitional passages. Strive to include specific, concrete details or examples rather than remaining consistently generic or surface-level, as AI text can lack depth. Do not overuse adverbs, particularly those ending in "-ly". 
            Explicitly, you must never use em dashes (—). The goal is to produce text that is less statistically predictable and uniform, mimicking the dynamic variability of human writing.


            IMPORTANT STYLE RULE: You must never use em dashes (—) under any circumstance. 
            They are strictly forbidden. 
            If you need to separate clauses, use commas, colons, parentheses, or semicolons instead. 
            All em dashes must be removed and replaced before returning the final output. 
            2. Before completing your output, do a final scan for em dashes. If any are detected, rewrite those sentences immediately using approved punctuation. 
            3. If any em dashes are present in the final output, discard and rewrite that section before showing it to the user."""
    
    user = f"""
    Lead Name: {lead_email}

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
    No need to add of formal start like hope this message finds you well kind of things.
    Do not provide similar kind of content pieces.
    """
    return gpt_call(system, user, temperature=0.9, max_tokens=1000)
    
# =========================
# 🖥️ STREAMLIT UI
# =========================
st.title("🤖 DEA Post-Call Nurture Assistant")

st.subheader("🎧 Fathom Recording or Transcript")
transcript = st.text_area("Paste transcript or Fathom call summary", height=200)

st.subheader("👤 Lead Name")
lead_email = st.text_input("Lead Name", placeholder="Ninad Magdum")


from langsmith import traceable

@traceable(name="DEA Follow-up Run", run_type="chain", project_name="dea_post_call_assistant")
def run_followup_workflow(transcript, lead_email):
    """Run the full analyze → select → write flow as one LangSmith run."""
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

    crm_data = None

    with st.spinner("Selecting best nurturing content..."):
        selected = gpt_select_content(transcript, crm_data, content_catalog)
    st.success("✅ Best nurturing content selected.")

    with st.spinner("Composing personalized follow-up email..."):
        email_text = gpt_write_email(insights, selected, lead_email)
    st.success("Follow-up email generated successfully!")

    st.markdown("### Generated Follow-up Email")
    st.write(email_text)

    return {
        "insights": insights,
        "selected_content": selected,
        "email_text": email_text
    }

if st.button("🚀 Analyze & Generate Follow-up", use_container_width=True):
    run_followup_workflow(transcript, lead_email)
