import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from app import load_data

df = load_data()

st.title("🤖 Ask the Data (beta)")
st.markdown(
    "Ask questions about solar generation, smoke events, and weather conditions. "
    "The chatbot retrieves relevant data from the KKP1 dataset to answer your questions."
)
st.info(
    "⚠️ **Note:** This chatbot has no memory between questions — each question is answered independently. "
    "For best results, include all relevant context in your question (e.g. 'What were the PM2.5 readings during smoke events in May 2023?'). "
    "Responses may also be cut off due to character limits."
)

from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient

HF_TOKEN = st.secrets["HF_TOKEN"]

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_hf_client():
    return InferenceClient(api_key=HF_TOKEN)

@st.cache_data
def build_documents(_df):
    df_day = _df.copy()
    df_day["date"] = df_day["dt"].dt.date
    daily = df_day.groupby("date").agg(
        avg_generation=("Volume", "mean"),
        max_generation=("Volume", "max"),
        avg_pm25=("pm25_mean", "mean"),
        avg_shortwave=("shortwave", "mean"),
        avg_cloud=("cloud_pct", "mean"),
    ).reset_index()

    docs = {}
    for _, row in daily.iterrows():
        smoke = "a smoke event was occurring" if row["avg_pm25"] > 50 else "no significant smoke"
        text = (
            f"On {row['date']}, average solar generation was {row['avg_generation']:.3f} MW "
            f"with a peak of {row['max_generation']:.3f} MW. "
            f"Average PM2.5 was {row['avg_pm25']:.1f} µg/m³ — {smoke}. "
            f"Average shortwave radiation was {row['avg_shortwave']:.1f} W/m² "
            f"and cloud coverage was {row['avg_cloud']:.1f}%."
        )
        docs[str(row["date"])] = text
    return docs

@st.cache_data
def build_embeddings(docs):
    embedder = load_embedder()
    return {k: embedder.encode(v, convert_to_tensor=True) for k, v in docs.items()}

def retrieve_context(query, doc_embeddings, documents, top_k=5):
    embedder = load_embedder()
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    import re
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    
    year_match = re.search(r'\b(20\d{2})\b', query.lower())
    month_match = next((m for m in months if m in query.lower()), None)
    
    # If asking about worst/highest/most smoke, sort by PM2.5 directly
    superlative_terms = ["worst", "highest", "most", "peak", "maximum", "max", "bad"]
    smoke_terms = ["smoke", "pm2.5", "pm25", "air quality"]
    
    if any(t in query.lower() for t in superlative_terms) and \
    any(t in query.lower() for t in smoke_terms):
        
        pm25_scores = {}
        for k, v in documents.items():
            if k == "app_usage":
                continue
            if year_match and year_match.group(1) not in k:
                continue
            if month_match and f"-{months[month_match]}-" not in k:
                continue
            match = re.search(r'PM2\.5 was ([\d.]+)', v)
            if match:
                pm25_scores[k] = float(match.group(1))
        
        if pm25_scores:
            top_keys = sorted(pm25_scores, key=pm25_scores.get, reverse=True)[:top_k]
            return "\n\n".join(documents[k] for k in top_keys)
    
    # Date filter
    if year_match or month_match:
        year = year_match.group(1) if year_match else None
        month = months[month_match] if month_match else None
        
        filtered_keys = []
        for k in doc_embeddings:
            if k == "app_usage":
                continue
            if year and year not in k:
                continue
            if month and f"-{month}-" not in k:
                continue
            filtered_keys.append(k)
        
        if filtered_keys:
            scores = {k: util.pytorch_cos_sim(query_embedding, doc_embeddings[k]).item()
                    for k in filtered_keys}
            top_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
            return "\n\n".join(documents[k] for k in top_keys)
    
    # Fall back to full semantic search
    scores = {k: util.pytorch_cos_sim(query_embedding, doc_embeddings[k]).item()
            for k in doc_embeddings}
    top_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return "\n\n".join(documents[k] for k in top_keys)

def query_hf(prompt):
    client = load_hf_client()
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a solar energy analyst. Answer questions clearly and concisely based on the data provided."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Build docs and embeddings
with st.spinner("Loading data..."):
    documents = build_documents(df)
    doc_embeddings = build_embeddings(documents)

# App usage guide
app_guide = {
    "app_usage": (
        "This app has 8 pages. The Map page shows NASA satellite imagery of smoke events. "
        "The Compare to Client page fills missing generation data using KKP1 as a proxy. "
        "The Prediction Check page lets you look up model predictions vs actual generation for any date. "
        "The Paradox page shows how smoke events reduce daily generation but surrounding weeks "
        "often have higher generation due to dry sunny conditions. "
        "The Hourly Smoke Analysis page shows PM2.5 and generation hour by hour around smoke events. "
        "The XAI page shows feature importance and partial dependence plots. "
        "The Future Work page describes planned improvements using Copernicus data."
    )
}
documents.update(app_guide)
app_embedder = load_embedder()
doc_embeddings["app_usage"] = app_embedder.encode(
    app_guide["app_usage"], convert_to_tensor=True
)

query = st.text_input(
    "Ask a question about the solar data or how to use this app",
    placeholder="e.g. What was generation like during smoke events in 2023?"
)

if query:
    with st.spinner("Retrieving relevant data and generating answer..."):
        context = retrieve_context(query, doc_embeddings, documents, top_k=10)
        prompt = (
            f"Use the data below to answer the question clearly and concisely.\n\n"
            f"Data:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        answer = query_hf(prompt)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("View retrieved data chunks"):
        st.text(context)