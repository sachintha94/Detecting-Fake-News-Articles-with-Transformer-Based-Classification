import os
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import torch
from typing import Dict, Any, List, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------- UI SETUP --------------------
st.set_page_config(
    page_title="Fake News Detector (DistilBERT)",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.caption("DistilBERT fine-tuned on BuzzFeed + PolitiFact | 0 = real, 1 = fake")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Inference mode",
        ["Local model folder", "Remote Lambda endpoint"],
        help="Local: load the model from a folder on this machine. Remote: call your AWS Lambda /predict."
    )

    threshold = st.slider("Decision threshold for FAKE (class=1)", 0.0, 1.0, 0.50, 0.01)

    if mode == "Local model folder":
        default_path = os.path.join(os.path.dirname(__file__), "news_model")  # Auto-resolves to ./news_model
        model_path = st.text_input(
            "Model folder path",
            value=default_path,
            help="Folder containing model.safetensors, config.json, tokenizer.json, etc."
        )
        
    else:
        lambda_url = st.text_input(
            "Lambda Function URL (e.g., https://abc123.lambda-url.xx.on.aws)",
            value="",
            help="Your API endpoint root (no trailing slash). App will POST to /predict"
        )

    st.markdown("---")
    st.write("Tips:")
    st.write("• Use the same pre-processing format as training:")
    st.code('Title: <title> [SEP] Text: <text>', language="text")


# -------------------- HELPERS --------------------
def clean(s: str) -> str:
    return str(s).replace("\n", " ").strip()

def format_input(title: str, text: str) -> str:
    return f"Title: {clean(title)} [SEP] Text: {clean(text)}"


@st.cache_resource(show_spinner=False)
def load_local_model(path: str):
    """Cache the tokenizer+model so they load only once."""
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device


def local_predict(tokenizer, model, device, title: str, text: str, max_len: int = 512) -> Dict[str, Any]:
    input_text = format_input(title, text)
    enc = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()  # [p_real, p_fake]
    return {"p_real": float(probs[0]), "p_fake": float(probs[1])}


def remote_predict(api_base: str, title: str, text: str, thr: float) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/predict"
    payload = {"title": title, "text": text, "threshold": thr}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def present_result(p_fake: float, p_real: float, thr: float) -> Tuple[str, str]:
    label_id = 1 if p_fake >= thr else 0
    label_name = "fake (1)" if label_id == 1 else "real (0)"
    badge = "🟥 FAKE" if label_id == 1 else "🟩 REAL"
    return label_name, badge


# -------------------- SINGLE PREDICTION UI --------------------
st.subheader("🔎 Try a single prediction")

c1, c2 = st.columns(2)
with c1:
    title_in = st.text_input("Title", placeholder="Enter headline…")
with c2:
    st.write("")  # spacing

text_in = st.text_area("Text", height=180, placeholder="Paste article text…")

if st.button("Predict", type="primary"):
    if mode == "Local model folder":
        if not os.path.isdir(model_path):
            st.error("Model folder not found. Check the path in the sidebar.")
        else:
            with st.spinner("Loading model & predicting…"):
                tok, mdl, dev = load_local_model(model_path)
                out = local_predict(tok, mdl, dev, title_in, text_in, max_len=512)
    else:
        if not lambda_url:
            st.error("Please provide your Lambda Function URL in the sidebar.")
        else:
            with st.spinner("Calling remote endpoint…"):
                out = remote_predict(lambda_url, title_in, text_in, threshold)

    if out:
        if "p_fake" in out and "p_real" in out:
            # unify shape in case of remote
            p_fake = float(out["p_fake"])
            p_real = float(out["p_real"])
        else:
            # some remote implementations return only label; try to infer
            st.warning("Response did not include probabilities; showing raw JSON.")
            st.json(out)
            st.stop()

        label_name, badge = present_result(p_fake, p_real, threshold)

        st.success(f"Prediction: **{badge}**  \n"
                   f"p(fake): **{p_fake:.3f}**,  p(real): **{p_real:.3f}**  \n"
                   f"(threshold = {threshold:.2f})")

        st.progress(min(max(p_fake, 0.0), 1.0), text=f"Probability fake: {p_fake:.3f}")
        st.progress(min(max(p_real, 0.0), 1.0), text=f"Probability real: {p_real:.3f}")

        with st.expander("Show formatted input (as seen by the model)"):
            st.code(format_input(title_in, text_in), language="text")


# -------------------- BATCH PREDICTION UI --------------------
# st.subheader("📦 Batch prediction (CSV)")
# st.caption("Upload a CSV with columns: **title**, **text**.")

# csv_file = st.file_uploader("Upload CSV", type=["csv"])

# if csv_file is not None:
#     try:
#         df_in = pd.read_csv(csv_file)
#         if not {"title", "text"}.issubset(df_in.columns):
#             st.error("CSV must have columns: title, text")
#         else:
#             do_batch = st.button("Run batch prediction")
#             if do_batch:
#                 rows = []
#                 if mode == "Local model folder":
#                     if not os.path.isdir(model_path):
#                         st.error("Model folder not found. Check the path in the sidebar.")
#                         st.stop()
#                     tok, mdl, dev = load_local_model(model_path)

#                 progress = st.progress(0)
#                 t0 = time.time()

#                 for i, row in df_in.iterrows():
#                     title = str(row["title"])
#                     text  = str(row["text"])
#                     if mode == "Local model folder":
#                         out = local_predict(tok, mdl, dev, title, text, max_len=512)
#                     else:
#                         out = remote_predict(lambda_url, title, text, threshold)
#                     p_fake = float(out["p_fake"])
#                     p_real = float(out["p_real"])
#                     label_name, _ = present_result(p_fake, p_real, threshold)
#                     rows.append({
#                         "title": title,
#                         "text": text,
#                         "p_fake": p_fake,
#                         "p_real": p_real,
#                         "pred_label": label_name
#                     })
#                     if (i + 1) % max(1, len(df_in)//100) == 0:
#                         progress.progress((i + 1) / len(df_in))

#                 dt = time.time() - t0
#                 st.success(f"Done in {dt:.1f}s")

#                 df_out = pd.DataFrame(rows)
#                 st.dataframe(df_out.head(50), use_container_width=True)
#                 st.download_button(
#                     "Download results (CSV)",
#                     data=df_out.to_csv(index=False).encode("utf-8"),
#                     file_name="predicted_results.csv",
#                     mime="text/csv"
#                 )
#     except Exception as e:
#         st.exception(e)


# -------------------- FOOTER --------------------
# st.markdown("---")
# st.caption(
#     "Tip: for best results, keep the input format consistent with training: "
#     "`Title + Text, max length abou 300 words."
# )
