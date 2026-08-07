import os
import pickle
import numpy as np
import pandas as pd
import faiss
from pypdf import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

XLSX_PATH = "Cricket_Players_5000_Final.xlsx"
PDF_PATH = "Merged_Cricket_Reference_Collection.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"
def row_to_text(row):
    lines = [
        f"Player: {row['Player Name']}" + (f" (aka {row['Nicknames']})" if row['Nicknames'] else ""),
        f"Country: {row['Country']} | Role: {row['Role']} | Tier: {row['Tier']}",
        f"Batting Style: {row['Batting Style']} | Bowling Style: {row['Bowling Style']}",
        f"Career Span: {row['Career Span']} | Teams: {row['Teams']}",
        f"Height: {row['Height (cm)']} cm | Date of Birth: {row['Date of Birth']}",
        f"Base Price: {row['Base Price (Cr)']} Cr | Points: {row['Points']}",
        f"Overall — Matches: {row['Total Mat']}, Runs: {row['Total Runs']}, Wickets: {row['Total Wkts']}, "
        f"Average: {row['Overall Avg']}, Stumpings: {row['Stumpings']}, 4s: {row['4s']}, 6s: {row['6s']}",
        f"Test — Matches: {row['Test Mat']}, Runs: {row['Test Runs']}, Avg: {row['Test Avg']}, Wkts: {row['Test Wkts']}",
        f"ODI — Matches: {row['ODI Mat']}, Runs: {row['ODI Runs']}, Avg: {row['ODI Avg']}, Wkts: {row['ODI Wkts']}",
        f"T20I — Matches: {row['T20I Mat']}, Runs: {row['T20I Runs']}, Avg: {row['T20I Avg']}, Wkts: {row['T20I Wkts']}",
        f"IPL — Matches: {row['IPL Mat']}, Runs: {row['IPL Runs']}, Avg: {row['IPL Avg']}, Wkts: {row['IPL Wkts']}",
    ]
    if row.get('Awards'): lines.append(f"Awards: {row['Awards']}")
    if row.get('Records'): lines.append(f"Records: {row['Records']}")
    if row.get('Aliases'): lines.append(f"Also known as: {row['Aliases']}")
    return "\n".join(lines)

def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            for sep in ["\n\n", ". ", "\n"]:
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx > start + chunk_size * 0.4:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks

def build_index():
    print("⚡ Processing Excel dataset...")
    df = pd.read_excel(XLSX_PATH).fillna("")
    player_docs = [{"text": row_to_text(r), "source": XLSX_PATH, "type": "player_profile", "title": r["Player Name"]} for _, r in df.iterrows()]

    print("⚡ Extracting PDF documents...")
    reader = PdfReader(PDF_PATH)
    pdf_docs = []
    for i, page in enumerate(tqdm(reader.pages, desc="Extracting pages")):
        text = page.extract_text() or ""
        if text.strip():
            for c in chunk_text(text):
                pdf_docs.append({"text": c, "source": PDF_PATH, "type": "reference", "title": f"PDF Page {i+1}"})

    all_docs = player_docs + pdf_docs
    print(f"Total chunks created: {len(all_docs)}")

    print("⚡ Generating embeddings...")
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"] for d in all_docs]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print("⚡ Saving FAISS index locally...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, "cricket_index.faiss")
    with open("cricket_docs.pkl", "wb") as f:
        pickle.dump(all_docs, f)

    print("✅ Ingestion completed successfully!")

if __name__ == "__main__":
    build_index()