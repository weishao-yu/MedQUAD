import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "medquad.csv"
OUTPUT_DIR = BASE_DIR / "vector_db"

MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
BATCH_SIZE = 32
MAX_LENGTH = 512
CHECKPOINT_EVERY = 1600  # 每 1600 筆（50 個 batch）自動存一次進度


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(CSV_PATH)

    required_columns = ["id", "question", "answer", "source", "focus_area"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    df = df.dropna(subset=["question", "answer"]).copy()

    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ].reset_index(drop=True)

    total_docs = len(df)
    print(f"Total documents to embed: {total_docs}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_emb_path = OUTPUT_DIR / "checkpoint_embeddings.npy"
    ckpt_meta_path = OUTPUT_DIR / "checkpoint_meta.json"

    start_idx = 0
    all_embeddings = []

    if ckpt_emb_path.exists() and ckpt_meta_path.exists():
        with open(ckpt_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("total_docs") == total_docs and meta.get("completed", 0) > 0:
            saved_emb = np.load(ckpt_emb_path)
            start_idx = int(meta["completed"])
            all_embeddings.append(saved_emb)
            print(f"偵測到檢查點，從第 {start_idx}/{total_docs} 筆繼續編碼...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    for start in tqdm(range(start_idx, total_docs, BATCH_SIZE), desc="Embedding MedQuAD"):
        batch = df.iloc[start:start + BATCH_SIZE]

        titles = batch["question"].tolist()
        abstracts = batch["answer"].tolist()

        inputs = tokenizer(
            titles,
            abstracts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings = embeddings.cpu().numpy().astype("float32")
        all_embeddings.append(embeddings)

        completed = min(start + BATCH_SIZE, total_docs)
        if completed % CHECKPOINT_EVERY == 0 and completed < total_docs:
            merged_so_far = np.vstack(all_embeddings).astype("float32")
            np.save(ckpt_emb_path, merged_so_far)
            with open(ckpt_meta_path, "w", encoding="utf-8") as f:
                json.dump({"total_docs": total_docs, "completed": completed}, f)

    embeddings = np.vstack(all_embeddings).astype("float32")
    dimension = embeddings.shape[1]

    # MedCPT uses inner product similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = OUTPUT_DIR / "medcpt.index"
    docs_path = OUTPUT_DIR / "medcpt_docs.json"

    faiss.write_index(index, str(index_path))

    documents = df[
        ["id", "question", "answer", "source", "focus_area"]
    ].to_dict(orient="records")

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2, default=str)

    # 全部完成後清除暫存檢查點檔案
    if ckpt_emb_path.exists():
        ckpt_emb_path.unlink()
    if ckpt_meta_path.exists():
        ckpt_meta_path.unlink()

    print("\nIndex creation completed!")
    print(f"Documents: {index.ntotal}")
    print(f"Vector dimension: {dimension}")
    print(f"Index saved to: {index_path}")
    print(f"Documents saved to: {docs_path}")


if __name__ == "__main__":
    main()