import json
from pathlib import Path

import faiss
import torch
from transformers import AutoTokenizer, AutoModel


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector_db" / "medcpt.index"
DOCS_PATH = BASE_DIR / "vector_db" / "medcpt_docs.json"

MODEL_NAME = "ncbi/MedCPT-Query-Encoder"


class MedCPTRetriever:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading MedCPT Query Encoder on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if self.index.ntotal != len(self.documents):
            raise ValueError("FAISS index and document count do not match.")

        print(f"Loaded {self.index.ntotal} documents.")

    def encode_query(self, query):
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding.cpu().numpy().astype("float32")

    def search(self, query, top_k=5):
        query_vector = self.encode_query(query)

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            document = self.documents[int(idx)].copy()
            document["score"] = float(score)

            results.append(document)

        return results


if __name__ == "__main__":
    retriever = MedCPTRetriever()

    query = input("\nEnter a medical question: ")

    results = retriever.search(query, top_k=5)

    for i, result in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Source: {result['source']}")
        print(f"Focus Area: {result['focus_area']}")