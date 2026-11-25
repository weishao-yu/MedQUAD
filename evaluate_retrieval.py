import sys
from pathlib import Path
from time import time

import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "medical_question_answering_assistant"))

import minsearch
from medcpt_retrieval import MedCPTRetriever


def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1.0 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function, desc="Evaluating"):
    relevance_total = []
    for q in tqdm(ground_truth, desc=desc):
        doc_id = int(q["id"])
        results = search_function(q["question"])
        relevance = [int(d["id"]) == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


def run_benchmark(ground_truth, docs_subset, faiss_sub_index, retriever, label):
    # 1. Minsearch 無權重 (Basic)
    ms_index = minsearch.Index(
        text_fields=["question", "answer", "source", "focus_area"],
        keyword_fields=["id"],
    )
    ms_docs = [
        {
            "id": int(d["id"]),
            "question": str(d["question"]),
            "answer": str(d["answer"]),
            "source": str(d["source"]),
            "focus_area": str(d["focus_area"]),
        }
        for d in docs_subset
    ]
    ms_index.fit(ms_docs)

    ms_basic = evaluate(
        ground_truth,
        lambda q: ms_index.search(query=q, filter_dict={}, boost_dict={}, num_results=10),
        desc=f"[{label}] Minsearch Basic (Top-10)",
    )

    # 2. Minsearch 調校權重 (Improved)
    boost = {
        "question": 0.3417604466420481,
        "answer": 1.8419912735713542,
        "source": 0.25631365201530965,
        "focus_area": 0.22228140744464786,
    }
    ms_tuned = evaluate(
        ground_truth,
        lambda q: ms_index.search(query=q, filter_dict={}, boost_dict=boost, num_results=10),
        desc=f"[{label}] Minsearch Tuned (Top-10)",
    )

    # 3. MedCPT + FAISS 檢索函式（支援子索引與全量索引）
    def medcpt_search(query, top_k=10):
        q_vec = retriever.encode_query(query)
        scores, indices = faiss_sub_index.search(q_vec, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(docs_subset):
                results.append(docs_subset[idx])
        return results

    t0 = time()
    medcpt_top10 = evaluate(
        ground_truth,
        lambda q: medcpt_search(q, top_k=10),
        desc=f"[{label}] MedCPT + FAISS (Top-10)",
    )
    t1 = time()

    medcpt_top5 = evaluate(
        ground_truth,
        lambda q: medcpt_search(q, top_k=5),
        desc=f"[{label}] MedCPT + FAISS (Top-5)",
    )

    return {
        "label": label,
        "num_docs": len(docs_subset),
        "num_queries": len(ground_truth),
        "ms_basic": ms_basic,
        "ms_tuned": ms_tuned,
        "medcpt_top10": medcpt_top10,
        "medcpt_top5": medcpt_top5,
        "medcpt_time": t1 - t0,
    }


def main():
    gt_path = BASE_DIR / "data" / "ground-truth-retrieval.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"找不到評估資料集: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    retriever = MedCPTRetriever()

    # --- 實驗 A：前 1,028 筆資料（對齊原專案 README 實驗基準） ---
    subset_size = min(1028, len(retriever.documents))
    docs_1028 = retriever.documents[:subset_size]
    valid_ids_1028 = {int(d["id"]) for d in docs_1028}
    gt_1028 = df_gt[df_gt["id"].astype(int).isin(valid_ids_1028)].to_dict(orient="records")

    # 從現有 FAISS 索引直接抽取出前 1,028 筆向量，免重建
    vectors_1028 = retriever.index.reconstruct_n(0, subset_size)
    index_1028 = faiss.IndexFlatIP(vectors_1028.shape[1])
    index_1028.add(np.ascontiguousarray(vectors_1028, dtype="float32"))

    res_1028 = run_benchmark(
        gt_1028, docs_1028, index_1028, retriever, label="前 1,028 筆子集"
    )

    # --- 實驗 B：全量 16,407 筆資料（真實完整知識庫規模） ---
    valid_ids_all = {int(d["id"]) for d in retriever.documents}
    gt_all = df_gt[df_gt["id"].astype(int).isin(valid_ids_all)].to_dict(orient="records")

    res_all = run_benchmark(
        gt_all, retriever.documents, retriever.index, retriever, label="全量 16,407 筆"
    )

    for res in (res_1028, res_all):
        print("\n" + "=" * 68)
        print(f"檢索效能評估結果【{res['label']}】（文件數: {res['num_docs']} | 測試題數: {res['num_queries']}）")
        print("=" * 68)
        print(f"Minsearch Basic (Top-10) -> Hit Rate: {res['ms_basic']['hit_rate']:.4f} | MRR: {res['ms_basic']['mrr']:.4f}")
        print(f"Minsearch Tuned (Top-10) -> Hit Rate: {res['ms_tuned']['hit_rate']:.4f} | MRR: {res['ms_tuned']['mrr']:.4f}")
        print(f"MedCPT + FAISS  (Top-10) -> Hit Rate: {res['medcpt_top10']['hit_rate']:.4f} | MRR: {res['medcpt_top10']['mrr']:.4f} (耗時 {res['medcpt_time']:.1f}s)")
        print(f"MedCPT + FAISS  (Top-5)  -> Hit Rate: {res['medcpt_top5']['hit_rate']:.4f} | MRR: {res['medcpt_top5']['mrr']:.4f}")
        print("=" * 68)


if __name__ == "__main__":
    main()