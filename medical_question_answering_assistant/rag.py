import json
import os
import sys
from pathlib import Path
from time import time

from dotenv import load_dotenv
from google import genai

# 優先載入 .env 環境變數，確保後續讀取 GEMINI_API_KEY 與 GEMINI_MODEL 時已就緒
load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from medcpt_retrieval import MedCPTRetriever

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# 初始化 Gemini Client（設定 30 秒請求逾時，避免網路或伺服器延遲導致程式無限期卡死）
client = genai.Client(http_options={"timeout": 30000})

# 初始化 MedCPT + FAISS 檢索器
retriever = MedCPTRetriever()


def search(query, top_k=5):
    results = retriever.search(query=query, top_k=top_k)
    return results


prompt_template = """
You're a medical question answering assistant. Answer the QUESTION based on the CONTEXT from our question and answer database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()


entry_template = """
question: {question}
answer: {answer}
source: {source}
focus_area: {focus_area}
""".strip()


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model=DEFAULT_GEMINI_MODEL):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    answer = response.text or ""

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    completion_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    total_tokens = (
        (getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens))
        if usage
        else 0
    )

    token_stats = {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }

    return answer, token_stats


evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer, model=DEFAULT_GEMINI_MODEL):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model=model)

    # 防呆清理：若模型仍回傳 ```json ... ``` 區塊，先去除外框再解析 JSON
    cleaned_eval = evaluation.strip()
    if cleaned_eval.startswith("```"):
        lines = cleaned_eval.splitlines()
        if len(lines) >= 3:
            cleaned_eval = "\n".join(lines[1:-1]).strip()

    try:
        json_eval = json.loads(cleaned_eval)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_gemini_cost(model, tokens):
    # 依專案規範，不假設或硬編碼 Gemini API 計費單價，預設回傳 0.0
    return 0.0


def rag(query, model=DEFAULT_GEMINI_MODEL):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer, model=model)

    t1 = time()
    took = t1 - t0

    gemini_cost_rag = calculate_gemini_cost(model, token_stats)
    gemini_cost_eval = calculate_gemini_cost(model, rel_token_stats)

    gemini_cost = gemini_cost_rag + gemini_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "gemini_cost": gemini_cost,
    }

    return answer_data