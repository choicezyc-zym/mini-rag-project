from pathlib import Path
from datetime import datetime
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_pickle

def retrieve_top_k(
    query: str,
    model,
    chunk_embeddings,
    chunks,
    top_k: int = 3,
    min_score: float = 0.15,
    relative_threshold: float = 0.55
):
    """
    检索最相关的 top-k chunks，并过滤掉弱相关内容。

    min_score:
        最低相似度阈值，低于这个分数的 chunk 不要。

    relative_threshold:
        相对阈值，要求 chunk 的分数不能比第一名低太多。
        例如 best_score * 0.55。
    """

    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    best_score = scores[ranked_indices[0]]
    dynamic_threshold = max(min_score, best_score * relative_threshold)

    results = []
    for idx in ranked_indices:
        if scores[idx] >= dynamic_threshold:
            results.append((chunks[idx], scores[idx]))

    return results

def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    """
    使用 Ollama 本地大模型，根据检索到的 chunks 生成最终答案，
    并在答案后面附上参考来源。
    """

    context_lines = []
    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        context_lines.append(f"资料 {i}：\n{chunk}")

    context = "\n\n".join(context_lines)

    prompt = f"""
你是一个严谨的 RAG 问答助手。

请你只根据下面提供的资料回答用户问题。
如果资料中没有答案，请直接说：资料中没有找到相关信息。
不要编造，不要使用资料之外的知识。

【资料】
{context}

【用户问题】
{query}

【回答要求】
1. 用中文回答。
2. 回答要清楚、简洁。
3. 只基于资料内容回答。
4. 如果资料不足，要明确说明资料不足。
5. 不要在答案中重复输出完整资料原文。

【最终答案】
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
    )

    response.raise_for_status()
    result = response.json()

    llm_answer = result["response"].strip()

    source_lines = []
    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        source_lines.append(
            f"[{i}] score={score:.4f}\n{chunk}"
        )

    sources = "\n\n".join(source_lines)

    final_answer = (
        f"{llm_answer}\n\n"
        f"参考来源：\n"
        f"{sources}"
    )

    return final_answer


def save_answer(answer: str, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(answer)

def save_history(query: str, answer: str, retrieved_chunks, output_path):
    """
    将每一次问答记录保存到 jsonl 文件中。
    每一行是一条完整的问答记录。
    """

    sources = []
    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        sources.append({
            "source_id": i,
            "score": float(score),
            "chunk": chunk
        })

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "answer": answer,
        "sources": sources
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"

    chunks = load_pickle(output_dir / "chunks.pkl")
    chunk_embeddings = load_pickle(output_dir / "chunk_embeddings.pkl")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Mini RAG System 已启动")
    print("输入你的问题，输入 exit / quit / q 退出。")

    while True:
        query = input("\n请输入你的问题: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("已退出 Mini RAG System。")
            break

        if not query:
            print("问题不能为空，请重新输入。")
            continue

        results = retrieve_top_k(
            query,
            model,
            chunk_embeddings,
            chunks,
            top_k=3,
            min_score=0.15,
            relative_threshold=0.55
        )

        if not results:
            answer = "资料中没有找到相关信息。"
            output_file = output_dir / "last_query_result.txt"
            history_file = output_dir / "query_history.jsonl"

            save_answer(answer, output_file)
            save_history(query, answer, results, history_file)

            print("\n" + "=" * 50)
            print("RAG 最终答案：")
            print(answer)
            print(f"\n结果已保存到: {output_file}")
            print(f"问答历史已追加到: {history_file}")
            continue

        print("\n最相关的文本块：")
        for i, (chunk, score) in enumerate(results, start=1):
            print(f"\nTop {i} | score={score:.4f}")
            print(chunk)

        answer = generate_answer(query, results)

        output_file = output_dir / "last_query_result.txt"
        history_file = output_dir / "query_history.jsonl"

        save_answer(answer, output_file)
        save_history(query, answer, results, history_file)

        print("\n" + "=" * 50)
        print("RAG 最终答案：")
        print(answer)
        print(f"\n结果已保存到: {output_file}")
        print(f"问答历史已追加到: {history_file}")

if __name__ == "__main__":
    main()