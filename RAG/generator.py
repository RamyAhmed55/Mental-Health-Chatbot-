from langchain_core.messages import HumanMessage

from .prompts      import RAG_PROMPT
from .memory       import RollingMemory
from .embedder     import embed_query
from .vector_store import retrieve


# ── Fine-tuned local model (TODO — uncomment when ready) ──────────
# from transformers import pipeline
# local_model = pipeline("text-generation", model="./your_finetuned_model")
# def _generate_local(prompt): return local_model(prompt)[0]["generated_text"]
# ─────────────────────────────────────────────────────────────────

def generate_rag_response(question: str, emotion: str, language: str,
                           memory: RollingMemory, llm, embedder, client,
                           top_k: int = 5) -> dict:

    # 1. embed query & retrieve
    query_vec = embed_query(question, embedder)
    retrieved = retrieve(query_vec, client, top_k=top_k)

    combined_context = "\n\n---\n\n".join([
        f"Similar question a counselor handled:\n{r['similar_question']}\n\n"
        f"Counselor's response:\n{r['counselor_answer']}"
        for r in retrieved
    ])

    # 2. history
    history = memory.get_context()

    # 3. build prompt & call LLM
    prompt = RAG_PROMPT.format(
        emotion=emotion,
        language=language,
        context=combined_context,
        history=history or "No previous conversation",
        question=question
    )

    # ── LLM API ──
    response = llm.invoke([HumanMessage(content=prompt)])
    answer   = response.content.strip()

    # ── Local model (TODO) ──
    # answer = _generate_local(prompt)

    return {
        "answer"    : answer,
        "retrieved" : retrieved,
        "top_scores": [r["score"] for r in retrieved],
    }

# =====================================================================================================

def module4_rag(user_message: str, emotion: str, language: str,
                memory: RollingMemory, llm, embedder, client) -> str:

    result = generate_rag_response(
        user_message, emotion, language, memory, llm, embedder, client
    )

    memory.add(user_message, result["answer"], llm)
    return result["answer"]