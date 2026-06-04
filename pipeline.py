import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from language_detection.detector       import LanguageDetector
from emotion_detection.classifier      import EmotionDetector
from intent_classifier.memory          import ConversationMemory
from intent_classifier.classifier      import module3_pipeline
from rag.data_loader                   import load_records
from rag.embedder                      import load_embedder, load_embeddings
from rag.vector_store                  import get_client
from rag.memory                        import RollingMemory
from rag.generator                     import module4_rag


load_dotenv()


# =====================================================================================================


def build_pipeline(verbose: bool = False):
    """
    Loads all models/clients once and returns a ready chat function.
    verbose=True  → shows each module output per turn  (debug mode)
    verbose=False → shows final answer only            (clean mode)
    """

    print("Loading models...")

    # Module 1
    lang_detector   = LanguageDetector()

    # Module 2
    emotion_detector = EmotionDetector()

    # LLM shared between module 3 & 4
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Module 4 resources
    records    = load_records()
    embedder   = load_embedder()
    embeddings = load_embeddings(records, embedder)
    qdrant     = get_client(
        url     = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY")
    )

    # Memories
    intent_memory  = ConversationMemory(max_turns=3)
    rolling_memory = RollingMemory(recent_window=3)

    print("All models loaded. Ready to chat!\n")

    # ─────────────────────────────────────────────────────────────────

    def chat(user_message: str) -> str:

        # Module 1
        language = lang_detector.detect_code(user_message)   # ممكن يرجع "unknown"

        # Module 2
        if language == "en":
            emotion = emotion_detector.classify(user_message)
        else:
            emotion = "unknown"

        # Module 3 — fixed intent classification + routing + direct response generation
        routing = module3_pipeline(
            user_message = user_message,
            emotion      = emotion,
            language     = language,
            memory       = intent_memory,
            llm          = llm
        )

        # use truth from module 1 & 2 if LLM not confident in detection
        language = routing["language"]
        emotion  = routing["emotion"]

        if verbose:
            print(f"\n{'─'*50}")
            print(f"[M1] Language : {language}")
            print(f"[M2] Emotion  : {emotion}")
            print(f"[M3] Intent   : {routing['intent']}")
            print(f"[M3] Action   : {routing['action']}")
            print(f"[M3] MH related: {routing['is_mental_health_related']}")

        # ── Module 4: RAG (only if needed) ────────────────────────────
        if routing["needs_rag"]:
            answer = module4_rag(
                user_message = user_message,
                emotion      = emotion,
                language     = language,
                memory       = rolling_memory,
                llm          = llm,
                embedder     = embedder,
                client       = qdrant
            )
            # update intent memory with real answer
            intent_memory.history[-1]["bot"] = answer

        else:
            answer = routing["response"]

        if verbose:
            print(f"[M4] RAG used : {routing['needs_rag']}")
            print(f"{'─'*50}")

        return answer

    return chat


# =====================================================================================================


def run_cli(verbose: bool = False):

    chat = build_pipeline(verbose=verbose)

    print("Mental Health Chatbot — type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bot: Take care of yourself. Goodbye!")
            break

        answer = chat(user_input)
        print(f"Bot: {answer}\n")


# =====================================================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show each module output per turn"
    )
    args = parser.parse_args()

    run_cli(verbose=args.verbose)