import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from language_detection.detector  import LanguageDetector
from emotion_detection.classifier import EmotionDetector
from intent_classifier.memory     import ConversationMemory
from intent_classifier.classifier import module3_pipeline
from rag.data_loader              import load_records
from rag.embedder                 import load_embedder, load_embeddings
from rag.vector_store             import get_client
from rag.memory                   import RollingMemory
from rag.generator                import module4_rag

load_dotenv()

# =====================================================================================================

class PipelineManager:
    """Single instance — loaded once when Flask starts."""

    def __init__(self):
        print("Loading pipeline...")

        self.lang_detector    = LanguageDetector()
        self.emotion_detector = EmotionDetector()

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        records          = load_records()
        self.embedder    = load_embedder()
        embeddings       = load_embeddings(records, self.embedder)
        self.qdrant      = get_client(
            url     = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )

        self.intent_memory  = ConversationMemory(max_turns=3)
        self.rolling_memory = RollingMemory(recent_window=3)

        print("Pipeline ready.")

# =====================================================================================================

    def chat(self, user_message: str) -> dict:

        language = self.lang_detector.detect_code(user_message)
        emotion  = self.emotion_detector.classify(user_message) if language == "en" else "unknown"

        routing = module3_pipeline(
            user_message = user_message,
            emotion      = emotion,
            language     = language,
            memory       = self.intent_memory,
            llm          = self.llm
        )

        language = routing["language"]
        emotion  = routing["emotion"]

        if routing["needs_rag"]:
            answer = module4_rag(
                user_message = user_message,
                emotion      = emotion,
                language     = language,
                memory       = self.rolling_memory,
                llm          = self.llm,
                embedder     = self.embedder,
                client       = self.qdrant
            )
            self.intent_memory.history[-1]["bot"] = answer
        else:
            answer = routing["response"]

        return {
            "answer"  : answer,
            "language": language,
            "emotion" : emotion,
            "intent"  : routing["intent"],
        }


# single instance shared across requests
pipeline = PipelineManager()