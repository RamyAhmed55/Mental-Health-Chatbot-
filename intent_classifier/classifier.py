import json
from langchain_core.messages import HumanMessage

from .prompts import FIRST_TURN_PROMPT, FOLLOWUP_TURN_PROMPT, DIRECT_RESPONSE_PROMPT
from .memory  import ConversationMemory


ROUTES = {
    "greeting" : "direct_reply",
    "goodbye"   : "direct_reply",
    "gratitude" : "direct_reply",
    "asking_mental_health_question": "rag_pipeline",
    "out_of_scope" : "polite_refusal",
}

# =====================================================================================================

def classify_intent(user_message: str, memory: ConversationMemory, llm) -> dict:

    if memory.is_empty():
        prompt = FIRST_TURN_PROMPT.format(user_message=user_message)
        
    else:
        prompt = FOLLOWUP_TURN_PROMPT.format(
            history=memory.get_context(),
            user_message=user_message
        )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content.strip())
    except json.JSONDecodeError:
        result = {"intent": "out_of_scope", "is_mental_health_related": False}

    return result

# =====================================================================================================

def generate_direct_response(intent: str, user_message: str,
                              emotion: str, language: str, llm) -> str:

    prompt = DIRECT_RESPONSE_PROMPT.format(
        intent=intent,
        emotion=emotion,
        language=language,
        user_message=user_message
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# =====================================================================================================

def route(user_message: str, emotion: str, language: str,
          memory: ConversationMemory, llm) -> dict:

    classification = classify_intent(user_message, memory, llm)
    intent         = classification["intent"]
    is_mh          = classification["is_mental_health_related"]

    # use LLM detection if modules not confident
    final_language = language if language != "unknown" else classification.get("language", "en")
    final_emotion  = emotion  if emotion  != "unknown" else classification.get("emotion",  "neutral")

    if intent == "out_of_scope" and is_mh:
        intent = "asking_mental_health_question"

    return {
        "intent"                  : intent,
        "action"                  : ROUTES.get(intent, "polite_refusal"),
        "emotion"                 : final_emotion,
        "language"                : final_language,
        "is_mental_health_related": is_mh,
    }

# =====================================================================================================

def module3_pipeline(user_message: str, emotion: str,language: str, memory: ConversationMemory, llm) -> dict:

    routing = route(user_message, emotion, language, memory, llm)
    intent  = routing["intent"]
    action  = routing["action"]

    if action == "rag_pipeline":
        result = {**routing, "response": None, "needs_rag": True}

    else:
        reply  = generate_direct_response(intent, user_message, emotion, language, llm)
        result = {**routing, "response": reply, "needs_rag": False}


    memory.add(user_message, result["response"] or "[rag_response]")

    return result