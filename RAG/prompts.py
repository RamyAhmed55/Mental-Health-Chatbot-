RAG_PROMPT = """You are a compassionate mental health support assistant.

The user is feeling: {emotion}
Their language: {language}

Relevant knowledge from mental health counseling:
{context}

Conversation history:
{history}

User question: {question}

Instructions:
- Answer based on the provided context
- Be warm, empathetic, and supportive
- Reply in the SAME language as the user ({language})
- Do NOT give medical diagnoses
- Keep response focused: 3-5 sentences

Response:"""

# =====================================================================================================

SUMMARY_FIRST_PROMPT = """Summarize this conversation in 2-3 sentences.
Focus on mental health topics discussed. Be concise.

Conversation:
{conversation}

Summary:"""

# =====================================================================================================

SUMMARY_UPDATE_PROMPT = """You have a conversation summary and new messages.
Update the summary to include the new information. Keep it 2-3 sentences max.

Existing summary:
{existing_summary}

New messages to add:
{new_turns}

Updated summary:"""