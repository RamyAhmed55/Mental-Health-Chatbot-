from collections import deque
from langchain_core.messages import HumanMessage

from .prompts import SUMMARY_FIRST_PROMPT, SUMMARY_UPDATE_PROMPT


class RollingMemory:
    """
    rolling summary + last N verbatim turns.
    Summary updated incrementally — never re-summarizes from scratch.
    """

    def __init__(self, recent_window: int = 3):
        self.rolling_summary = ""
        self.recent_turns    = deque(maxlen=recent_window)
        self.window          = recent_window
        self._pending        = []

# =====================================================================================================

    def add(self, user_msg: str, bot_response: str, llm):
        turn = {"user": user_msg, "bot": bot_response}
        self.recent_turns.append(turn)
        self._pending.append(turn)

        if len(self._pending) >= self.window:
            self._update_summary(llm)
            self._pending = []

# =====================================================================================================

    def _update_summary(self, llm):

        new_turns_text = "\n".join(
            f"User: {t['user']}\nBot: {t['bot']}"
            for t in self._pending
        )

        if not self.rolling_summary:
            prompt = SUMMARY_FIRST_PROMPT.format(conversation=new_turns_text)
        else:
            prompt = SUMMARY_UPDATE_PROMPT.format(
                existing_summary=self.rolling_summary,
                new_turns=new_turns_text
            )

        resp                 = llm.invoke([HumanMessage(content=prompt)])
        self.rolling_summary = resp.content.strip()

# =====================================================================================================

    def get_context(self) -> str:
        parts = []

        if self.rolling_summary:
            parts.append(f"[Conversation summary]\n{self.rolling_summary}")

        if self.recent_turns:
            recent_text = "\n".join(
                f"User: {t['user']}\nBot: {t['bot']}"
                for t in self.recent_turns
            )
            parts.append(f"[Recent messages]\n{recent_text}")

        return "\n\n".join(parts) if parts else ""

# =====================================================================================================

    def is_empty(self) -> bool:
        return not self.recent_turns and not self.rolling_summary