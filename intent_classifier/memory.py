from collections import deque


class ConversationMemory:

    def __init__(self, max_turns=3):
        self.history = deque(maxlen=max_turns)

    def add(self, user_msg, bot_response):
        self.history.append({"user": user_msg, "bot": bot_response})

# =====================================================================================================

    def get_context(self):
        if not self.history:
            return ""

        lines = []
        for turn in self.history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot: {turn['bot']}")

        return "\n".join(lines)

# =====================================================================================================

    def is_empty(self):
        return len(self.history) == 0