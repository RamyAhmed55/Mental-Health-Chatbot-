const messagesEl = document.getElementById("chatMessages");
const inputEl    = document.getElementById("userInput");
const sendBtn    = document.getElementById("sendBtn");
const themeToggle= document.getElementById("themeToggle");
const charCount  = document.getElementById("charCount");

// ── Theme ─────────────────────────────────────────────────────────
themeToggle.addEventListener("click", () => {
    const html    = document.documentElement;
    const current = html.getAttribute("data-theme");
    html.setAttribute("data-theme", current === "dark" ? "light" : "dark");
});

// ── Time helper ───────────────────────────────────────────────────
function getTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// set initial welcome time
document.querySelectorAll("[data-time='now']").forEach(el => {
    el.textContent = getTime();
});

// ── Auto-resize textarea ──────────────────────────────────────────
inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = inputEl.scrollHeight + "px";
    charCount.textContent = `${inputEl.value.length}/500`;
});

// ── Send on Enter (Shift+Enter = new line) ────────────────────────
inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener("click", sendMessage);

// ── Add message bubble ────────────────────────────────────────────
function addMessage(text, role) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}-message`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;

    const time = document.createElement("span");
    time.className   = "message-time";
    time.textContent = getTime();

    wrapper.appendChild(bubble);
    wrapper.appendChild(time);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    return wrapper;
}

// ── Typing indicator ──────────────────────────────────────────────
function showTyping() {
    const wrapper = document.createElement("div");
    wrapper.className = "message bot-message typing-indicator";
    wrapper.id = "typingIndicator";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = "<span></span><span></span><span></span>";

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
}

// ── Main send function ────────────────────────────────────────────
async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    // disable input while waiting
    inputEl.value      = "";
    inputEl.style.height = "auto";
    charCount.textContent = "0/500";
    sendBtn.disabled   = true;

    addMessage(text, "user");
    showTyping();

    try {
        const res  = await fetch("/chat", {
            method : "POST",
            headers: { "Content-Type": "application/json" },
            body   : JSON.stringify({ message: text }),
        });

        const data = await res.json();
        hideTyping();

        if (data.error) {
            addMessage("Something went wrong. Please try again.", "bot");
        } else {
            addMessage(data.answer, "bot");
        }

    } catch {
        hideTyping();
        addMessage("Connection error. Please try again.", "bot");
    }

    sendBtn.disabled = false;
    inputEl.focus();
}