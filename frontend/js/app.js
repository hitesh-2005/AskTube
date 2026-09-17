import { processVideo, askQuestion } from "./api.js";
import { initYouTubeApi, loadVideo, seekTo, resetPlayer } from "./player.js";

// DOM Elements
const urlInput = document.getElementById("url-input");
const submitUrlBtn = document.getElementById("submit-url-btn");
const newVideoBtn = document.getElementById("new-video-btn");
const videoInfoCard = document.getElementById("video-info-card");
const videoTitleEl = document.getElementById("video-title");
const videoStatusEl = document.getElementById("video-status");
const videoLangEl = document.getElementById("video-lang");
const videoTypeEl = document.getElementById("video-type");
const chatHistory = document.getElementById("chat-history");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const processingBanner = document.getElementById("processing-banner");
const errorBanner = document.getElementById("error-banner");
const errorMessageEl = document.getElementById("error-message");

// Application State
let activeVideoId = null;
let isProcessing = false;
let isAnswering = false;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  initYouTubeApi();
  setupEventListeners();
});

function setupEventListeners() {
  submitUrlBtn.addEventListener("click", handleUrlSubmit);
  urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleUrlSubmit();
    }
  });

  newVideoBtn.addEventListener("click", handleNewVideo);

  const retryBtn = document.getElementById("retry-btn");
  if (retryBtn) {
    retryBtn.addEventListener("click", () => {
      hideError();
      handleUrlSubmit();
    });
  }

  sendBtn.addEventListener("click", handleQuestionSubmit);
  promptInput.addEventListener("input", () => {
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 120) + "px";
  });
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuestionSubmit();
    }
  });

  // Delegate click for suggested chips
  chatHistory.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip-btn");
    if (chip && activeVideoId && !isAnswering) {
      const q = chip.getAttribute("data-question") || chip.textContent.trim();
      promptInput.value = q;
      handleQuestionSubmit();
    }
  });
}

function showError(message) {
  errorMessageEl.textContent = message;
  errorBanner.style.display = "flex";
  processingBanner.style.display = "none";
}

function hideError() {
  errorBanner.style.display = "none";
}

async function handleUrlSubmit() {
  const url = urlInput.value.trim();
  if (!url || isProcessing) return;

  hideError();
  isProcessing = true;
  submitUrlBtn.disabled = true;
  processingBanner.style.display = "flex";
  processingBanner.querySelector(".processing-text").textContent = "Indexing video captions & preparing semantic search...";

  try {
    const data = await processVideo(url);
    if (data.status === "error" || data.error) {
      throw new Error(data.error?.message || "Failed to process video");
    }

    activeVideoId = data.video_id;
    loadVideo(activeVideoId);

    // Update video info card
    videoTitleEl.textContent = data.title || `YouTube Video (${activeVideoId})`;
    videoStatusEl.className = "status-pill ready";
    videoStatusEl.innerHTML = `<span class="status-dot"></span> Ready`;
    videoLangEl.textContent = `Language: ${data.transcript_language.toUpperCase()}`;
    videoTypeEl.textContent = data.is_generated ? "Auto-generated captions" : "Manual captions";
    videoInfoCard.style.display = "flex";

    // Setup chat ready state
    chatHistory.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
        <h2 class="empty-state-title">Video is ready to explore</h2>
        <p class="empty-state-desc">Ask anything about what is explained in this video. Answers cite exact timestamps and are strictly grounded in spoken transcript evidence.</p>
        <div class="suggested-chips">
          <button class="chip-btn" data-question="What is the main topic of this video?">
            <span>What is the main topic of this video?</span>
          </button>
          <button class="chip-btn" data-question="What are the key points explained?">
            <span>What are the key points explained?</span>
          </button>
          <button class="chip-btn" data-question="What conclusion or summary does the speaker provide?">
            <span>What conclusion or summary does the speaker provide?</span>
          </button>
        </div>
      </div>
    `;

    promptInput.disabled = false;
    sendBtn.disabled = false;
    promptInput.focus();
  } catch (err) {
    showError(err.message);
  } finally {
    isProcessing = false;
    submitUrlBtn.disabled = false;
    processingBanner.style.display = "none";
  }
}

function handleNewVideo() {
  activeVideoId = null;
  urlInput.value = "";
  promptInput.value = "";
  promptInput.style.height = "auto";
  promptInput.disabled = true;
  sendBtn.disabled = true;
  videoInfoCard.style.display = "none";
  resetPlayer();
  hideError();

  chatHistory.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      </div>
      <h1 class="empty-state-title">Ask questions about any YouTube video</h1>
      <p class="empty-state-desc">Paste a video link above to index its spoken transcript. Every answer is grounded directly in the transcript with clickable timestamp evidence.</p>
    </div>
  `;
}

async function handleQuestionSubmit() {
  const question = promptInput.value.trim();
  if (!question || !activeVideoId || isAnswering) return;

  isAnswering = true;
  promptInput.disabled = true;
  sendBtn.disabled = true;

  // Clear welcome/empty state on first question
  if (chatHistory.querySelector(".empty-state")) {
    chatHistory.innerHTML = "";
  }

  // Append user bubble
  appendUserMessage(question);
  promptInput.value = "";
  promptInput.style.height = "auto";

  // Append loading state
  const loadingRow = appendAssistantLoading();
  scrollToBottom();

  try {
    const data = await askQuestion(activeVideoId, question);
    loadingRow.remove();

    if (data.status === "error" || data.error) {
      appendErrorMessage(data.error?.message || "Failed to retrieve answer");
    } else if (data.status === "no_context" || !data.relevant_context_found) {
      appendNoContextMessage(data.answer);
    } else {
      appendAssistantAnswer(data);
    }
  } catch (err) {
    loadingRow.remove();
    appendErrorMessage(err.message);
  } finally {
    isAnswering = false;
    promptInput.disabled = false;
    sendBtn.disabled = false;
    promptInput.focus();
    scrollToBottom();
  }
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
  chatHistory.appendChild(row);
}

function appendAssistantLoading() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="assistant-response">
      <div class="no-context-card">
        <div class="spinner"></div>
        <span>Searching transcript and grounding answer...</span>
      </div>
    </div>
  `;
  chatHistory.appendChild(row);
  return row;
}

function appendNoContextMessage(text) {
  const cleanText = (text || "").trim() || "I couldn't find enough information in the transcript.";
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="assistant-response">
      <div class="no-context-card">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <span>${escapeHtml(cleanText)}</span>
      </div>
    </div>
  `;
  chatHistory.appendChild(row);
}

function appendErrorMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="assistant-response">
      <div class="error-banner" style="margin: 0; border-radius: var(--rounded-md);">
        <div class="error-banner-content">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>${escapeHtml(text)}</span>
        </div>
      </div>
    </div>
  `;
  chatHistory.appendChild(row);
}

function formatMarkdownAnswer(rawText) {
  if (!rawText) return "";
  
  // Escape HTML first for XSS prevention
  const escaped = escapeHtml(rawText);

  // Split into lines
  const lines = escaped.split("\n");
  const htmlBlocks = [];
  let currentList = [];
  let inOrderedList = false;

  function flushList() {
    if (currentList.length > 0) {
      const tag = inOrderedList ? "ol" : "ul";
      htmlBlocks.push(`<${tag}>${currentList.map(li => `<li>${li}</li>`).join("")}</${tag}>`);
      currentList = [];
      inOrderedList = false;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    if (!line) {
      flushList();
      continue;
    }

    // Format bold **text**
    line = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Section headers like "Summary:" or "Key Points:"
    const headingMatch = line.match(/^(?:<strong>)?(Summary|Key Points|Conclusion|Overview|Takeaways):?(?:<\/strong>)?:?$/i);
    if (headingMatch) {
      flushList();
      htmlBlocks.push(`<div class="answer-section-heading">${headingMatch[1]}</div>`);
      continue;
    }

    // Unordered bullet list items (- item or * item or • item)
    const bulletMatch = line.match(/^[-*•]\s+(.*)$/);
    if (bulletMatch) {
      if (inOrderedList) flushList();
      currentList.push(bulletMatch[1]);
      continue;
    }

    // Numbered list items (1. item)
    const numMatch = line.match(/^\d+\.\s+(.*)$/);
    if (numMatch) {
      if (!inOrderedList) flushList();
      inOrderedList = true;
      currentList.push(numMatch[1]);
      continue;
    }

    // Regular line / paragraph
    flushList();
    htmlBlocks.push(`<p>${line}</p>`);
  }

  flushList();
  return htmlBlocks.join("");
}

function appendAssistantAnswer(data) {
  const row = document.createElement("div");
  row.className = "message-row assistant";

  const responseContainer = document.createElement("div");
  responseContainer.className = "assistant-response";

  // Answer card or defensive empty-answer guard
  const rawAnswer = (data.answer || "").trim();
  if (rawAnswer) {
    const answerCard = document.createElement("div");
    answerCard.className = "answer-card";
    answerCard.innerHTML = formatMarkdownAnswer(rawAnswer);
    responseContainer.appendChild(answerCard);
  } else {
    const noContextCard = document.createElement("div");
    noContextCard.className = "no-context-card";
    noContextCard.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </svg>
      <span>I couldn't find enough information in the transcript.</span>
    `;
    responseContainer.appendChild(noContextCard);
  }

  // Retrieved Transcript Sections Accordion
  if (data.retrieved_sections && data.retrieved_sections.length > 0) {
    const accordion = document.createElement("div");
    accordion.className = "sources-accordion";

    const count = data.retrieved_sections.length;
    accordion.innerHTML = `
      <div class="accordion-header" role="button" tabindex="0" aria-expanded="false" aria-label="Toggle retrieved transcript context">
        <span>Retrieved Transcript Evidence (${count})</span>
        <svg class="accordion-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>
      <div class="accordion-body"></div>
    `;

    const body = accordion.querySelector(".accordion-body");
    data.retrieved_sections.forEach((sec) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.innerHTML = `
        <div class="source-meta">
          <button class="timestamp-pill" data-seconds="${sec.start_seconds}" aria-label="Jump to timestamp ${escapeHtml(sec.timestamp)}">
            <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6 4 20 12 6 20 6 4"></polygon>
            </svg>
            <span>${escapeHtml(sec.timestamp)}</span>
          </button>
        </div>
        <div class="source-text">${escapeHtml(sec.text)}</div>
      `;

      // Click pill to seek player
      const pill = item.querySelector(".timestamp-pill");
      pill.addEventListener("click", () => {
        seekTo(parseFloat(sec.start_seconds));
      });

      body.appendChild(item);
    });

    // Toggle accordion
    const header = accordion.querySelector(".accordion-header");
    const toggleAccordion = () => {
      const isOpen = accordion.classList.toggle("open");
      header.setAttribute("aria-expanded", isOpen ? "true" : "false");
    };
    header.addEventListener("click", toggleAccordion);
    header.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleAccordion();
      }
    });

    responseContainer.appendChild(accordion);
  }

  row.appendChild(responseContainer);
  chatHistory.appendChild(row);
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(text) {
  if (!text) return "";
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}
