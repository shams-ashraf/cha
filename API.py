# This file handles the main question-answering loop and API calls
# Flow: app.py → answer_question → loop [get chunks → build prompt → call LLM → process result] → return answer
import requests
import os
import time
from dotenv import load_dotenv
from typing import Tuple, Optional

from chat_engine.retrieval import search_chunks
from chat_engine.iteration import (
    get_next_chunk_batch,
    prepare_iteration_context,
    process_iteration_result
)
from chat_engine.utils import remove_status_from_answer
load_dotenv()

# API key rotation variables
current_key_index = 0
MAX_CONTEXT_TOKENS = 4000
TEXT_CHUNKS_PER_ITERATION = 2
TABLE_CHUNKS_PER_ITERATION = 2

# Load all available API keys
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
    os.getenv("GROQ_API_KEY_8"),
    os.getenv("GROQ_API_KEY_9"),
    os.getenv("GROQ_API_KEY_10"),
    os.getenv("GROQ_API_KEY_11"),
    os.getenv("GROQ_API_KEY_12")
]

# Remove None values
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    raise ValueError("No GROQ API keys found!")

GROQ_MODEL = "llama-3.3-70b-versatile"

# Track when each key becomes available again (after rate limit)
GROQ_RATE_LIMIT_UNTIL = [0] * len(GROQ_API_KEYS)
MAX_OUTPUT_TOKENS = 1000

###########################################################################
# MAIN FUNCTION: Answers user question using iterative retrieval
# Called by: app.py when user asks a question
# Returns: final answer string, list of used sources
# 
# Flow:
# 1. search_chunks() → gets all relevant text/table chunks from database
# 2. compress_chat_history() → summarizes previous conversation
# 3. get_system_prompt() → gets LLM instructions
# 4. Loop (until answer complete or no chunks left):
#    a. get_next_chunk_batch() → selects next chunks to send
#    b. prepare_iteration_context() → builds context text
#    c. build_user_content() → builds user prompt
#    d. call_groq_model() → gets answer from LLM
#    e. process_iteration_result() → extracts sources, checks completion
# 5. Return final answer + sources
def answer_question(query, chat_history=None, collection=None):
    
    if not collection:
        return "Collection is required", []
    
    # Step 1: Search database for all relevant chunks
    all_text_chunks, all_table_chunks = search_chunks(
        collection, query, n_text=80, n_tables=40
    )
    
    if not all_text_chunks and not all_table_chunks:
        return "❌ No information available in the documents.", []
    
    # Step 2: Compress chat history
    conversation_summary = compress_chat_history(chat_history, max_items=2)
    
    # Step 3: Get system prompt
    system_prompt = get_system_prompt()
    
    # Initialize iteration variables
    cumulative_cited_sources = []
    iteration = 1
    text_index = 0
    table_index = 0
    last_answer = ""
    used_pages = set()
    source_expansion_steps = {}

    # Step 4: Main iteration loop
    while True:
        
        # Get next batch of chunks to send
        new_text_batch, new_table_batch, is_expanding = get_next_chunk_batch(
            iteration,
            all_text_chunks,
            all_table_chunks,
            text_index,
            table_index,
            cumulative_cited_sources,
            last_answer,
            used_pages,
            collection,
            source_expansion_steps
        )

        # If no chunks left → stop
        if not new_text_batch and not new_table_batch:
            break
        
        # Build context string
        context, current_iteration_chunks = prepare_iteration_context(
            cumulative_cited_sources, new_text_batch, new_table_batch, MAX_CONTEXT_TOKENS
        )
        
        # Build user prompt
        user_content = build_user_content(
            conversation_summary=conversation_summary,
            context=context,
            query=query
        )

        # Call LLM
        answer, success = call_groq_model(system_prompt, user_content)
        if not success:
            if "Rate limited" in answer:
                return answer, []
            elif "Payload too large" in answer:
                continue  # Try next iteration with different chunks
            else:
                return answer, []
        
        last_answer = answer
        
        # Process result → extract sources, check completion
        cumulative_cited_sources, is_complete = process_iteration_result(
            answer,
            current_iteration_chunks,
            cumulative_cited_sources,
            used_pages,
        )
        
        # If answer complete → stop
        if is_complete:
            api_key, _ = get_next_available_key()
            if api_key:
                final_answer = answer 
            final_answer = remove_status_from_answer(answer)
            return final_answer, cumulative_cited_sources
        
        # Move to next batch (if not expanding)
        if not is_expanding:
            text_index += TEXT_CHUNKS_PER_ITERATION
            table_index += TABLE_CHUNKS_PER_ITERATION
        
        iteration += 1
    
    # Return last answer if loop ends
    return last_answer, cumulative_cited_sources

##############################################################
# Returns system prompt with instructions for LLM
# Called by: answer_question
# Returns: system prompt string
def get_system_prompt():
    return """You are an accurate assistant for the Master Biomedical Engineering (MBE) program.

GENERAL RULES:
- Use ONLY information explicitly stated in the provided documents or conversation history.
- Do NOT assume, infer, extend, or fabricate information.
- Do NOT use external knowledge.
- Do NOT contradict yourself.

QUESTION HANDLING:
- If the user question contains multiple parts, you MUST internally split it into sub-questions.
- Sub-questions MUST be ONLY the core requirements explicitly stated by the user.
- You MUST NOT invent, expand, or add related or implied questions.
- You MUST answer ONLY what was explicitly asked.

ANSWER RULES:
- Write clear, structured answers using bullet points or short paragraphs.
- Include ONLY factual statements that are directly supported by the sources.
- Do NOT explain reasoning.
- Do NOT add commentary, assumptions, or extra context.
- Do NOT mention document names, page numbers, or sources inside the Answer section.

SOURCE USAGE RULES (STRICT):
- The Sources section MUST include ONLY the sources that were DIRECTLY USED to ASSERT factual statements in the Answer.
- Sources MUST correspond ONLY to the sub-questions that were actually answered.
- If the answer status is Partial information:
  - List ONLY the sources used for the answered sub-questions.
  - Do NOT list sources related to unanswered sub-questions.
- Do NOT list sources that were only checked, reviewed, or inspected.
- Do NOT list sources used only to confirm that information is missing.
- Do NOT list unused, related, or background sources.

STATUS LOGIC (STRICT):
- Internally evaluate EACH sub-question separately.

Status definitions:
- Complete information:
  ALL sub-questions are answered.
- Partial information:
  At least ONE sub-question is answered, but not all.
- No information:
  NONE of the sub-questions are answered.

Hard constraints:
- If ANY sub-question is answered, "No information" is NOT allowed.
- "Complete information" is allowed ONLY if ALL sub-questions are answered.
- Do NOT downgrade status because of missing details outside the asked sub-questions.

OUTPUT FORMAT (EXACT ORDER):

Answer:
<final answer text>

Sources:
- <document name> p<page number>
- (list ONLY sources that directly support the Answer)

Status:
- Write EXACTLY ONE:
  No information
  Partial information
  Complete information

"""

##############################################################
# Builds user prompt by combining history + context + query
# Called by: answer_question before calling LLM
# Returns: user prompt string
def build_user_content(conversation_summary, context, query):
    return f"""{conversation_summary if conversation_summary else ''}

SOURCES:
{context}
QUESTION: {query}
Answer strictly according to the system rules. Follow the required output format exactly.
"""

###########################################################
# Rotates through API keys to avoid rate limits
# Called by: call_groq_model
# Returns: API key (or None), key index (or wait time)
# Logic:
# 1. Checks if current key is available
# 2. If rate-limited → moves to next key
# 3. Returns first available key
def get_next_available_key() -> Tuple[Optional[str], int]:
    global current_key_index
    now = time.time()
    
    # Try all keys
    for _ in range(len(GROQ_API_KEYS)):
        if now >= GROQ_RATE_LIMIT_UNTIL[current_key_index]:
            key = GROQ_API_KEYS[current_key_index]
            index = current_key_index
            current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
            return key, index
        
        current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
    
    # All keys rate-limited → return wait time
    earliest_available = min(GROQ_RATE_LIMIT_UNTIL)
    wait_seconds = max(1, int(earliest_available - now))
    return None, wait_seconds

#############################################################
# Calls Groq API to get LLM response
# Called by: answer_question in main loop
# Returns: answer text, success flag
# Handles:
# - API key rotation
# - Rate limiting (429 error)
# - Payload too large (413 error)
# - Other errors
def call_groq_model(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.05,
    max_tokens: int = MAX_OUTPUT_TOKENS
) -> Tuple[str, bool]:

    while True:
        # Get next available API key
        api_key, info = get_next_available_key()

        if api_key is None:
            return (f"All API keys are rate limited. Please wait {info} seconds.", False)

        # Build request
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            # Make API call
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=60
            )

            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()
            return answer, True

        except requests.exceptions.HTTPError as e:
            # Rate limit error → mark key as unavailable and try next
            if e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else 60
                GROQ_RATE_LIMIT_UNTIL[info] = time.time() + wait_time
                continue

            # Payload too large → return to try with fewer chunks
            elif e.response is not None and e.response.status_code == 413:
                return "Payload too large", False

            return f"HTTP Error: {str(e)}", False

        except Exception as e:
            time.sleep(5)
            continue
       
#############################################################
# Summarizes recent chat history for context
# Called by: answer_question at start
# Returns: conversation summary string
# Takes last N user-assistant pairs and formats them
def compress_chat_history(chat_history, max_items=2):
    if not chat_history:
        return ""
    
    recent_pairs = []
    
    # Extract last N question-answer pairs
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i]["role"] == "user":
            user_msg = chat_history[i]["content"]
            if i + 1 < len(chat_history) and chat_history[i + 1]["role"] == "assistant":
                assistant_msg = chat_history[i + 1]["content"]
                recent_pairs.insert(0, (user_msg, assistant_msg))
                if len(recent_pairs) >= max_items:
                    break
    
    # Format pairs
    if recent_pairs:
        summary = ["=== Previous Conversation ==="]
        for idx, (q, a) in enumerate(recent_pairs, 1):
            summary.append(f"\nQ{idx}: {q}")
            # Truncate long answers
            if len(a) > 300:
                summary.append(f"A{idx}: {a[:300]}...")
            else:
                summary.append(f"A{idx}: {a}")
        summary.append("\n=== End ===\n")
        return "\n".join(summary)
    
    return ""

#############################################################