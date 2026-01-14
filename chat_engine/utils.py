# This file has utility functions used across the project
# Used by: iteration.py and API.py

import re

################################################################################
# Checks if LLM answer says "partial information"
# Called by: iteration.get_next_chunk_batch and iteration.process_iteration_result
# Returns: True if incomplete, False otherwise
# Used to decide if we need more context
def check_if_answer_incomplete(answer: str) -> bool:
    answer_lower = answer.lower()
    
    if "status:" in answer_lower and "partial information" in answer_lower:
        return True

    return False

################################################################################
# Checks if LLM answer says "no information found"
# Called by: iteration.process_iteration_result
# Returns: True if insufficient, False otherwise
# Used to decide if we should extract sources from this answer
def check_if_answer_insufficient(answer: str) -> bool:
    answer_lower = answer.lower()

    if "❌ no sufficient information found" in answer_lower:
        return True

    if "status:" in answer_lower and "no information" in answer_lower:
        return True

    return False

################################################################################
# Extracts which sources were actually used in the answer
# Called by: iteration.process_iteration_result
# Returns: list of chunks that were cited
# Logic:
# 1. Finds "Sources:" section in answer
# 2. Parses source names and page numbers
# 3. Matches them against used_chunks to get full chunk data
def extract_used_sources_from_answer(answer: str, used_chunks: list) -> list:
    actually_used = []

    # Find Sources section
    match = re.search(r"Sources:\s*(.*)", answer, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    sources_text = match.group(1).lower()

    # Parse each line in Sources section
    source_lines = [
        line.strip("-• ").strip()
        for line in sources_text.splitlines()
        if line.strip()
    ]

    # Match each chunk against cited sources
    for chunk in used_chunks:
        if not isinstance(chunk, dict):
            continue

        source = chunk.get("source") or chunk.get("metadata", {}).get("source", "")
        page = chunk.get("page") or chunk.get("metadata", {}).get("page", "")

        if not source or not page:
            continue

        # Clean source name
        source_name = source.split("/")[-1].replace(".pdf", "").lower()

        # Check if this chunk was cited
        for line in source_lines:
            if source_name in line and str(page) in line:
                actually_used.append(chunk)
                break

    return actually_used

################################################################################
# Rough token estimation (used for sizing chunks)
# Called by: everywhere that needs to check chunk size
# Returns: approximate token count
# Simple rule: 1 token ≈ 4 characters
def estimate_tokens(text: str) -> int:
    return len(text) // 4

################################################################################
import re

def remove_status_from_answer(answer: str) -> str:
    return re.sub(
        r"(?im)^status:\s*.*$",
        "",
        answer
    ).strip()
