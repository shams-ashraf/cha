# This file manages iterative retrieval (asking LLM multiple times with different chunks)
# Flow: API.py loops calling → get_next_chunk_batch → prepare_iteration_context → process_iteration_result

from typing import Tuple

from .retrieval import get_surrounding_pages_smart
from .utils import (
    check_if_answer_incomplete,
    check_if_answer_insufficient,
    extract_used_sources_from_answer,
    estimate_tokens
)

MAX_CONTEXT_TOKENS = 4000

TEXT_CHUNKS_PER_ITERATION = 6
TABLE_CHUNKS_PER_ITERATION = 4

################################################################################
# Decides which chunks to send to LLM in this iteration
# Called by: API.answer_question in main loop
# Returns: text chunks, table chunks, is_expanding flag
# Logic:
# 1. First iteration → send first batch of chunks
# 2. If answer incomplete → get surrounding pages of cited sources
# 3. Otherwise → send next batch of unused chunks
def get_next_chunk_batch(
    iteration: int,
    all_text_chunks: list,
    all_table_chunks: list,
    text_index: int,
    table_index: int,
    cumulative_cited_sources: list,
    last_answer: str,
    used_pages: set,
    collection,
    source_expansion_steps: dict
) -> Tuple[list, list, bool]:

    # First iteration → just send first batch
    if iteration == 1:
        new_text_batch = all_text_chunks[text_index:text_index + TEXT_CHUNKS_PER_ITERATION]
        new_table_batch = all_table_chunks[table_index:table_index + TABLE_CHUNKS_PER_ITERATION]
        return new_text_batch, new_table_batch, False
    
    # Check if last answer was incomplete
    is_incomplete = check_if_answer_incomplete(last_answer)

    # If incomplete → get surrounding pages of cited sources
    if is_incomplete and cumulative_cited_sources:
        seen_sources = set()
        BASE_SURROUNDING_RANGE = 1
        MAX_SURROUNDING_RANGE = 4

        expanded_chunks = []

        # For each cited source, get nearby pages
        for chunk in cumulative_cited_sources:
            source = chunk.get("source")
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)

            # Track how many times we expanded this source
            if source not in source_expansion_steps:
                source_expansion_steps[source] = 0

            current_step = source_expansion_steps.get(source, 0)

            # Increase range each time (1 → 2 → 3 → 4 pages)
            dynamic_range = min(
                BASE_SURROUNDING_RANGE + current_step,
                MAX_SURROUNDING_RANGE
            )

            # Get surrounding pages from database
            surrounding = get_surrounding_pages_smart(
                collection,
                [chunk],
                pages_range=dynamic_range
            )

            if surrounding:
                source_expansion_steps[source] = current_step + 1
                expanded_chunks.extend(surrounding)

        # Filter out already used pages
        filtered_chunks = []
        for ch in expanded_chunks:
            key = f"{ch.get('source')}_{ch.get('page')}"
            if key not in used_pages:
                used_pages.add(key)
                filtered_chunks.append(ch)

        # Separate text and tables
        new_text_batch = [
            c for c in filtered_chunks
            if c.get("type") not in ["table", "table_with_context"]
        ]

        new_table_batch = [
            c for c in filtered_chunks
            if c.get("type") in ["table", "table_with_context"]
        ]

        if new_text_batch or new_table_batch:
            return new_text_batch, new_table_batch, True
    
    # If no more chunks → stop
    if text_index >= len(all_text_chunks) and table_index >= len(all_table_chunks):
       return [], [], False
    
    # Otherwise → send next batch of unused chunks
    new_text_batch = []
    new_table_batch = []
    
    candidate_text = all_text_chunks[text_index:text_index + TEXT_CHUNKS_PER_ITERATION]
    for chunk in candidate_text:
        key = f"{chunk.get('source')}_{chunk.get('page')}"
        if key not in used_pages:
            new_text_batch.append(chunk)
    
    candidate_tables = all_table_chunks[table_index:table_index + TABLE_CHUNKS_PER_ITERATION]
    for chunk in candidate_tables:
        key = f"{chunk.get('source')}_{chunk.get('page')}"
        if key not in used_pages:
            new_table_batch.append(chunk)
    
    return new_text_batch, new_table_batch, False

################################################################################
# Prepares context text to send to LLM
# Called by: API.answer_question before calling LLM
# Returns: context string, list of chunks in this iteration
# Logic:
# 1. Add previously cited sources (last 6)
# 2. Add new table chunks
# 3. Add new text chunks
# 4. Trim if total exceeds max tokens
def prepare_iteration_context(
    cumulative_cited_sources: list,
    new_text_batch: list,
    new_table_batch: list,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    max_used_sources: int = 6
) -> Tuple[str, list]:

    context_parts = []
    current_iteration_chunks = []
    seen_keys = set()

    # Add previously used sources (for context)
    if cumulative_cited_sources:
        used_sources_limited = cumulative_cited_sources[-max_used_sources:]

        for chunk in used_sources_limited:
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "N/A")
            content = chunk.get("content", "")
            chunk_type = chunk.get("type", "text")

            key = f"{source}_{page}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Mark as "USED" so LLM knows it saw this before
            type_marker = "[TABLE]" if chunk_type in ["table", "table_with_context"] else "[TEXT]"
            context_parts.append(
                f"[📌 USED {type_marker} {source} p{page}]\n{content}"
            )
            current_iteration_chunks.append(chunk)

    # Add new tables (priority over text)
    for chunk in new_table_batch:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "N/A")
        key = f"{source}_{page}"

        if key in seen_keys:
            continue
        seen_keys.add(key)

        context_parts.append(
            f"[📊 NEW {source} p{page}]\n{chunk.get('content', '')}"
        )
        current_iteration_chunks.append(chunk)

    # Add new text chunks
    for chunk in new_text_batch:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "N/A")
        key = f"{source}_{page}"

        if key in seen_keys:
            continue
        seen_keys.add(key)

        context_parts.append(
            f"[📄 NEW {source} p{page}]\n{chunk.get('content', '')}"
        )
        current_iteration_chunks.append(chunk)

    # Trim if too long
    context = trim_context_to_fit(context_parts, max_tokens)

    return context, current_iteration_chunks

################################################################################
# Processes LLM answer to extract cited sources and check if complete
# Called by: API.answer_question after getting LLM response
# Returns: updated cumulative sources, is_complete flag
# Logic:
# 1. Check if answer is insufficient (no info found)
# 2. If sufficient → extract cited sources from answer
# 3. Add new sources to cumulative list (avoid duplicates)
# 4. Check if answer is complete
def process_iteration_result(
    answer: str,
    current_iteration_chunks: list,
    cumulative_cited_sources: list,
    used_pages: set,
) -> Tuple[list, bool]:
    
    # Check answer quality
    is_insufficient = check_if_answer_insufficient(answer)
    is_incomplete = check_if_answer_incomplete(answer)
    
    # If no info found → don't extract sources
    if is_insufficient:
        cited_in_this_iteration = []
    else:
        # Extract which sources LLM actually cited
        cited_in_this_iteration = extract_used_sources_from_answer(
            answer, current_iteration_chunks
        )

    # Add new cited sources to cumulative list
    updated_cumulative = cumulative_cited_sources.copy()
    for new_source in cited_in_this_iteration:
        # Check if already in list
        is_duplicate = False
        for existing in updated_cumulative:
            if (existing.get("source") == new_source.get("source") and 
                existing.get("page") == new_source.get("page")):
                is_duplicate = True
                break
        
        if not is_duplicate:
            updated_cumulative.append(new_source)
            key = f"{new_source.get('source')}_{new_source.get('page')}"
            used_pages.add(key)

    # Answer is complete if it has info and is not partial
    is_complete = (
        (not is_insufficient) and 
        (not is_incomplete)
    )
    
    return updated_cumulative, is_complete

################################################################################
# Trims context to fit within token limit
# Called by: prepare_iteration_context
# Returns: trimmed context string
# Logic: If too long → remove chunks from end until it fits
def trim_context_to_fit(context_parts: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    total_text = "\n\n---\n\n".join(context_parts)
    current_tokens = estimate_tokens(total_text)
    
    if current_tokens <= max_tokens:
        return total_text
    
    # Remove chunks from end until fits
    while current_tokens > max_tokens and len(context_parts) > 1:
        context_parts.pop()
        total_text = "\n\n---\n\n".join(context_parts)
        current_tokens = estimate_tokens(total_text)
    
    return total_text

################################################################################