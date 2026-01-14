# This file retrieves relevant chunks from the database
# Flow: API.py → search_chunks → returns text and table chunks
#       iteration.py → get_surrounding_pages_smart → returns nearby pages

MIN_CHUNK_TOKENS = 200
from .utils import estimate_tokens

################################################################################
# Main search function → queries database for relevant chunks
# Called by: API.answer_question at the start
# Returns: text chunks list, table chunks list
# Flow:
# 1. Searches for table chunks (max 40)
# 2. Searches for text chunks (max 80)
# 3. Filters out chunks that are too small (< 200 tokens)
def search_chunks(collection, query: str, n_text: int = 80, n_tables: int = 40):
    text_chunks = []
    table_chunks = []

    # Define two separate queries
    queries = [
        {
            "n_results": n_tables,
            "where": {"type": {"$in": ["table_with_context", "table"]}},
            "target": table_chunks,
            "min_tokens": None,  # Don't filter tables by size
            "default_type": "table"
        },
        {
            "n_results": n_text,
            "where": {"type": {"$nin": ["table_with_context", "table"]}},
            "target": text_chunks,
            "min_tokens": MIN_CHUNK_TOKENS,  # Filter small text chunks
            "default_type": "text"
        }
    ]

    # Run both queries
    for q in queries:
        try:
            # Query ChromaDB collection
            results = collection.query(
                query_texts=[query],
                n_results=q["n_results"],
                where=q["where"]
            )

            # Process results
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                # Skip small text chunks
                if q["min_tokens"] and estimate_tokens(doc) < q["min_tokens"]:
                    continue

                # Add to target list
                q["target"].append({
                    "content": doc,
                    "metadata": meta,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "N/A"),
                    "type": meta.get("type", q["default_type"])
                })
        except Exception:
            pass

    return text_chunks, table_chunks

################################################################################
# Gets pages surrounding cited chunks (pages before and after)
# Called by: iteration.get_next_chunk_batch when answer is incomplete
# Returns: list of surrounding chunks
# Logic:
# 1. For each cited chunk
# 2. Get pages N before and N after (N = pages_range)
# 3. Avoids duplicates using seen set
def get_surrounding_pages_smart(collection, cited_chunks: list, pages_range: int = 1) -> list:
    surrounding_chunks = []
    seen = set()
    
    for chunk in cited_chunks:
        source = chunk.get("source", "")
        page = chunk.get("page", "")
        
        # Handle merged pages (e.g., "3-5")
        if isinstance(page, str) and "-" in str(page):
            try:
                pages = [int(p) for p in str(page).split("-")]
                current_pages = list(range(pages[0], pages[-1] + 1))
            except:
                try:
                    current_pages = [int(page)]
                except:
                    continue
        else:
            try:
                current_pages = [int(page)]
            except:
                continue
        
        # For each current page, get surrounding pages
        for current_page in current_pages:
            for offset in range(-pages_range, pages_range + 1):
                # Skip the current page itself
                if offset == 0:
                    continue
                    
                target_page = current_page + offset
                
                # Skip negative pages
                if target_page < 1:
                    continue
                
                # Skip if already retrieved
                key = f"{source}_{target_page}"
                if key in seen:
                    continue
                seen.add(key)
                
                # Query database for this specific page
                try:
                    results = collection.query(
                        query_texts=["context retrieval"],
                        n_results=5,
                        where={
                            "$and": [
                                {"source": source},
                                {"page": target_page}
                            ]
                        }
                    )
                    
                    # Add results
                    if results["documents"][0]:
                        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                            surrounding_chunks.append({
                                "content": doc,
                                "metadata": meta,
                                "source": meta.get("source", "Unknown"),
                                "page": meta.get("page", "N/A"),
                                "type": meta.get("type", "text")
                            })
                except Exception as e:
                    continue
    
    return surrounding_chunks

################################################################################