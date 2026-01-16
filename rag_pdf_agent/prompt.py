"""
Prompt templates for grounded RAG responses.
"""

SYSTEM_INSTRUCTION = """You are a STRICTLY GROUNDED document assistant. You MUST answer ONLY from the retrieved document chunks provided below.

ABSOLUTE RULES (NO EXCEPTIONS):
1. Answer ONLY using information EXPLICITLY present in the retrieved chunks
2. NEVER use your general knowledge, training data, or outside information
3. Do NOT infer, estimate, calculate, or assume ANY information
4. If the answer is NOT explicitly in the chunks, respond EXACTLY: "Not found in the document."
5. For numeric questions: Quote EXACT numbers as written in the document
6. NEVER approximate, round, or calculate numbers
7. Every factual statement MUST have a citation

MANDATORY RESPONSE FORMAT:

Answer:
<short, direct answer using only document text>

Citations:
[pX:cY], [pA:cB]

Evidence:
[pX:cY] "<exact quote from chunk>"
[pA:cB] "<exact quote from chunk>"

If information is missing, ambiguous, or requires inference:
Respond EXACTLY: "Not found in the document."

Do NOT deviate from this format or these rules under ANY circumstances."""


def build_grounded_prompt(query: str, retrieved_chunks: list, conversation_history: list = None) -> str:
    """
    Build a prompt that grounds the answer in retrieved chunks.
    
    Args:
        query: User's current question
        retrieved_chunks: List of (PDFChunk, score) tuples
        conversation_history: List of previous (user_msg, assistant_msg) tuples
        
    Returns:
        Formatted prompt string
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
        citation = chunk.get_citation()
        context_parts.append(
            f"[CHUNK {i}] {citation}\n"
            f"Page {chunk.page_number}\n"
            f"Text: {chunk.text}\n"
        )
    
    context = "\n---\n".join(context_parts)
    
    # Build conversation context if history exists
    history_text = ""
    if conversation_history:
        history_parts = []
        for user_msg, assistant_msg in conversation_history[-3:]:  # Last 3 turns
            history_parts.append(f"User: {user_msg}")
            history_parts.append(f"Assistant: {assistant_msg}")
        history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(history_parts)
    
    prompt = f"""RETRIEVED DOCUMENT CHUNKS:
{context}

{history_text}

CURRENT USER QUESTION:
{query}

INSTRUCTIONS:
Answer the question using ONLY the information in the retrieved chunks above. 

If the answer is not explicitly in the chunks, respond with EXACTLY:
"Not found in the document."

If you can answer:
1. Provide a clear, concise answer
2. Include citations using [p{page}:c{chunk_id}] format
3. Include an "Evidence" section with relevant quotes

Your response:"""
    
    return prompt


def rewrite_query_with_history(query: str, conversation_history: list) -> str:
    """
    Rewrite a follow-up question to be standalone using conversation history.
    
    Args:
        query: Current user query (might be a follow-up)
        conversation_history: List of previous (user_msg, assistant_msg) tuples
        
    Returns:
        Rewritten standalone query
    """
    if not conversation_history:
        return query
    
    # Build recent history context
    history_parts = []
    for user_msg, assistant_msg in conversation_history[-2:]:  # Last 2 turns
        history_parts.append(f"User: {user_msg}")
        history_parts.append(f"Assistant: {assistant_msg}")
    
    history_text = "\n".join(history_parts)
    
    rewrite_prompt = f"""Given the conversation history below, rewrite the follow-up question to be a standalone question that captures the full context.

CONVERSATION HISTORY:
{history_text}

FOLLOW-UP QUESTION:
{query}

INSTRUCTIONS:
- If the question refers to previous context (e.g., "it", "they", "that company"), incorporate that context
- If the question is already standalone, return it as-is
- Keep it concise and clear
- ONLY rewrite for clarity - do NOT add facts or assumptions

STANDALONE QUESTION:"""
    
    return rewrite_prompt


# Response parsing helpers
def parse_response_for_citations(response: str) -> list:
    """
    Extract citations from a response.
    
    Args:
        response: Generated response text
        
    Returns:
        List of citation strings like ['p5:c12', 'p7:c25']
    """
    import re
    pattern = r'\[p(\d+):c(\d+)\]'
    matches = re.findall(pattern, response)
    return [f"p{page}:c{chunk}" for page, chunk in matches]


def is_not_found_response(response: str) -> bool:
    """
    Check if response indicates information was not found.
    
    Args:
        response: Generated response text
        
    Returns:
        True if response indicates not found
    """
    response_lower = response.lower().strip()
    not_found_phrases = [
        "not found in the document",
        "not mentioned in the document",
        "does not contain",
        "information is not available"
    ]
    return any(phrase in response_lower for phrase in not_found_phrases)
