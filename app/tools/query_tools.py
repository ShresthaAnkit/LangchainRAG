from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def vector_search_tool(
    query: str, vectorstore: VectorStore, langfuse_handler: BaseCallbackHandler
) -> tuple[str, list[dict]]:
    """
    Tool for LangChain agent or direct LLM call to perform a vector search.
    Returns:
        - context: concatenated content from top-K docs
        - sources: structured list of source metadata
    """
    logger.info("Performing Vector Search Tool call")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.VECTOR_SEARCH_TOP_K,
            "score_threshold": settings.VECTOR_SEARCH_SIMILARITY_THRESHOLD,
        },
    )

    # Invoke retriever with Langfuse callback for metrics
    docs: list[Document] = retriever.invoke(
        query, config={"callbacks": [langfuse_handler]}
    )

    # Format context for LLM
    formatted_docs = [
        f"Source ID: [{i + 1}]\nContent: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    context = "\n\n".join(formatted_docs)

    # Build structured source metadata
    sources = [{"source_id": i + 1, **doc.model_dump()} for i, doc in enumerate(docs)]

    return context, sources


def web_search_tool(
    query: str,
    langfuse_handler: BaseCallbackHandler,
) -> tuple[str, list[dict]]:
    """
    Tool for LangChain agent or direct LLM call to perform a web search.

    Returns:
        - context: concatenated top-K results for LLM
        - sources: structured metadata for each result
    """
    logger.info("Performing Web Search Tool call")

    tavily_retriever = TavilySearch(k=settings.WEB_SEARCH_TOP_K)

    # Invoke the web search with Langfuse callback
    web_docs: dict[str, list[dict]] = tavily_retriever.invoke(
        query, config={"callbacks": [langfuse_handler], "verbose": False}
    )

    results = web_docs.get("results", [])

    # Format context for LLM
    formatted_docs = [
        f"Source ID: [{i + 1}]\nTitle: {doc.get('title', '')}\nContent: {doc.get('content', '')}"
        for i, doc in enumerate(results)
    ]
    context = "\n\n".join(formatted_docs)

    # Build structured source metadata
    sources = [
        {
            "source_id": i + 1,
            "metadata": {
                "source": doc.get("url", ""),
                "title": doc.get("title", ""),
            },
            "type": "websearch",
        }
        for i, doc in enumerate(results)
    ]

    return context, sources
