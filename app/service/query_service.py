from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableWithMessageHistory
from langfuse.langchain import CallbackHandler
from langfuse import observe
from operator import itemgetter

from app.core.prompt_manager import PromptManager
from app.schema.query import QueryResponse, RAGResponse
from app.exception import QueryError
from app.core.logging_config import get_logger
from app.core.db import get_session_history
from langchain_tavily import TavilySearch

logger = get_logger(__name__)


class QueryService:
    def __init__(self):
        pass

    def _web_search(
        self,
        query: str,
        langfuse_handler: CallbackHandler,
    ) -> tuple[str, list[dict]]:
        logger.info("Performing Web Search")
        tavily_retriever = TavilySearch(k=5)
        web_docs = tavily_retriever.invoke(
            query, config={"callbacks": [langfuse_handler], "verbose": False}
        )
        formatted_docs = []
        for i, doc in enumerate(web_docs.get("results", [])):
            formatted_docs.append(
                f"Source ID: [{i + 1}]\nTitle: {doc.get('title', '')}\nContent: {doc.get('content', '')}"
            )

        context = "\n\n".join(formatted_docs)
        sources = [
            {
                "source_id": i + 1,
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
            }
            for i, doc in enumerate(web_docs.get("results", []))
        ]
        return context, sources

    def _vector_search(
        self, query, vectorstore: VectorStore, langfuse_handler: CallbackHandler
    ) -> tuple[str, list[dict]]:
        logger.info("Performing Vector Search")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "score_threshold": 0.5},
        )
        docs = retriever.invoke(query, config={"callbacks": [langfuse_handler]})
        formatted_docs = []

        for i, doc in enumerate(docs):
            formatted_docs.append(f"Source ID: [{i + 1}]\nContent: {doc.page_content}")

        context = "\n\n".join(formatted_docs)

        sources = [
            {"source_id": i + 1, **doc.model_dump()} for i, doc in enumerate(docs)
        ]
        return context, sources

    @observe()
    def query(
        self,
        query: str,
        session_id: str,
        llm: BaseChatModel,
        vectorstore: VectorStore,
        prompt_manager: PromptManager,
    ) -> QueryResponse:
        try:
            langfuse_handler = CallbackHandler()

            TEMPLATE_SYSTEM = prompt_manager.get_prompt("query_system")
            TEMPLATE_HUMAN = prompt_manager.get_prompt("query")

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", TEMPLATE_SYSTEM),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(TEMPLATE_HUMAN),
                ]
            )

            context, sources = self._vector_search(
                query=query,
                vectorstore=vectorstore,
                langfuse_handler=langfuse_handler,
            )
            if not sources:
                logger.info("No sources found through vector search")
                context, sources = self._web_search(
                    query=query,
                    langfuse_handler=langfuse_handler,
                )
                context_source = "web"
            else:
                context_source = "vectorstore"

            response = self._process_query(
                query, context, session_id, llm, langfuse_handler, prompt_template
            )

            if not response.found_answer and context_source == "vectorstore":
                logger.info(
                    "Answer not found in vectorstore continuing with web search"
                )
                context, sources = self._web_search(
                    query=query,
                    langfuse_handler=langfuse_handler,
                )
                response = self._process_query(
                    query, context, session_id, llm, langfuse_handler, prompt_template
                )

            return QueryResponse(answer=response.answer, sources=sources)
        except ValueError as e:
            raise QueryError(str(e)) from e
        except Exception as e:
            logger.exception("Error occurred during query processing")
            raise QueryError("An error occurred while processing the query.") from e

    def _process_query(
        self, query, context, session_id, llm, langfuse_handler, prompt_template
    ):
        chain = (
            {
                "context": itemgetter("context"),
                "query": itemgetter("query"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt_template
            | llm.with_structured_output(RAGResponse)
        )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="query",
            history_messages_key="chat_history",
        )

        response = chain_with_history.invoke(
            {"query": query, "context": context},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [langfuse_handler],
                "verbose": False,
            },
        )

        return response
