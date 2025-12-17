import json
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langfuse import observe
from operator import itemgetter

from app.core.prompt_manager import PromptManager
from app.schema.query import QueryResponse, RAGResponse
from app.exception import QueryError
from app.core.logging_config import get_logger
from app.tools.query_tools import vector_search_tool, web_search_tool
from app.core.db import get_session_history
from app.core.config import settings

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
        tavily_retriever = TavilySearch(k=settings.WEB_SEARCH_TOP_K)
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
                "metadata": {
                    "source": doc.get("url", ""),
                    "title": doc.get("title", ""),
                },
                "type": "websearch",
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
            search_kwargs={
                "k": settings.VECTOR_SEARCH_TOP_K,
                "score_threshold": settings.VECTOR_SEARCH_SIMILARITY_THRESHOLD,
            },
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

            response = self._process_query(
                query, context, session_id, llm, langfuse_handler, prompt_template
            )

            if not response.found_answer:
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
        self,
        query: str,
        context: str,
        session_id: str,
        llm: BaseChatModel,
        langfuse_handler: CallbackHandler,
        prompt_template: ChatPromptTemplate,
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

    @observe()
    def query_agentic(
        self,
        query: str,
        session_id: str,
        llm: BaseChatModel,
        vectorstore: VectorStore,
        prompt_manager: PromptManager,
    ) -> QueryResponse:
        """
        Agentic query function using VectorSearch and WebSearch tools.
        Fully streaming-ready with Langfuse tracking (TTFT, latency).
        """
        try:
            # Initialize Langfuse handler
            langfuse_handler = CallbackHandler()

            # Build prompt template
            TEMPLATE_SYSTEM = prompt_manager.get_prompt("query_system")

            history_obj = get_session_history(session_id)
            past_messages = history_obj.messages
            
            # Create the new user message
            new_message = HumanMessage(content=query)
            
            # Combine them: [Old History] + [New Query]
            # We pass this entire list to the agent.
            input_messages = past_messages + [new_message]

            # Define tools
            vector_tool = Tool(
                name="VectorSearch",
                func=lambda q: vector_search_tool(
                    query=q, vectorstore=vectorstore, langfuse_handler=langfuse_handler
                ),
                description="Searches documents in the vectorstore and returns context and sources",
            )

            web_tool = Tool(
                name="WebSearch",
                func=lambda q: web_search_tool(
                    query=q, langfuse_handler=langfuse_handler
                ),
                description="Searches the web and returns top-K results with content and URLs",
            )

            tools = [vector_tool, web_tool]

            # Initialize agent (conversational-reactive agent allows LLM to pick tools)
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=TEMPLATE_SYSTEM,
            )

            # Run the agent
            response_state = agent.invoke(
                {"messages": input_messages},
                config={
                    "callbacks": [langfuse_handler],
                },
            )

            final_answer = response_state['messages'][-1].content
            try:
                sources = json.loads(response_state['messages'][-2].model_dump()['content'])[1]
            except Exception as e:
                logger.error(f"Failed to parse sources from agent response: {e}")
                sources = []

            history_obj.add_user_message(query)
            history_obj.add_ai_message(final_answer)
            return QueryResponse(answer=final_answer, sources=sources)
        except ValueError as e:
            raise QueryError(str(e)) from e
        except Exception as e:
            logger.exception("Error occurred during query processing")
            raise QueryError("An error occurred while processing the query.") from e
