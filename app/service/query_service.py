from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from app.core.prompt_manager import PromptManager
from app.schema.query import QueryResponse
from app.exception import QueryError
from app.core.logging_config import get_logger
from app.core.db import get_session_history

logger = get_logger(__name__)


class QueryService:
    def __init__(self):
        pass

    def query(
        self,
        query: str,
        session_id: str,
        llm: BaseChatModel,
        vectorstore: VectorStore,
        prompt_manager: PromptManager,
    ) -> QueryResponse:
        try:
            TEMPLATE_SYSTEM = prompt_manager.get_prompt("query_system")
            TEMPLATE_HUMAN = prompt_manager.get_prompt("query")

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", TEMPLATE_SYSTEM),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(TEMPLATE_HUMAN),
                ]
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "lambda_mult": 0.7, "score_threshold": 0.5},
            )
            docs = retriever.invoke(query)

            formatted_docs = []

            for i, doc in enumerate(docs):
                formatted_docs.append(
                    f"Source ID: [{i + 1}]\nContent: {doc.page_content}"
                )

            context = "\n\n".join(formatted_docs)

            sources = [
                {"source_id": i + 1, **doc.model_dump()} for i, doc in enumerate(docs)
            ]

            chain = (
                {
                    "context": itemgetter("context"),
                    "query": itemgetter("query"),
                    "chat_history": itemgetter("chat_history"),
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )

            chain_with_history = RunnableWithMessageHistory(
                runnable=chain,
                get_session_history=get_session_history,
                input_messages_key="query",
                history_messages_key="chat_history",
            )

            response = chain_with_history.invoke(
                {"query": query, "context": context},
                config={"configurable": {"session_id": session_id}},
            )

            return QueryResponse(answer=response, sources=sources)
        except ValueError as e:
            raise QueryError(str(e)) from e
        except Exception as e:
            logger.exception("Error occurred during query processing")
            raise QueryError("An error occurred while processing the query.") from e
