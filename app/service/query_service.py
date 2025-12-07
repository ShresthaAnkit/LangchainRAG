from langchain_core.vectorstores import VectorStore

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.prompt_manager import PromptManager
from app.schema.query import QueryResponse


class QueryService:
    def __init__(self):
        pass

    def query(
        self,
        query: str,
        llm: BaseChatModel,
        vectorstore: VectorStore,
        prompt_manager: PromptManager,
    ) -> QueryResponse:
        TEMPLATE = prompt_manager.get_prompt("query")

        prompt_template = PromptTemplate.from_template(TEMPLATE)

        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7, "score_threshold": 0.5}
        )
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])
        sources = [doc.model_dump() for doc in docs]

        chain = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({"query": query, "context": context})

        return QueryResponse(answer=response, sources=sources)
