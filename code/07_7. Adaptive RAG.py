from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import lancedb
from enum import Enum
from typing import List


class QueryType(Enum):
    SIMPLE = "simple"           # 简单事实查询
    MULTI_HOP = "multi_hop"     # 多跳推理查询
    OPEN_ENDED = "open_ended"   # 开放性问题
    NO_RETRIEVAL = "no_retrieval"  # 不需要检索


class AdaptiveRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        # 使用轻量级的 all-MiniLM-L6-v2 模型，仅 80MB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = lancedb.connect("/tmp/lancedb_adaptive_rag")
        self.vectorstore = None

    def build_index(self, documents: List[str]):
        """构建向量索引"""
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = LanceDB.from_documents(
            docs,
            self.embeddings,
            connection=self.db,
            table_name="adaptive_rag_docs"
        )

    def classify_query(self, query: str) -> QueryType:
        """分类查询类型"""
        prompt = ChatPromptTemplate.from_template(
            """分析以下查询的类型，返回对应类别：
查询: {query}

类别说明:
- SIMPLE: 简单的事实性问题，可以直接从单个文档找到答案
- MULTI_HOP: 需要综合多个信息源进行推理的复杂问题
- OPEN_ENDED: 开放性问题，需要广泛的知识和创造性思考
- NO_RETRIEVAL: 通用知识问题，不需要检索即可回答

只返回类别名称(SIMPLE/MULTI_HOP/OPEN_ENDED/NO_RETRIEVAL):"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query}).strip().upper()
        mapping = {
            "SIMPLE": QueryType.SIMPLE,
            "MULTI_HOP": QueryType.MULTI_HOP,
            "OPEN_ENDED": QueryType.OPEN_ENDED,
            "NO_RETRIEVAL": QueryType.NO_RETRIEVAL
        }
        return mapping.get(response, QueryType.SIMPLE)

    def simple_rag(self, query: str) -> str:
        """简单RAG：单次检索"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        prompt = ChatPromptTemplate.from_template(
            "基于以下内容回答问题：\n{context}\n\n问题：{question}\n答案："
        )

        def format_docs(docs):
            return "\n".join([d.page_content for d in docs])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def multi_hop_rag(self, query: str, max_hops: int = 3) -> str:
        """多跳RAG：迭代检索"""
        accumulated_context = []
        current_query = query
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        for hop in range(max_hops):
            # 检索
            docs = retriever.invoke(current_query)
            accumulated_context.extend([d.page_content for d in docs])

            # 检查是否已有足够信息
            context = "\n".join(accumulated_context)
            check_prompt = ChatPromptTemplate.from_template(
                """基于当前收集的信息，判断是否足够回答问题。
收集的信息: {context}
问题: {query}

回答YES如果信息足够，回答NO如果需要更多信息。
如果回答NO，请提供下一步应该搜索的子问题。
格式: YES 或 NO|子问题"""
            )
            check_chain = check_prompt | self.llm | StrOutputParser()
            check_response = check_chain.invoke({"context": context, "query": query})

            if check_response.strip().upper().startswith("YES"):
                break
            elif "|" in check_response:
                current_query = check_response.split("|")[1].strip()

        # 生成最终答案
        final_context = "\n".join(accumulated_context)
        final_prompt = ChatPromptTemplate.from_template(
            "综合以下信息回答问题：\n{context}\n\n问题：{question}\n答案："
        )
        final_chain = final_prompt | self.llm | StrOutputParser()
        return final_chain.invoke({"context": final_context, "question": query})

    def open_ended_rag(self, query: str) -> str:
        """开放性RAG：广泛检索+创造性生成"""
        # 扩展查询
        expand_prompt = ChatPromptTemplate.from_template(
            "为以下问题生成3个相关的搜索查询：\n{query}\n查询列表："
        )
        expand_chain = expand_prompt | self.llm | StrOutputParser()
        expanded = expand_chain.invoke({"query": query})
        queries = [query] + [q.strip() for q in expanded.split("\n") if q.strip()][:3]

        # 多查询检索
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend([d.page_content for d in docs])

        # 去重
        unique_docs = list(set(all_docs))
        context = "\n".join(unique_docs[:5])

        final_prompt = ChatPromptTemplate.from_template(
            """基于以下信息，对问题给出全面、有见地的回答：
信息: {context}
问题: {question}

请提供详细的分析和见解："""
        )
        final_chain = final_prompt | self.llm | StrOutputParser()
        return final_chain.invoke({"context": context, "question": query})

    def no_retrieval_generate(self, query: str) -> str:
        """直接生成：不使用检索"""
        prompt = ChatPromptTemplate.from_template("请回答：{query}")
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def query(self, question: str) -> str:
        """自适应查询主流程 - 使用 LangChain 路由"""
        # 1. 分类查询
        query_type = self.classify_query(question)
        print(f"查询类型: {query_type.value}")

        # 2. 路由到对应策略
        routing_map = {
            QueryType.SIMPLE: self.simple_rag,
            QueryType.MULTI_HOP: self.multi_hop_rag,
            QueryType.OPEN_ENDED: self.open_ended_rag,
            QueryType.NO_RETRIEVAL: self.no_retrieval_generate
        }
        return routing_map[query_type](question)


# 使用示例
arag = AdaptiveRAG()
arag.build_index(["公司财报数据...", "市场分析报告...", "行业趋势..."])
answer = arag.query("分析公司未来的发展前景")  # 会被识别为 OPEN_ENDED
print(answer)