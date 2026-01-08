from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import lancedb
from typing import List


class CorrectiveRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = lancedb.connect("/tmp/lancedb_corrective_rag")
        self.vectorstore = None
        # 使用 Tavily 进行网络搜索
        self.web_search = TavilySearchResults(max_results=3)

    def build_index(self, documents: List[str]):
        """构建向量索引"""
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = LanceDB.from_documents(
            docs,
            self.embeddings,
            connection=self.db,
            table_name="corrective_rag_docs"
        )

    def evaluate_relevance(self, query: str, document: str) -> str:
        """评估文档与查询的相关性"""
        prompt = ChatPromptTemplate.from_template(
            """评估以下文档与查询的相关性。
查询: {query}
文档: {document}

请回答: CORRECT(相关), INCORRECT(不相关), 或 AMBIGUOUS(模糊)
只返回一个词。"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "document": document})
        return response.strip().upper()

    def search_web(self, query: str) -> List[str]:
        """当本地文档不足时进行网络搜索"""
        try:
            results = self.web_search.invoke(query)
            return [r["content"] for r in results if "content" in r]
        except:
            return []

    def retrieve_and_correct(self, query: str, top_k: int = 5) -> List[str]:
        """检索并修正文档"""
        # 1. 初始检索
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)

        # 2. 评估每个文档的相关性
        correct_docs = []
        need_web_search = True

        for doc in docs:
            relevance = self.evaluate_relevance(query, doc.page_content)
            if relevance == "CORRECT":
                correct_docs.append(doc.page_content)
                need_web_search = False
            elif relevance == "AMBIGUOUS":
                # 对模糊文档进行知识精炼
                refined = self.refine_document(query, doc.page_content)
                correct_docs.append(refined)

        # 3. 必要时进行网络搜索补充
        if need_web_search or len(correct_docs) < 2:
            web_results = self.search_web(query)
            correct_docs.extend(web_results)

        return correct_docs

    def refine_document(self, query: str, document: str) -> str:
        """精炼文档，提取与查询相关的部分"""
        prompt = ChatPromptTemplate.from_template(
            """从以下文档中提取与查询最相关的信息：
查询: {query}
文档: {document}

请只返回相关的精炼内容："""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "document": document})

    def query(self, question: str) -> str:
        """生成最终答案"""
        corrected_docs = self.retrieve_and_correct(question)
        context = "\n\n".join(corrected_docs)

        prompt = ChatPromptTemplate.from_template(
            """基于以下经过修正的上下文回答问题：
上下文: {context}
问题: {question}
答案:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})


# 使用示例
crag = CorrectiveRAG()
crag.build_index(["文档1...", "文档2...", "文档3..."])
answer = crag.query("你的问题是什么？")
print(answer)