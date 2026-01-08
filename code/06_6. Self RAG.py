from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import lancedb
from typing import List, Tuple


class SelfRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        # 使用轻量级的 all-MiniLM-L6-v2 模型，仅 80MB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = lancedb.connect("/tmp/lancedb_self_rag")
        self.vectorstore = None

    def build_index(self, documents: List[str]):
        """构建向量索引"""
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = LanceDB.from_documents(
            docs,
            self.embeddings,
            connection=self.db,
            table_name="self_rag_docs"
        )

    def should_retrieve(self, query: str) -> bool:
        """判断是否需要检索 (Retrieve 标记)"""
        prompt = ChatPromptTemplate.from_template(
            """判断以下问题是否需要检索外部知识来回答。
问题: {query}

如果问题需要事实性知识、最新信息或特定领域知识，回答 YES。
如果问题是通用问题或推理问题，回答 NO。
只回答 YES 或 NO:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query}).strip().upper()
        return "YES" in response

    def evaluate_relevance(self, query: str, document: str) -> Tuple[bool, float]:
        """评估文档相关性 (ISREL 标记)"""
        prompt = ChatPromptTemplate.from_template(
            """评估文档与问题的相关性，打分 1-5 分。
问题: {query}
文档: {document}

返回格式: 分数|理由
示例: 4|文档直接回答了问题的核心内容"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "document": document})
        try:
            score = int(response.split("|")[0].strip())
            return score >= 3, score / 5.0
        except:
            return True, 0.6

    def evaluate_support(self, document: str, answer: str) -> Tuple[bool, float]:
        """评估答案是否被文档支持 (ISSUP 标记)"""
        prompt = ChatPromptTemplate.from_template(
            """评估答案是否被文档内容支持，打分 1-5 分。
文档: {document}
答案: {answer}

返回格式: 分数|理由"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"document": document, "answer": answer})
        try:
            score = int(response.split("|")[0].strip())
            return score >= 3, score / 5.0
        except:
            return True, 0.6

    def evaluate_usefulness(self, query: str, answer: str) -> Tuple[bool, float]:
        """评估答案有用性 (ISUSE 标记)"""
        prompt = ChatPromptTemplate.from_template(
            """评估答案对用户问题的有用程度，打分 1-5 分。
问题: {query}
答案: {answer}

返回格式: 分数|理由"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "answer": answer})
        try:
            score = int(response.split("|")[0].strip())
            return score >= 3, score / 5.0
        except:
            return True, 0.6

    def generate_with_context(self, query: str, context: str) -> str:
        """基于上下文生成答案"""
        prompt = ChatPromptTemplate.from_template(
            """基于以下上下文回答问题。如果上下文不足以回答，请说明。
上下文: {context}
问题: {query}
答案:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})

    def generate_without_context(self, query: str) -> str:
        """不使用检索直接生成"""
        prompt = ChatPromptTemplate.from_template("请回答以下问题: {query}")
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def query(self, question: str) -> str:
        """Self-RAG 主流程"""
        # 1. 检索决策
        need_retrieval = self.should_retrieve(question)

        if not need_retrieval:
            # 直接生成
            answer = self.generate_without_context(question)
            _, usefulness = self.evaluate_usefulness(question, answer)
            return answer

        # 2. 检索文档
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        # 3. 对每个文档生成候选答案并评分
        candidates = []
        for doc in docs:
            # 评估相关性 (ISREL)
            is_relevant, rel_score = self.evaluate_relevance(question, doc.page_content)
            if not is_relevant:
                continue

            # 生成答案
            answer = self.generate_with_context(question, doc.page_content)

            # 评估支持度 (ISSUP)
            is_supported, sup_score = self.evaluate_support(doc.page_content, answer)

            # 评估有用性 (ISUSE)
            is_useful, use_score = self.evaluate_usefulness(question, answer)

            # 综合评分
            total_score = rel_score * 0.3 + sup_score * 0.4 + use_score * 0.3
            candidates.append((answer, total_score))

        # 4. 选择最佳答案
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        else:
            # 如果没有合适的检索结果，直接生成
            return self.generate_without_context(question)


# 使用示例
srag = SelfRAG()
srag.build_index(["文档1内容...", "文档2内容...", "文档3内容..."])
answer = srag.query("你的问题是什么？")
print(answer)