from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import lancedb
from typing import List, Dict


class SFRRAG:
    def __init__(self):
        # 使用 BGE 高质量 Embedding 模型
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="为检索任务生成查询表示: "
        )
        # 重排序模型
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)
        self.db = lancedb.connect("./lancedb_sfr_rag")
        self.vectorstore = None
        self.documents = []

    def build_index(self, documents: List[str]):
        """构建高质量向量索引"""
        self.documents = documents
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = LanceDB.from_documents(
            docs,
            self.embeddings,
            connection=self.db,
            table_name="sfr_rag_docs"
        )

    def get_retriever_with_reranker(self, top_k: int = 5):
        """创建带重排序的检索器"""
        # 基础检索器
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        # 重排序压缩器
        reranker = CrossEncoderReranker(
            model=self.reranker_model,
            top_n=top_k
        )

        # 组合检索器
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

    def compress_context(self, query: str, documents: List[Document]) -> str:
        """上下文压缩"""
        doc_texts = "\n".join([f"[{i+1}] {doc.page_content}"
                              for i, doc in enumerate(documents)])
        prompt = ChatPromptTemplate.from_template(
            """提取以下文档中与问题相关的关键信息：
问题: {query}

文档:
{documents}

请返回压缩后的关键信息，保留文档编号以便引用："""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "documents": doc_texts})

    def generate_with_citations(self, query: str, context: str) -> str:
        """生成带引用的答案"""
        prompt = ChatPromptTemplate.from_template(
            """基于以下上下文回答问题，并标注引用来源[1][2]等。

上下文: {context}

问题: {query}

要求：
1. 准确回答问题
2. 在相关陈述后标注引用来源
3. 如果上下文不足以回答，请说明

答案："""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})

    def verify_answer(self, query: str, answer: str, documents: List[Document]) -> Dict:
        """验证答案质量"""
        doc_contents = [doc.page_content for doc in documents]
        prompt = ChatPromptTemplate.from_template(
            """评估以下答案的质量：
问题: {query}
答案: {answer}
参考文档: {documents}

评估维度(1-5分)：
1. 准确性：答案是否被文档支持
2. 完整性：答案是否全面回答了问题
3. 相关性：答案是否紧扣问题

返回格式: 准确性分数|完整性分数|相关性分数|总评"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "answer": answer,
            "documents": doc_contents
        })
        try:
            parts = response.split("|")
            return {
                "accuracy": int(parts[0].strip()),
                "completeness": int(parts[1].strip()),
                "relevance": int(parts[2].strip()),
                "summary": parts[3].strip() if len(parts) > 3 else ""
            }
        except:
            return {"accuracy": 3, "completeness": 3, "relevance": 3, "summary": ""}

    def query(self, question: str) -> Dict:
        """SFR-RAG 主流程"""
        # 1. 初始检索 + 重排序
        retriever = self.get_retriever_with_reranker(top_k=5)
        docs = retriever.invoke(question)

        # 2. 上下文压缩
        compressed_context = self.compress_context(question, docs)

        # 3. 生成带引用的答案
        answer = self.generate_with_citations(question, compressed_context)

        # 4. 质量验证
        quality = self.verify_answer(question, answer, docs)

        return {
            "answer": answer,
            "sources": [{"content": doc.page_content[:100]} for doc in docs],
            "quality": quality
        }


# 使用示例
sfr_rag = SFRRAG()
sfr_rag.build_index([
    "人工智能是计算机科学的一个分支...",
    "机器学习是AI的核心技术之一...",
    "深度学习使用神经网络进行学习..."
])
result = sfr_rag.query("什么是人工智能？")
print(f"答案: {result['answer']}")
print(f"质量评估: {result['quality']}")