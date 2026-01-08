from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import lancedb


class NaiveRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)
        # 使用轻量级的 all-MiniLM-L6-v2 模型，仅 80MB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = lancedb.connect("/tmp/lancedb_naive_rag")
        self.vectorstore = None

    def build_index(self, documents: list):
        """构建向量索引"""
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = LanceDB.from_documents(
            docs,
            self.embeddings,
            connection=self.db,
            table_name="naive_rag_docs"
        )

    def query(self, question: str) -> str:
        """执行检索并生成答案"""
        # 创建检索器
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""基于以下上下文回答问题：
上下文: {context}
问题: {question}
答案:"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )

        return qa_chain.invoke({"query": question})["result"]


# 使用示例
naive_rag = NaiveRAG()
naive_rag.build_index(["文档1内容...", "文档2内容...", "文档3内容..."])
answer = naive_rag.query("What is issue date of lease?")
print(answer)