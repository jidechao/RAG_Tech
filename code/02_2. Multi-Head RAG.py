from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer
import torch
import lancedb
from typing import List


class MultiHeadEmbeddings(Embeddings):
    """自定义多头注意力Embedding，继承LangChain的Embeddings基类"""

    def __init__(self, model_name="bert-base-uncased", head_index=0, num_heads=12):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.head_index = head_index
        self.num_heads = num_heads
        self.head_dim = 768 // num_heads  # BERT hidden size / num_heads

    def _get_head_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取指定头的embedding"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-2]  # 倒数第二层
        start = self.head_index * self.head_dim
        end = (self.head_index + 1) * self.head_dim
        head_emb = hidden_states[:, 0, start:end].numpy()
        return head_emb.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_head_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_head_embedding([text])[0]


class MultiHeadRAG:
    def __init__(self, num_heads=12):
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)
        self.num_heads = num_heads
        self.db = lancedb.connect("/tmp/lancedb_multihead_rag")
        self.vectorstores = []  # 每个头一个向量存储
        self.documents = []

    def build_index(self, documents: List[str]):
        """为每个头构建独立的LanceDB向量存储"""
        self.documents = documents
        docs = [Document(page_content=d) for d in documents]

        for head_idx in range(self.num_heads):
            embeddings = MultiHeadEmbeddings(head_index=head_idx, num_heads=self.num_heads)
            vectorstore = LanceDB.from_documents(
                docs,
                embeddings,
                connection=self.db,
                table_name=f"head_{head_idx}_docs"
            )
            self.vectorstores.append(vectorstore)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """多头并行检索并融合结果"""
        all_results = set()
        for vectorstore in self.vectorstores:
            docs = vectorstore.similarity_search(query, k=top_k)
            for doc in docs:
                all_results.add(doc.page_content)
        return list(all_results)

    def query(self, question: str) -> str:
        """检索并生成答案"""
        retrieved_docs = self.search(question)
        context = "\n\n".join(retrieved_docs)

        from langchain.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            """基于以下多维度检索的上下文回答问题：
上下文: {context}
问题: {question}
答案:"""
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return response.content


# 使用示例
mrag = MultiHeadRAG(num_heads=12)
documents = ["文档1的内容...", "文档2的内容...", "文档3的内容..."]
mrag.build_index(documents)
answer = mrag.query("查询问题")
print(answer)