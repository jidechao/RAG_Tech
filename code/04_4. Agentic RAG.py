from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import lancedb
from typing import List


class AgenticRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        # 使用轻量级的 all-MiniLM-L6-v2 模型，仅 80MB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = lancedb.connect("/tmp/lancedb_agentic_rag")
        self.vectorstore = None
        self.agent_executor = None

    def build_index(self, documents: List[str]):
        """构建向量索引"""
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = LanceDB.from_documents(
            docs,
            self.embeddings,
            connection=self.db,
            table_name="agentic_rag_docs"
        )

    def setup_agent(self):
        """配置 Agent 和工具"""
        vectorstore = self.vectorstore  # 闭包引用

        @tool
        def semantic_search(query: str) -> str:
            """用于语义搜索，当需要理解问题含义并查找相关文档时使用"""
            docs = vectorstore.similarity_search(query, k=3)
            return "\n".join([d.page_content for d in docs])

        @tool
        def keyword_search(query: str) -> str:
            """用于关键词搜索，当需要精确匹配特定术语时使用"""
            docs = vectorstore.similarity_search(query, k=2)
            return "\n".join([d.page_content for d in docs])

        @tool
        def calculator(expression: str) -> str:
            """用于数学计算，输入数学表达式"""
            try:
                return str(eval(expression))
            except:
                return "计算错误"

        tools = [semantic_search, keyword_search, calculator]

        # 使用新版本的 Agent 提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能助手，可以使用工具来回答问题。

可用工具：
- semantic_search: 用于语义搜索，查找相关文档
- keyword_search: 用于关键词精确匹配
- calculator: 用于数学计算

请根据问题选择合适的工具，可以多次调用工具来获取完整信息。"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 创建 Tool Calling Agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def query(self, question: str) -> str:
        """执行查询"""
        if not self.agent_executor:
            self.setup_agent()
        result = self.agent_executor.invoke({"input": question})
        return result["output"]


# 使用示例
arag = AgenticRAG()
arag.build_index(["产品A价格100元...", "产品B价格200元...", "优惠政策..."])
answer = arag.query("产品A和产品B的总价是多少？有什么优惠？")
print(answer)