"""
配置管理模块
用于加载和管理环境变量
"""
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()


class Config:
    """配置类，统一管理所有配置项"""
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Tavily配置（用于Corrective RAG）
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    
    # Neo4j配置（用于Graph RAG）
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # LLM模型配置
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0"))
    
    # Embedding模型配置
    DEFAULT_EMBEDDING_MODEL = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cpu")  # cpu 或 cuda
    
    # 向量数据库配置
    LANCEDB_BASE_PATH = os.getenv("LANCEDB_BASE_PATH", "/tmp")
    
    @classmethod
    def validate(cls):
        """验证必需的配置项"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY未设置！请在.env文件中设置OPENAI_API_KEY，"
                "或参考env.example.txt文件。"
            )
        return True
    
    @classmethod
    def get_lancedb_path(cls, name: str) -> str:
        """获取LanceDB数据库路径"""
        return os.path.join(cls.LANCEDB_BASE_PATH, f"lancedb_{name}")


# 在导入时验证配置（可选，如果不想强制验证可以注释掉）
# Config.validate()
