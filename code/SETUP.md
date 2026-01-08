# 快速开始指南

## 1. 环境准备

### Python版本
确保已安装 Python 3.8 或更高版本：
```bash
python --version
```

### 创建虚拟环境（推荐）
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 配置API密钥

### 方法1：使用.env文件（推荐）

1. 复制环境变量示例文件：
```bash
# Windows PowerShell
Copy-Item env.example.txt .env

# Linux/Mac
cp env.example.txt .env
```

2. 编辑`.env`文件，填入你的API密钥：
```
OPENAI_API_KEY=sk-your-actual-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here  # 可选
```

### 方法2：直接设置环境变量

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-api-key"
```

## 4. 获取API密钥

### OpenAI API密钥
1. 访问 https://platform.openai.com/
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的API密钥

### Tavily API密钥（可选，仅Corrective RAG需要）
1. 访问 https://tavily.com/
2. 注册账号
3. 获取API密钥

## 5. （可选）配置Neo4j（仅Graph RAG需要）

### 安装Neo4j
1. 下载Neo4j Desktop: https://neo4j.com/download/
2. 安装并启动Neo4j
3. 创建新数据库
4. 设置密码（默认用户名：neo4j）

### 修改Graph RAG配置
编辑 `05_5. Graph RAG.py`，修改连接参数：
```python
grag = GraphRAG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password"
)
```

## 6. 运行示例

### 运行单个RAG实现
```bash
python "01_1. Naive RAG.py"
```

### 在代码中使用
```python
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件

# 现在可以使用环境变量
llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## 7. 常见问题排查

### 问题1：ModuleNotFoundError
**解决方案：** 确保已安装所有依赖
```bash
pip install -r requirements.txt
```

### 问题2：API密钥错误
**解决方案：** 
- 检查`.env`文件是否存在且格式正确
- 确认API密钥有效且有足够的余额
- 尝试直接设置环境变量

### 问题3：LanceDB路径错误（Windows）
**解决方案：** 修改代码中的路径，例如：
```python
# 将 /tmp/lancedb_xxx 改为 Windows路径
self.db = lancedb.connect("C:/temp/lancedb_naive_rag")
```

### 问题4：模型不存在（gpt-5）
**解决方案：** 将代码中的`gpt-5`改为`gpt-4`或`gpt-3.5-turbo`

### 问题5：Neo4j连接失败
**解决方案：**
- 确认Neo4j服务正在运行
- 检查连接URI、用户名和密码
- 确认防火墙设置允许连接

## 8. 下一步

- 阅读 `README.md` 了解各RAG实现的详细说明
- 查看各Python文件中的示例代码
- 根据需要修改模型和配置参数
