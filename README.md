# RAG 系统实现集合

本项目包含了8种不同的RAG（Retrieval-Augmented Generation）实现方式，展示了从基础到高级的各种RAG技术。

## 项目结构

- `01_1. Naive RAG.py` - 基础RAG实现
- `02_2. Multi-Head RAG.py` - 多头注意力RAG
- `03_3. Corrective RAG.py` - 纠错RAG
- `04_4. Agentic RAG.py` - 智能体RAG
- `05_5. Graph RAG.py` - 图RAG
- `06_6. Self RAG.py` - 自监督RAG
- `07_7. Adaptive RAG.py` - 自适应RAG
- `08_8. SFR RAG.py` - SFR（Search, Filter, Rerank）RAG

## 环境要求

- Python 3.8+
- OpenAI API Key（用于LLM调用）
- （可选）Neo4j数据库（用于Graph RAG）

## 安装步骤

1. 克隆或下载项目到本地
  
2. 安装依赖：
  
  ```bash
  pip install -r requirements.txt
  ```
  
3. 配置环境变量：
  创建 `.env` 文件，添加以下内容：
  
  ```
  OPENAI_API_KEY=your_openai_api_key_here
  TAVILY_API_KEY=your_tavily_api_key_here  # 用于Corrective RAG的网络搜索
  ```
  
4. （可选）配置Neo4j（仅Graph RAG需要）：
  

- 安装并启动Neo4j数据库
- 默认连接信息：`bolt://localhost:7687`
- 用户名：`neo4j`
- 密码：`password`
- 可在 `05_5. Graph RAG.py` 中修改连接参数

## 使用方法

### 基础使用

每个RAG实现都是独立的Python文件，可以直接运行：

```python
# 示例：运行Naive RAG
python "01_1. Naive RAG.py"
```

### 自定义使用

每个文件都包含一个类，可以导入并在自己的代码中使用：

```python
from "01_1. Naive RAG" import NaiveRAG

# 创建RAG实例
rag = NaiveRAG()

# 构建索引
documents = ["文档1内容...", "文档2内容...", "文档3内容..."]
rag.build_index(documents)

# 查询
answer = rag.query("你的问题")
print(answer)
```

## 各RAG实现说明

### 1. Naive RAG

最基础的RAG实现，使用向量检索和LLM生成答案。

### 2. Multi-Head RAG

使用BERT的多头注意力机制，从不同维度进行检索并融合结果。

### 3. Corrective RAG

在检索后评估文档相关性，必要时进行网络搜索补充。

### 4. Agentic RAG

使用LangChain Agent框架，智能选择检索策略和工具。

### 5. Graph RAG

基于知识图谱的RAG，提取实体和关系，进行图结构检索。

### 6. Self RAG

自监督RAG，自动评估检索需求和答案质量。

### 7. Adaptive RAG

根据查询类型自适应选择检索策略（简单/多跳/开放性问题）。

### 8. SFR RAG

Search-Filter-Rerank RAG，使用高质量embedding和重排序模型。

## 注意事项

1. **API密钥**：确保设置了OpenAI API密钥，部分实现还需要Tavily API密钥
2. **模型选择**：代码中使用了`gpt-5`和`gpt-4`，请根据实际情况修改为可用的模型
3. **向量数据库**：默认使用LanceDB，数据存储在`/tmp/`目录（Windows上可能需要修改路径）
4. **Neo4j**：Graph RAG需要Neo4j数据库，其他实现不需要
5. **设备配置**：默认使用CPU，如需GPU加速，修改`model_kwargs={"device": "cuda"}`

## 常见问题

### Q: 如何修改LLM模型？

A: 在每个文件的`__init__`方法中，修改`ChatOpenAI(model="...")`的参数。

### Q: 如何修改向量数据库路径？

A: 修改`lancedb.connect()`中的路径参数。

### Q: 如何启用GPU加速？

A: 在`HuggingFaceEmbeddings`的`model_kwargs`中设置`{"device": "cuda"}`。

## 许可证

本项目仅供学习和研究使用。
