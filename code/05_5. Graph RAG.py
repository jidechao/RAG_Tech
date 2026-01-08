from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import networkx as nx
from typing import List, Dict
import json


class GraphRAG:
    def __init__(self, neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j", neo4j_password="password"):
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)
        # 使用 LangChain 的 Neo4j 集成
        self.graph_db = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        self.nx_graph = nx.Graph()

    def extract_entities_and_relations(self, text: str) -> Dict:
        """使用 LLM 抽取实体和关系"""
        prompt = ChatPromptTemplate.from_template(
            """从以下文本中抽取实体和关系，返回 JSON 格式：
文本: {text}

返回格式（只返回 JSON）:
{{
    "entities": ["实体1", "实体2", ...],
    "relations": [["实体1", "关系", "实体2"], ...]
}}"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"text": text})
        try:
            return json.loads(response)
        except:
            return {"entities": [], "relations": []}

    def build_knowledge_graph(self, documents: List[str]):
        """构建知识图谱"""
        for doc in documents:
            extracted = self.extract_entities_and_relations(doc)

            # 添加到 NetworkX 图
            for entity in extracted["entities"]:
                self.nx_graph.add_node(entity)

            for rel in extracted["relations"]:
                if len(rel) == 3:
                    self.nx_graph.add_edge(rel[0], rel[2], relation=rel[1])

            # 存储到 Neo4j
            for entity in extracted["entities"]:
                self.graph_db.query(
                    "MERGE (e:Entity {name: $name})",
                    {"name": entity}
                )
            for rel in extracted["relations"]:
                if len(rel) == 3:
                    self.graph_db.query(
                        """MATCH (a:Entity {name: $from})
                           MATCH (b:Entity {name: $to})
                           MERGE (a)-[r:RELATED {type: $rel}]->(b)""",
                        {"from": rel[0], "to": rel[2], "rel": rel[1]}
                    )

    def detect_communities(self) -> List[List[str]]:
        """社区检测"""
        from networkx.algorithms import community
        if len(self.nx_graph.nodes()) == 0:
            return []
        communities = community.louvain_communities(self.nx_graph)
        return [list(c) for c in communities]

    def generate_community_summaries(self, communities: List[List[str]]) -> List[Dict]:
        """为每个社区生成摘要"""
        summaries = []
        for i, comm in enumerate(communities):
            subgraph = self.nx_graph.subgraph(comm)
            edges_info = [(u, v, d.get('relation', ''))
                          for u, v, d in subgraph.edges(data=True)]

            prompt = ChatPromptTemplate.from_template(
                """为以下实体群组生成简短摘要：
实体: {entities}
关系: {relations}
摘要:"""
            )
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"entities": comm, "relations": edges_info})
            summaries.append({"community": i, "entities": comm, "summary": summary})
        return summaries

    def query(self, question: str) -> str:
        """基于图的检索和回答"""
        # 1. 从问题中提取关键实体
        entities = self.extract_entities_and_relations(question)["entities"]

        # 2. 在 Neo4j 中查找相关子图
        graph_context = self.graph_db.query(
            """MATCH (e:Entity)-[r]-(related)
               WHERE e.name IN $entities
               RETURN e.name AS entity, type(r) AS rel_type, 
                      r.type AS relation, related.name AS related_entity
               LIMIT 20""",
            {"entities": entities}
        )

        # 3. 获取社区摘要
        communities = self.detect_communities()
        summaries = self.generate_community_summaries(communities[:3])

        # 4. 生成答案
        context = f"图关系: {graph_context}\n社区摘要: {summaries}"
        prompt = ChatPromptTemplate.from_template(
            """基于以下知识图谱信息回答问题：
{context}

问题: {question}
答案:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})


# 使用示例
grag = GraphRAG()
grag.build_knowledge_graph([
    "张三是ABC公司的CEO，该公司位于北京",
    "李四是ABC公司的CTO，他与张三是大学同学",
    "ABC公司开发了产品X，市场份额领先"
])
answer = grag.query("ABC公司的领导层有哪些人？")
print(answer)