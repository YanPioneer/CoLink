import json
import torch
import ast
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import time
import re
import os

####  定义WIKG类  
####  以neo4j存储 进行初始化
####  定义编码实体的模型，作用：与抽取的对话实体进行语义对齐（除完全匹配外）
####  定义编码关系的模型，作用：计算检索到的路径与当前对话的相似度
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class WIKGRetriever2:
    def __init__(self, load_embeddings_on_init=True):
        MODEL_PATH = "../gte-multilingual-base/"
        NODE_EMBS_PATH = "../WI_Simple/wi_node_embeddings.pt"
        URI = "bolt://localhost:7687"
        AUTH = ("neo4j", "12345678")
        try:
            self.driver = GraphDatabase.driver(URI, auth=AUTH)
            self.driver.verify_connectivity()
            print("Neo4j 连接成功 ✅")
            self.check_and_create_indexes()
        except Exception as e:
            print(f"Neo4j 连接失败: {e}")
            raise
        print("正在加载句向量模型...")
        self.encode_model = SentenceTransformer(MODEL_PATH, device=device, trust_remote_code=True)
        self.node_embs_path = NODE_EMBS_PATH
        
        if load_embeddings_on_init:
            if os.path.exists(self.node_embs_path):
                print(f"正在加载预计算的节点嵌入文件: {self.node_embs_path}")
                saved_data = torch.load(self.node_embs_path, map_location=device, weights_only=False)
                self.node_names = saved_data['names']
                self.node_embeddings = saved_data['embeddings'].to(device)
                print(f"成功加载 {len(self.node_names)} 个节点的嵌入 ✅")
            else:
                raise FileNotFoundError(
                    f"错误：节点嵌入文件未找到: '{self.node_embs_path}'\n"
                    "请先运行一次本脚本以生成嵌入文件。"
                )
        else:
            self.node_names = None
            self.node_embeddings = None


    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j 连接已关闭。")


    def check_and_create_indexes(self):
        """
        【新知识图谱适配核心】
        由于 neo4j-admin import 默认只对 :ID (Name_Label) 建索引，
        但我们查询是用 text (n.name)，所以必须手动对所有 Label 的 name 属性建索引。
        """
        print("正在检查索引状态...")
        # 1. 获取数据库中所有的 Label
        labels_result = self.run_query("CALL db.labels()")
        labels = [r['label'] for r in labels_result]
        
        with self.driver.session() as session:
            for label in labels:
                # 针对每个 Label 创建 name 属性的索引
                query = f"CREATE INDEX index_{label}_name IF NOT EXISTS FOR (n:`{label}`) ON (n.name)"
                try:
                    session.run(query)
                except Exception as e:
                    print(f"创建索引警告 ({label}): {e}")
        print(f"已确保所有 ({len(labels)}) 个 Label 的 name 属性都有索引。")


    def _format_node(self, props, label):
        """
        将属性字典和标签合并为标准结构
        """
        if not props:
            props = {}
        
        p = dict(props)
        
        name = p.pop("name", "未知")

        # 注意：Label 可能为空，做个防护
        safe_label = label if label else "Unknown"
        
        return {
            "name": f"{name}..{safe_label}",
            "properties": p
        }
        # return {
        #     "name": name,
        #     "label": label if label else "Unknown",  # 把 Label 单独放出来
        #     "properties": p
        # }

    def run_query(self, query, params=None):
        """执行 Cypher 查询"""
        with self.driver.session() as session:
            result = session.run(query, params)
            return [r.data() for r in result]

    def precompute_and_save_embeddings(self):
        """
        从 Neo4j 数据库中获取所有节点名称，并计算、保存它们的嵌入。
        只需在数据库更新后运行一次。
        """
        print("开始预计算所有节点的嵌入...")
        # 获取所有拥有 'name' 属性的节点名称
        all_nodes = self.run_query("MATCH (n) RETURN DISTINCT n.name AS name")
        
        if not all_nodes:
            print("错误：数据库中没有找到任何带有 'name' 属性的节点。请确认数据已正确导入。")
            return

        unique_node_names = sorted([n['name'] for n in all_nodes if n['name']])
        print(f"从数据库获取了 {len(unique_node_names)} 个唯一的节点名称。开始编码...")
        
        node_embs = self.encode_model.encode(unique_node_names, convert_to_tensor=True, show_progress_bar=True)
        
        data_to_save = {
            'names': unique_node_names,
            'embeddings': node_embs
        }
        torch.save(data_to_save, self.node_embs_path)
        print(f"节点嵌入已计算并保存到 '{self.node_embs_path}' ✅")
        
        self.node_names = unique_node_names
        self.node_embeddings = node_embs


    def semantic_search(self, text, top_k=3):
        """使用向量相似度检索节点"""
        if self.node_embeddings is None:
            print("错误: 节点嵌入未加载。无法进行语义搜索。")
            return []
        print(f"语义搜索: '{text}'")
        query_emb = self.encode_model.encode(text, convert_to_tensor=True).to(device)
        scores = util.cos_sim(query_emb, self.node_embeddings)[0]
        
        top_scores, top_indices = scores.topk(top_k)
        
        results = []
        for i in range(top_k):
            index = top_indices[i].item()
            score = top_scores[i].item()
            results.append({'term': self.node_names[index], 'score': score})
            
        # print(f"语义检索结果: {results}")
        return results


    def query_paths_between_entities_woMedicine_visual(self, entities, max_hops=2):
        """
        [可视化版] 查询实体间的路径，直接返回拼接好的字符串
        """
        if len(entities) < 2:
            return []

        query = f"""
        MATCH (n), (m)
        WHERE n.name IN $entities AND m.name IN $entities AND id(n) < id(m)
        
        // 1. 匹配两点间所有 max_hops 内的路径
        MATCH path = (n)-[*1..{max_hops}]-(m)
        
        WHERE 
            // 3. 【新增】药品/药物 节点硬过滤
            // 只要路径中有一个节点是药品，直接剔除（防止通过药物建立无意义的关联）
            NONE(node IN nodes(path) WHERE 
                node.name CONTAINS '药品' OR 
                node.name CONTAINS '药物' OR 
                '药品' IN labels(node) OR 
                '药物' IN labels(node)
            )

        // =======================================================
        // 【核心优化 step 1】数据库内强力归一化去重
        // =======================================================
        // 逻辑：Name(厂)[号] -> split('[') -> Name(厂) -> split('(') -> Name
        
        WITH path,
             [node IN nodes(path) | 
                trim(split(split(node.name, '[')[0], '(')[0])
             ] AS clean_node_names,
             [rel IN relationships(path) | type(rel)] AS rel_types
        
        // 根据“清洗得只剩核心名称”的骨架进行去重，保留每组中的第一条
        WITH clean_node_names, rel_types, head(collect(path)) AS unique_path
        
        // =======================================================
        // 【核心优化 step 2】排序与截断
        // =======================================================
        // 对于实体间路径，我们通常最关心“最短路径”（直接关系），所以按长度 ASC 排序
        
        WITH unique_path
        ORDER BY length(unique_path) ASC
        LIMIT 50
        
        // =======================================================
        // 【输出拼接】
        // =======================================================
        
        WITH nodes(unique_path) AS ns, relationships(unique_path) AS rs
        
        RETURN reduce(s = head(ns).name, i IN range(0, size(rs)-1) | 
            s + 
            CASE 
                WHEN startNode(rs[i]) = ns[i] THEN ' -[' + type(rs[i]) + ']-> ' 
                ELSE ' <-[' + type(rs[i]) + ']- ' 
            END + 
            (ns[i+1]).name) AS path_string,
            ns AS nodes_data
        """
        return self.run_query(query, {"entities": entities})

    def query_multihop_paths_with_woMedicine_visual(self, entities, max_hops=2):
        """
        [可视化版] 查询多跳邻居路径（排除药品/药物）
        """
        if not entities:
            return []

        total_limit = 100
        limit_hop1 = int(total_limit * 0.6)
        limit_hop2 = int(total_limit * 0.4)
        
        query = f"""
        MATCH (start_node) WHERE start_node.name IN $entities
        MATCH path = (start_node)-[*1..{max_hops}]-(end_node)
        
        WHERE 
            start_node <> end_node
            
            // =======================================================
            // 【新增需求】 药品/药物 节点硬过滤
            // =======================================================
            // 含义：路径中【没有任何一个节点】的名字包含“药品/药物”，或者标签包含“药品/药物”
            // 这样可以彻底把 "疾病 -> 药物" 或 "药物 -> 疾病" 的路径都筛掉
            AND NONE(n IN nodes(path) WHERE 
                n.name CONTAINS '药品' OR 
                n.name CONTAINS '药物' OR 
                '药品' IN labels(n) OR 
                '药物' IN labels(n)
            )

        // =======================================================
        // 【核心优化 step 1】数据库内强力归一化去重
        // =======================================================
        
        WITH path,
             [n IN nodes(path) | 
                trim(split(split(n.name, '[')[0], '(')[0])
             ] AS clean_node_names,
             [r IN relationships(path) | type(r)] AS rel_types
        
        // 根据“清洗得只剩核心名称”的骨架进行去重
        WITH clean_node_names, rel_types, head(collect(path)) AS unique_path
        
        // =======================================================
        // 【核心优化 step 2】基于跳数的动态配额
        // =======================================================
        
        WITH unique_path, length(unique_path) AS hops
        WITH hops, collect(unique_path) AS paths_list
        
        WITH hops, 
             CASE hops
                 WHEN 1 THEN paths_list[0..$limit_h1]  
                 WHEN 2 THEN paths_list[0..$limit_h2]  
                 ELSE paths_list[0..10]
             END AS paths_slice
        
        UNWIND paths_slice AS path

        // =======================================================
        // 【输出】
        // =======================================================

        WITH nodes(path) AS ns, relationships(path) AS rs
        
        RETURN reduce(s = head(ns).name, i IN range(0, size(rs)-1) | 
            s + 
            CASE 
                WHEN startNode(rs[i]) = ns[i] THEN ' -[' + type(rs[i]) + ']-> ' 
                ELSE ' <-[' + type(rs[i]) + ']- ' 
            END + 
            (ns[i+1]).name) AS path_string,
            ns AS nodes_data
        """
        
        params = {
            "entities": entities, 
            "limit_h1": limit_hop1,
            "limit_h2": limit_hop2
        }
        
        return self.run_query(query, params)

