import networkx as nx
import openai
import ast
from typing import List, Dict, Optional
from CMeKG_ACT import *
import os

### step 1 抽取历史和当前对话中的实体，并保留时间戳
"""
从包含历史对话和当前query的list中使用GPT-4o抽取实体，并构建字典保留每个实体的时间戳
时间戳定义：
- 当前对话中抽取的实体为0
- 历史对话抽取实体的时间戳为据当前对话的轮次
- 实体按最新的时间戳记录（如果同一实体在多个对话中出现，保留最小的时间戳）
"""
def init_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """
    初始化OpenAI客户端
    
    Args:
        api_key: OpenAI API密钥
        base_url: API基础URL
    
    Returns:
        OpenAI客户端实例
    """
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def extract_entities_with_gpt4o(text: str, client: openai.OpenAI) -> List[str]:
    """
    使用GPT-4o从文本中抽取实体
    
    Args:
        text: 需要抽取实体的文本
        client: OpenAI客户端实例
    
    Returns:
        抽取到的实体列表，如果失败返回空列表
    """
    prompt = f"""
    你的任务是从以下文本中，尽可能全面地抽取出所有医学领域的专有名词或实体。

    请重点关注以下类别的实体：
    - **疾病与诊断**: 疾病, 症状, 体征, 并发症, 诊断方法, 病理分型, 发病部位
    - **治疗与药物**: 治疗方法, 治疗方案, 药物, 手术, 适应症, 不良反应
    - **病因与检查**: 病因, 发病机制, 检查项目, 辅助检查
    - **其他**: 所属科室, 高危人群, 预防措施

    返回格式要求：
    1.  必须严格返回一个列表。
    2.  列表中包含所有抽取的实体字符串。
    3.  如果文本中没有发现任何符合条件的医学实体，请返回一个空列表 `[]`。

    ---
    示例文本1: "患者因头痛、发热入院，诊断为病毒性感冒，建议服用布洛芬治疗。"
    预期返回: ["头痛", "发热", "病毒性感冒", "布洛芬"]
    示例文本2: "12小时尿沉渣计数"
    预期返回: ["12小时尿沉渣计数"]
    ---

    需要处理的文本如下：
    ---
    {text}
    ---
    """
    max_retries = 3
    attempts = 0
    
    while attempts < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的医学实体抽取助手。你的任务是精准、全面地从文本中识别出所有医学实体，并严格按照用户要求的列表格式返回结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            
            # 检查响应是否有效
            if not response or not response.choices or len(response.choices) == 0:
                print(f"API响应无效 (尝试 {attempts + 1}/{max_retries})：响应为空或没有choices")
                attempts += 1
                if attempts < max_retries:
                    print("正在尝试重新生成...")
                continue
            
            # 检查content是否为None
            content = response.choices[0].message.content
            if content is None:
                print(f"API返回的content为None (尝试 {attempts + 1}/{max_retries})")
                attempts += 1
                if attempts < max_retries:
                    print("正在尝试重新生成...")
                continue
            
            content = content.strip()
            
            # 检查content是否为空
            if not content:
                print(f"API返回的content为空字符串 (尝试 {attempts + 1}/{max_retries})")
                attempts += 1
                if attempts < max_retries:
                    print("正在尝试重新生成...")
                continue
            
            result = ast.literal_eval(content)
            if isinstance(result, list):
                if all(isinstance(item, str) for item in result):
                    return result
                else:
                    print(f"格式错误：列表中的元素不全是字符串。内容：{content}")
            else:
                print(f"格式错误：返回的不是一个列表。内容：{content}")

        except (ValueError, SyntaxError) as e:
            content_str = content if 'content' in locals() and content is not None else 'None'
            print(f"解析错误 (尝试 {attempts + 1}/{max_retries})：返回内容无法解析为列表。错误：{e}。内容：'{content_str}'")
        except Exception as e:
            print(f"API调用异常 (尝试 {attempts + 1}/{max_retries})：{type(e).__name__}: {e}")
        
        attempts += 1
        if attempts < max_retries:
            print("正在尝试重新生成...")

    print(f"在尝试 {max_retries} 次后，仍然无法获取有效格式的结果。")
    return []


def extract_entities_with_timestamps(
    conversation_list: List[str],
    client: openai.OpenAI,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, int]:
    """
    从包含历史对话和当前query的list中抽取实体，并构建字典保留每个实体的时间戳
    
    Args:
        conversation_list: 包含历史对话和当前query的列表，最后一个元素是当前对话
        client: OpenAI客户端实例（如果为None，则使用api_key和base_url创建）
        api_key: OpenAI API密钥（仅在client为None时使用）
        base_url: API基础URL（仅在client为None时使用）
    
    Returns:
        字典，键为实体名称，值为该实体的最新时间戳
        时间戳规则：
        - 当前对话（最后一个元素）的实体时间戳为0
        - 历史对话的实体时间戳为距离当前对话的轮次（倒数第1个历史对话是1，倒数第2个是2，以此类推）
        - 如果同一实体在多个对话中出现，保留最新的时间戳（即较小的数字）
    
    Example:
        conversation_list = [
            "患者有高血压病史",
            "最近出现头痛症状",
            "需要检查血压"
        ]
        返回: {"高血压": 2, "头痛": 1, "血压": 0}
    """
    if not conversation_list:
        return {}
    
    # 如果client为None，尝试使用api_key和base_url创建
    if client is None:
        if api_key is None or base_url is None:
            raise ValueError("必须提供client或同时提供api_key和base_url")
        client = init_openai_client(api_key, base_url)
    
    # 存储实体和时间戳的字典
    # 存储实体和时间戳的字典
    entity_timestamps = {}
    
    # 从后往前遍历对话列表（从当前对话到历史对话）
    # 最后一个元素是当前对话（时间戳0），倒数第二个是历史对话1（时间戳1），以此类推
    total_conversations = len(conversation_list)
    
    for idx, conversation in enumerate(conversation_list):
        # 计算时间戳：当前对话（最后一个）为0，历史对话为距离当前对话的轮次
        timestamp = total_conversations - 1 - idx
        
        print(f"正在处理对话 {idx + 1}/{total_conversations} (时间戳: {timestamp})...")
        print(f"对话内容: {conversation[:50]}..." if len(conversation) > 50 else f"对话内容: {conversation}")
        
        # 抽取实体
        entities = extract_entities_with_gpt4o(conversation, client)
        
        # 更新实体时间戳字典
        # 如果实体已存在，只保留更小的时间戳（即更新的时间戳）
        for entity in entities:
            # 验证实体名称有效性（过滤空字符串和无效实体）
            if not entity or not isinstance(entity, str) or len(entity.strip()) == 0:
                print(f"  警告：跳过无效实体: {repr(entity)}")
                continue
            
            # 确保时间戳是整数
            if not isinstance(timestamp, int):
                print(f"  警告：时间戳类型错误，期望int，得到{type(timestamp).__name__}，已转换为整数")
                try:
                    timestamp = int(timestamp)
                except (ValueError, TypeError):
                    print(f"  错误：无法将时间戳转换为整数: {timestamp}")
                    continue
            
            if entity not in entity_timestamps or timestamp < entity_timestamps[entity]:
                entity_timestamps[entity] = timestamp
                print(f"  实体 '{entity}' 时间戳更新为: {timestamp}")
    
    # 最终验证：确保所有时间戳都是整数
    validated_timestamps = {}
    for entity, ts in entity_timestamps.items():
        if isinstance(ts, int):
            validated_timestamps[entity] = ts
        else:
            print(f"  警告：实体 '{entity}' 的时间戳 '{ts}' 不是整数，已跳过")
    
    return validated_timestamps

### step 2 将抽取的实体与现有知识图谱中实体进行对齐  （文本/语义）
def entity_alignment(WIKG, entities):
    entity2kgentity = {}
    kg_entities = []
    for entity in entities:
        entity2kgentity[entity] = []
        query = f"""
            MATCH (n) WHERE n.name='{entity}' RETURN n
        """
        entity_kg = WIKG.run_query(query)
        align_flag = False
        if len(entity_kg) >= 1:
            for e in entity_kg:
                if 'name' in e['n']:
                    entity2kgentity[entity].append(e['n']['name'])
                    kg_entities.append(e['n']['name'])
                    align_flag = True
                    break
        if not align_flag:
            #### 完全匹配找不到实体
            similar = WIKG.semantic_search(entity, top_k=3)
            if similar:
                entity2kgentity[entity] = [similar[0]['term']]
                kg_entities.append(similar[0]['term'])
    return entity2kgentity, kg_entities   ### 前者保留对话实体与上层知识实体的对齐，后者保留进行关系查询的实体

### step3 找到对齐实体对应的2跳关系 return 形式 {'entity1':['relation1', 'relation2']}
def kg_path_retrieval_CMeKG_OLD(WIKG, entity2kgentity, kg_entities, max_hops=2):
    '''
    不太适配原抽取代码，替换WIKG版本
    '''
    entity2relation = {}
    raw_results = []
    hops = max_hops
    # --- 常规邻居查询 ---
    if not kg_entities:
            return []

    triples = []
    print(f"🔍 开始精确匹配 (在 {hops} 跳内)...")
    if hops > 1:
        triples = WIKG.query_multihop_by_entities(kg_entities, max_hops=hops)
    else:
        triples = WIKG.query_by_entities(kg_entities)

        
    unique_triples = [dict(t) for t in {tuple(d.items()) for d in triples}]

    valid_paths = []
    seen_signatures = set()
    pattern_clean = re.compile(r'(?<!-)(?:\[.*?\]|\(.*?\))')
    
    for item in raw_results:
        p_str = item['path_string']
        cleaned_str = pattern_clean.sub('', p_str)
        cleaned_str = cleaned_str.strip()
        
        if cleaned_str in seen_signatures:
            continue
        
        seen_signatures.add(cleaned_str)
        valid_paths.append(cleaned_str)

    print(f"归一化去重后剩余 {len(valid_paths)} 条路径。", flush=True)

    time_b_end = time.time()
    ### 记录头实体  尾实体  谁是头谁是尾呢？？？？
    initial_entities = []
    end_entities = []
    ### 将检索到的路径与对话实体进行结合，以便于在RWR后加上时间戳权重
    for path in raw_results:  # 有点问题？# 一跳
        initial_entities.append(path['nodes_data'][0]['name'])
        end_entities.append(path['nodes_data'][1]['name'])
        if len(path['nodes_data']) > 2:  # 2跳关系
            initial_entities.append(path['nodes_data'][1]['name'])
            end_entities.append(path['nodes_data'][2]['name'])
    for d_entity, kg_entity in entity2kgentity.items():
        entity2relation[d_entity] = []
        for path in raw_results:
            # initial_entities.append(path['nodes_data'][0]['name'])
            # end_entities.append(path['nodes_data'][1]['name'])
            for nm in range(len(path['nodes_data'])):
                if path['nodes_data'][nm]['name'] in kg_entity:
                    entity2relation[d_entity].append(path)
                    break
    
    return entity2relation, initial_entities, end_entities

def kg_path_retrieval(WIKG, entity2kgentity, kg_entities, max_hops=2):
    entity2relation = {}
    raw_results = []
        
    # ==========================================
    # A. 检索阶段 (实体间路径 + 邻居路径)
    # ==========================================
    time_a_start = time.time()

    # --- A1. 优先查询实体间的路径 ---
    if len(kg_entities) > 1:
        path_visuals = WIKG.query_paths_between_entities_woMedicine_visual(kg_entities, max_hops=max_hops)
        if path_visuals:
            print(f"✅ 命中实体间路径: {len(path_visuals)} 条")
            raw_results.extend(path_visuals)
    
    time_a_end = time.time()

    # --- A2. 常规邻居查询 ---
    time_neighbor_start = time.time()
    neighbor_visuals = WIKG.query_multihop_paths_with_woMedicine_visual(kg_entities, max_hops=max_hops)
    raw_results.extend(neighbor_visuals)

    print(f"初始检索到 {len(raw_results)} 条路径。", flush=True)
    
    time_neighbor_end = time.time()

    # ==========================================
    # B. 归一化去重（药品-国药准字）
    # ==========================================
    time_b_start = time.time()

    valid_paths = []
    seen_signatures = set()
    pattern_clean = re.compile(r'(?<!-)(?:\[.*?\]|\(.*?\))')
    
    for item in raw_results:
        p_str = item['path_string']
        cleaned_str = pattern_clean.sub('', p_str)
        cleaned_str = cleaned_str.strip()
        
        if cleaned_str in seen_signatures:
            continue
        
        seen_signatures.add(cleaned_str)
        valid_paths.append(cleaned_str)

    print(f"归一化去重后剩余 {len(valid_paths)} 条路径。", flush=True)

    time_b_end = time.time()
    ### 记录头实体  尾实体  谁是头谁是尾呢？？？？
    initial_entities = []
    end_entities = []
    ### 将检索到的路径与对话实体进行结合，以便于在RWR后加上时间戳权重
    for path in raw_results:  # 有点问题？# 一跳
        initial_entities.append(path['nodes_data'][0]['name'])
        end_entities.append(path['nodes_data'][1]['name'])
        if len(path['nodes_data']) > 2:  # 2跳关系
            initial_entities.append(path['nodes_data'][1]['name'])
            end_entities.append(path['nodes_data'][2]['name'])
    for d_entity, kg_entity in entity2kgentity.items():
        entity2relation[d_entity] = []
        for path in raw_results:
            # initial_entities.append(path['nodes_data'][0]['name'])
            # end_entities.append(path['nodes_data'][1]['name'])
            for nm in range(len(path['nodes_data'])):
                if path['nodes_data'][nm]['name'] in kg_entity:
                    entity2relation[d_entity].append(path)
                    break
    
    return entity2relation, initial_entities, end_entities


### step4 RWR对关系打分 所有关系weight=1
def rwr_score(extract_entity_relation, kg_entities, entity2kgentity, initial_entities, end_entities, entity_timestamps, max_timestamp):
    """
    extract_entity_relation: {'entity1':['relation1', 'relation2']}
    kg_entities: 对话实体对应的上层知识实体
    entity2kgentity：对话实体与KG实体对齐
    initial_entities: 关系中的头实体 
    end_entities: 关系中的尾实体，与上一个一起构图
    entity_timestamps: 带时间戳的抽取实体
    max_timestamp: 最大的时间戳作为最大的权重
    """
    center_entity = list(set(kg_entities))
    initial_entity = initial_entities  # 头实体
    target_entity = end_entities   # 尾实体
    all_nodes = list(set(initial_entity + target_entity))

    ###  构建RWR所需图
    weight_edges = []
    for i in range(len(initial_entity)):
        weight_edges.append((initial_entity[i], target_entity[i], 1))

    # 1. 外部初始化一次图
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_weighted_edges_from(weight_edges)
    G.remove_nodes_from(list(nx.isolates(G))) # 移除孤立点防止 ZeroDivisionError

    # 2. 循环计算
    rwr_ = {}
    for entity in center_entity:
        if entity not in G:
            continue
        # 只计算当前实体的 RWR 分数
        # a = nx.pagerank(G, alpha=0.6, personalization={entity: 1})
        # rwr_[entity] = a 
        a = nx.pagerank(G, alpha=0.6, personalization={entity: 1})
        # 剔除中心实体本身，然后重新归一化
        if entity in a:
            # 保存中心实体的原始分数（如果需要的话）
            entity_self_score = a.pop(entity)
            # 重新归一化剩余节点的分数
            if a:  # 确保还有其他节点
                total_score = sum(a.values())
                if total_score > 0:
                    a = {node: score / total_score for node, score in a.items()}
        rwr_[entity] = a
    
    ###  对关系进行打分 rwr_[entity]记录了entity到与该实体有关的链接实体的分数，从中找到相关路径上对应实体的最大分数作为该路径的分数（要排除本身）,要乘上对应的时间戳
    # relation_score = {}
    # for dentity, relations in extract_entity_relation.items():
    #     for relation in relations:
    #         if frozenset(relation.items()) not in relation_score.keys():
    #             relation_score[frozenset(relation.items())] = 0
    #         re_entity = []  ###  改 从关系中抽出实体
    #         for node in relation['nodes_data']:
    #             re_entity.append(node['name'])
    #         relation_score_ = []   ## 记录该关系上对应实体的分数
    #         for rentity in re_entity:
    #             if (rentity in rwr_[entity2kgentity[dentity][0]]) and (rentity != entity2kgentity[dentity][0]):  # extract_entity_relation键为对话实体，这里要对应到KG实体
    #                 relation_score_.append(rwr_[entity2kgentity[dentity][0]][rentity])
            
    #         relation_score[frozenset(relation.items())] += (max_timestamp - entity_timestamps[dentity]) * max(relation_score_)
            
    relation_score = {}
    for dentity, relations in extract_entity_relation.items():
        for relation in relations:
            # 使用路径字符串作为唯一键
            path_key = relation['path_string']
            
            if path_key not in relation_score:
                relation_score[path_key] = 0
                
            re_entity = [node['name'] for node in relation['nodes_data']]
            
            relation_score_ = []
            center_kg_node = entity2kgentity[dentity][0]
            
            for rentity in re_entity:
                # 确保实体在 RWR 结果中且不是中心节点本身
                if rentity in rwr_[center_kg_node] and rentity != center_kg_node:
                    relation_score_.append(rwr_[center_kg_node][rentity])
            
            if relation_score_: # 确保列表不为空
                current_score = (max_timestamp - entity_timestamps[dentity]) * max(relation_score_)
                relation_score[path_key] += current_score
    return relation_score

### step5 使用ACT对路径打分重排序 选取top-k 
def relation_semantic_base(WIKG, relation_score, query):
    """
    记录当前对话和检索路径的语义相似度，作为ACT的base分数
    """
    for rk in relation_score.keys():
        query_emb = WIKG.encode_model.encode(query, convert_to_tensor=True)
        path_embs = WIKG.encode_model.encode(rk, convert_to_tensor=True)
        
        score = util.cos_sim(query_emb, path_embs)[0] ### 看一下这里是否对
        relation_score[rk] += score

    ### 依据分数排序
    sorted_relation_score = sorted(relation_score.items(), key=lambda x: x[1], reverse=True)  ## 分数由大到小
    final_path = []
    num = 0
    for k,v in sorted_relation_score:
        if num < 50:
            final_path.append(k)
            num += 1
    
    return final_path


def main():
    """
    主函数：示例用法
    """
    # 配置OpenAI客户端（使用与项目其他文件相同的配置）
    API_KEY = "xxxx"
    BASE_URL = "xxx"
    
    client = init_openai_client(API_KEY, BASE_URL)

    ## 外部知识图谱
    NODE_EMBS_PATH = "../CMeKG/cmekg_node_embeddings.pt"

    # --- 首次运行或数据库更新后，生成嵌入文件 ---
    if not os.path.exists(NODE_EMBS_PATH):
        print("--- 首次运行：开始生成节点嵌入文件 ---")
        retriever_gen = CMeKGRetriever(load_embeddings_on_init=False)
        retriever_gen.precompute_and_save_embeddings()
        retriever_gen.close()
        print("--- 嵌入文件生成完毕 ---")

    # --- 查询示例 ---
    if os.path.exists(NODE_EMBS_PATH):
        external_kg = CMeKGRetriever()
        search_entity_via_semantic = True  ### 暂时测试

        ### 读取对话文件，包括MMD的训练和测试，CMtMedQA的训练和测试
        ### 在对知识路径进行计算重排后，直接保存到文件中，key为'related_kg_path=[]'
        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path))
        grand_path = os.path.abspath(os.path.dirname(father_path))  
        gfrand_path = os.path.abspath(os.path.dirname(grand_path))
        data_path = os.path.join(gfrand_path, 'Data', 'CMtMedQA_train_sampled400.json')
        save_path = os.path.join(gfrand_path, 'Data', 'CMtMedQA_train_sampled400_CMeKGpath_gpt4o_mini_alpha_06.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            dialog_data = json.load(f)
        for i in range(len(dialog_data)):
            # if i > 10:
            #     break
            if 'related_kg_path' in dialog_data[i].keys():
                continue
            print('==' * 20)
            print(f'deal with {i} sample')
            print('==' * 20)
            dialog_history = dialog_data[i]['history']
            current_q = dialog_data[i]['current_q']
            conversation_list = dialog_history + [current_q]
        
        # 示例：包含历史对话和当前query的列表
        # conversation_list = [
        #     "患者有高血压病史，长期服用降压药",
        #     "最近出现头痛、头晕症状，血压控制不佳",
        #     "需要检查血压和心电图，调整用药方案"
        # ]
        # for i in range(len(conversation_list)):
        
            print("=" * 60)
            print(f"开始抽取第{i}个样本的实体并构建时间戳字典...")
            print("=" * 60)
            
            # step1 抽取实体并获取时间戳字典
            entity_timestamps = extract_entities_with_timestamps(conversation_list, client)
            # entity_timestamps = {'高血压': 2, '降压药': 2, '头痛': 1, '头晕': 1, '血压控制不佳': 1, '血压': 0, '心电图': 0, '用药方案': 0}  # 测试数据
            print("\n" + "=" * 60)
            print("抽取结果：")
            print("=" * 60)
            print(f"共抽取到 {len(entity_timestamps)} 个唯一实体：\n")
            # 按时间戳排序输出
            sorted_entities = sorted(entity_timestamps.items(), key=lambda x: x[1])
            ##  如果抽取出的实体为0，依据q与KG实体的语义相似度找
            if len(sorted_entities) == 0 and search_entity_via_semantic:
                print("实体抽取为空，尝试语义搜索入口节点...")
                similar = external_kg.semantic_search(conversation_list[-1], top_k=3)
                entities = []
                if similar:
                    entities = [similar[0]['term']]
                
                # 保持 sorted_entities 为列表格式（元组列表）
                sorted_entities = []
                for e in entities:
                    if e:  # 确保实体不为空
                        sorted_entities.append((e, 0))  # 添加为元组 (entity, timestamp)
            
            max_weight = -1
            for entity, timestamp in sorted_entities:
                # 验证时间戳类型，确保是整数
                if not isinstance(timestamp, int):
                    print(f"  警告：实体 '{entity}' 的时间戳 '{timestamp}' 不是整数，尝试转换...")
                    try:
                        timestamp = int(timestamp)
                    except (ValueError, TypeError):
                        print(f"  错误：无法将实体 '{entity}' 的时间戳 '{timestamp}' 转换为整数，跳过该实体")
                        continue
                
                print(f"  实体: {entity:20s} | 时间戳: {timestamp}")
                max_weight = max(max_weight, timestamp) # 记录最大的时间戳，作为最大的权重
            
            # 如果所有时间戳都无效，设置默认值
            if max_weight == -1:
                print("  警告：所有时间戳都无效，使用默认最大权重 1")
                max_weight = 0
            max_weight += 1  # 避免0权重
            
            ### step2  对齐
            entity2kgentity, kg_entities = entity_alignment(external_kg, entity_timestamps.keys())

            ### step3 找到对齐实体对应的2跳关系
            entity2relation, initial_entities, end_entities = kg_path_retrieval(external_kg, entity2kgentity, kg_entities)

            ### step4 RWR打分
            relation_score = rwr_score(entity2relation, kg_entities, entity2kgentity, initial_entities, end_entities, entity_timestamps, max_weight)

            ### step5 加上baseline语义分数
            final_path = relation_semantic_base(external_kg, relation_score, conversation_list[-1])

            ### 将检索路径保存到原始文件中，避免每次重复查询
            ### 将检索路径保存到原始文件中，避免每次重复查询
            dialog_data[i]['related_kg_path'] = final_path
            dialog_data[i]['extract_dialog_entity'] = entity_timestamps
            if i % 10 == 0:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(dialog_data, f, ensure_ascii=False, indent=4)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dialog_data, f, ensure_ascii=False, indent=4)

main()




