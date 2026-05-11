import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import sys
import json
import torch
import re
from collections import defaultdict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from peft import PeftModel
sys.path.append("..")
sys.path.append("..")

# from WI.WI_KG import WIKGRetriever
# from WI_Simple.WI_Simple_KG import WIKGRetriever_Simple

# ===== 配置 =====
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
grand_path = os.path.abspath(os.path.dirname(father_path))
# DATA_PATH = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text_kgpath_gpt4o_mini_alpha_06_allkg_final.json')
# DATA_PATH = os.path.join(grand_path, 'Data', 'medical_data_v3_eval_kgpath_gpt4o_mini_alpha_06_allkg_final.json')
DATA_PATH = os.path.join(grand_path, 'Data', 'CMtMedQA_train_sampled400_WIKGpath_gpt4o_mini_alpha_06.json')
# DATA_PATH = os.path.join(grand_path, 'Data', 'CMtMedQA_test_sampled50_WIKGpath_gpt4o_mini_alpha_06.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    output_file = "../Abstract_and_Link_A100_86modify/Data/CMtMedQA_train_sampled400_WIKGpath_gpt4o_mini_alpha_06_know_simple.json"
    # output_file = "../Abstract_and_Link_A100_86modify/Data/CMtMedQA_test_sampled50_WIKGpath_gpt4o_mini_alpha_06_know_simple.json"

    # 2. 加载数据
    print(f"\n加载测试数据: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_data = test_data[:]
        
    # 3. 准备结果存储
    all_results = []
    for idx, example in enumerate(tqdm(test_data)):
        dialog_history = example['history']
        summary = example['summary']
        current_q = example['current_q']
        
        knowledge_data = {
            "id": example['id'], 
            "history": example['history'], 
            "summary": example['summary'],
            "current_q": example['current_q'], 
            "response": example['response'], 
            "knowledge": "", 
            # "initial": example['initial'], 
            "cate1": example['cate1'], 
            "cate2": example['cate2'], 
        }

####################  三元组使用方式 ####################
        # 直接连接
        # triples_text = '\n'.join(formatted_triples)
        
#         # 预定义模板
#         triples_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(formatted_triples)])
#         triples_text = f"""
# [医疗知识图谱检索数据]
# 类型：原始结构化医学三元组
# 用途：辅助医生进行鉴别诊断。

# [符号阅读指南]
# 请按以下逻辑解析数据结构：
# 1. 格式："实体A -[关系]-> 实体B" 或 "实体A <-[关系]- 实体B"。
# 2. 箭头含义：箭头指示因果、属性或治疗方向。
# - "症状 -[是临床表现]-> 疾病"：表示该症状支持该疾病的诊断。
# - "症状 <-[改善]- 药物"：表示该药物可治疗该症状。

# [推理与过滤逻辑]
# 1. **聚合路径优先**：高度关注"多对一"结构。如果[历史对话]或[当前输入]中的多个不同症状，通过路径同时指向同一个中心节点（例如：症状A -> 疾病X <- 症状B），这是极高置信度的诊断依据。
# 2. **上下文强一致性校验**：
# - 严厉过滤噪声：对于三元组中出现的药物，只有当患者明确提及相关病史或正在询问治疗方案时才予以考虑。
# - 否则，请忽略这些药物节点，防止产生不符合病情的用药幻觉。

# [检索到的原始三元组列表]
# {triples_block}

# [数据结束]
# """
        # 简化模板
        triples_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(example['related_kg_path'])])
        triples_text = f"""
[辅助诊断知识]
{triples_block}
"""

        knowledge_data["knowledge"] = triples_text
        all_results.append(knowledge_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

