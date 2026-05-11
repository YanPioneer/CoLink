#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本
"""

import json
import os
import time
from typing import List, Dict, Any
import openai
from tqdm import tqdm
from openai import OpenAI
from data_process_prompt import *

class MedicalDataProcessor:
    def __init__(self, openai_api_key: str = None):
        """
        初始化数据处理器
        
        Args:
            openai_api_key: OpenAI API密钥，如果为None则从环境变量获取
        """
        if openai_api_key:
            openai.api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["OPENAI_BASE_URL"] = "xxxx"
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai.api_key:
            raise ValueError("请设置OpenAI API密钥")
    
    def load_original_data(self, file_path: str) -> List[Dict]:
        """
        加载原始数据
        
        Args:
            file_path: 原始数据文件路径
            
        Returns:
            原始数据列表
        """
        print(f"正在加载数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # data = data[:1000]
        # data = data[1000:2000]
        # data = data[2000:3000]
        # data = data[3000:4000]
        data = data[4000:]
        print(f"加载完成，共 {len(data)} 条原始数据")
        return data
    
    def generate_summary_with_gpt4(self, knowledge_text, max_retries) -> str:
        """
        使用GPT-4生成历史对话的摘要
        
        Args:
            history: 历史对话列表
            max_retries: 最大重试次数
            
        Returns:
            生成的摘要文本
        """
#         prompt = f"""请为以下序列化医学三元组列表生成一个简洁的摘要，要求：
# 1. 提取关键医疗信息（症状、诊断、治疗、建议等）
# 2. 保持医学专业性
# 3. 摘要长度控制在100-200字
# 4. 使用中文

# 对话历史：
# {knowledge_text}

# 摘要："""

        # prompt = knowledge_summary_prompt.format(knowledge_text=knowledge_text)
        prompt = knowledge_simple_summary_prompt.format(knowledge_text=knowledge_text)
        
        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        # {"role": "system", "content": system_summary_prompt},
                        {"role": "system", "content": system_simple_summary_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    # max_tokens=300,
                    # temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                return summary
                
            except Exception as e:
                print(f"GPT-4 API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    # 如果API调用失败，返回一个简单的摘要
                    return f"检索知识摘要：涉及患者症状和医疗建议。"
    
    def process_knowledge(self, dialogue_data: Dict, idx: int) -> List[Dict]:
        """
        处理单个对话数据
        
        Args:
            dialogue_data: 单个对话数据
            idx: 数据索引
            
        Returns:
            处理后的数据列表
        """
        knowledge_text = dialogue_data["knowledge"]
        knowledge_summary = self.generate_summary_with_gpt4(knowledge_text, 3)
        
        # 构建新的数据条目
        new_data = {
            "id": dialogue_data['id'], 
            "history": dialogue_data['history'], 
            "summary": dialogue_data['summary'],
            "current_q": dialogue_data['current_q'], 
            "response": dialogue_data['response'], 
            "knowledge": dialogue_data['knowledge'],
            "knowledge_summary": knowledge_summary,
            # "initial": dialogue_data['initial'], 
            "cate1": dialogue_data['cate1'], 
            "cate2": dialogue_data['cate2'], 
        }
        
        return new_data
    
    def process_all_data(self, input_file: str, output_file: str):
        """
        处理所有数据
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        # 加载原始数据
        original_data = self.load_original_data(input_file)
        
        all_processed_data = []
        
        print("开始处理数据...")
        for idx, dialogue_data in enumerate(tqdm(original_data, desc="处理进度")):
            try:
                processed_data = self.process_knowledge(dialogue_data, idx)
                all_processed_data.append(processed_data)
                
                # 每处理10个数据保存一次（防止数据丢失）
                if (idx + 1) % 10 == 0:
                    self.save_data(all_processed_data, output_file)
                    print(f"已保存 {len(all_processed_data)} 条数据到 {output_file}")
                
            except Exception as e:
                print(f"处理数据 {idx} 时出错: {e}")
                continue
        
        # 最终保存
        self.save_data(all_processed_data, output_file)
        print(f"数据处理完成！共生成 {len(all_processed_data)} 条训练数据")
        print(f"数据已保存到: {output_file}")
    
    def save_data(self, data: List[Dict], file_path: str):
        """
        保存数据到JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    

def main():
    """主函数"""
    # 数据
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    input_file = os.path.join(father_path, "CMtMedQA_train_sampled400_WIKGpath_gpt4o_mini_alpha_06_know_simple.json")
    output_file = os.path.join(father_path, "CMtMedQA_train_sampled400_WIKGpath_gpt4o_mini_alpha_06_know_simple_summary_4000_5331.json")
    
    # 初始化处理器
    try:
        processor = MedicalDataProcessor('xxxx')
    except ValueError as e:
        print(f"错误: {e}")
        print("请设置环境变量 OPENAI_API_KEY 或直接在代码中提供API密钥")
        return
    
    # 处理数据
    try:
        processor.process_all_data(input_file, output_file)
    except Exception as e:
        print(f"处理过程中出错: {e}")


if __name__ == "__main__":
    main()
