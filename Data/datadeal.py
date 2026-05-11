#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本：将多轮对话数据转换为训练格式
处理medical_data_v1_500.json，生成包含history/summary/current_q/response的数据
"""

import json
import os
import time
from typing import List, Dict, Any
import openai
from tqdm import tqdm
from openai import OpenAI


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
            os.environ["OPENAI_BASE_URL"] = "xxx"
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
        print(f"加载完成，共 {len(data)} 条原始数据")
        return data
    
    def generate_summary_with_gpt4(self, history: List[str], max_retries: int = 3) -> str:
        """
        使用GPT-4生成历史对话的摘要
        
        Args:
            history: 历史对话列表
            max_retries: 最大重试次数
            
        Returns:
            生成的摘要文本
        """
        # 构建历史对话文本
        history_text = "\n".join([f"第{i+1}轮: {dialogue}" for i, dialogue in enumerate(history)])
        
        prompt = f"""请为以下医疗对话历史生成一个简洁的摘要，要求：
1. 提取关键医疗信息（症状、诊断、治疗、建议等）
2. 保持医学专业性
3. 摘要长度控制在100-200字
4. 使用中文

对话历史：
{history_text}

摘要："""

        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "你是一个专业的医疗助手，擅长总结医疗对话。"},
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
                    return f"医疗对话摘要：包含{len(history)}轮对话，涉及患者症状和医疗建议。"
    
    def process_single_dialogue(self, dialogue_data: Dict, idx: int) -> List[Dict]:
        """
        处理单个对话数据
        
        Args:
            dialogue_data: 单个对话数据
            idx: 数据索引
            
        Returns:
            处理后的数据列表
        """
        processed_data = []
        first_session_dialogue = dialogue_data.get("first_session_dialogue", [])
        
        if len(first_session_dialogue) < 4:  # 至少需要4条对话才能生成训练数据
            return processed_data
        
        # 从用户第2轮开始处理（索引为2, 4, 6...）
        for i in range(2, len(first_session_dialogue) - 1, 2):  # 步长为2，只遍历用户说话的轮次
            # history: 前i条对话 [0, i)
            history = first_session_dialogue[:i]
            
            # current_q: 当前用户问题（第i条）
            current_q = first_session_dialogue[i]
            
            # response: 模型回复（第i+1条）
            response = first_session_dialogue[i + 1]
            
            # 生成摘要
            user_round = i // 2 + 1
            print(f"正在为数据 {idx} 的用户第 {user_round} 轮对话生成摘要...")
            summary = self.generate_summary_with_gpt4(history)
            
            # 构建新的数据条目
            new_data = {
                "id": f"processed_{idx}_user{i//2+1}",  # 格式：processed_原始idx_用户第几轮
                "history": history,
                "summary": summary,
                "current_q": current_q,
                "response": response,
                "initial": f"idx{idx}-{0}-{i-1}"  # 记录来源信息
            }
            
            processed_data.append(new_data)
            
            # 添加延迟避免API限制
            time.sleep(0.5)
        
        return processed_data
    
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
                processed_data = self.process_single_dialogue(dialogue_data, idx)
                all_processed_data.extend(processed_data)
                
                print(f"数据 {idx} 处理完成，生成 {len(processed_data)} 条训练数据")
                
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
    
    def preview_processed_data(self, file_path: str, num_samples: int = 3):
        """
        预览处理后的数据
        
        Args:
            file_path: 数据文件路径
            num_samples: 预览样本数量
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据文件包含 {len(data)} 条记录")
        print("\n预览样本:")
        print("=" * 80)
        
        for i, sample in enumerate(data[:num_samples]):
            print(f"\n样本 {i+1}:")
            print(f"ID: {sample['id']}")
            print(f"来源: {sample['initial']}")
            print(f"历史对话数量: {len(sample['history'])}")
            print(f"摘要: {sample['summary'][:100]}...")
            print(f"当前问题: {sample['current_q'][:100]}...")
            print(f"回复: {sample['response'][:100]}...")


def main():
    """主函数"""
    # 数据
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    input_file = os.path.join(father_path, "final_v3_merged.json")
    output_file = os.path.join(father_path, "medical_data_v3_summary_text.json")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    # 初始化处理器
    try:
        processor = MedicalDataProcessor('xxxxx')
    except ValueError as e:
        print(f"错误: {e}")
        print("请设置环境变量 OPENAI_API_KEY 或直接在代码中提供API密钥")
        return
    
    # 处理数据
    try:
        processor.process_all_data(input_file, output_file)
        
        # 预览处理结果
        if os.path.exists(output_file):
            processor.preview_processed_data(output_file)
            
    except Exception as e:
        print(f"处理过程中出错: {e}")


if __name__ == "__main__":
    main()
