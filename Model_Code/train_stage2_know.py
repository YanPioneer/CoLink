#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版：使用Qwen3-1.7B进行多轮对话历史信息压缩训练
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
from datetime import datetime
from slot_attention_stage2 import DialogueCompressionModel
from slot_attention_stage2_know import Know_DialogueCompressionModel
import random
import time
import numpy as np

from test_stage2_know import validate_compressor
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

# from WI.WI_KG import WIKGRetriever

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
def setup_logging(log_dir="logs"):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


### OpenAI_api
import openai
client_A = openai.OpenAI(
    api_key="xxx",
    base_url="xxx",
)


class DialogueDataset(torch.utils.data.Dataset):
    """对话数据集"""
    
    def __init__(self, data_path, tokenizer, model, device, max_len_history=512, max_len_summary=512, max_len_qy=1024):
        self.tokenizer = tokenizer
        self.max_len_history = max_len_history
        self.max_len_summary = max_len_summary
        self.max_len_qy = max_len_qy
        self.model = model
        self.model.eval()
        self.device = device
        print('history max len', max_len_history)
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[:]
        
        # 预处理数据
        self.processed_data = []
        for item in self.data:
            if self._process_dialogue(item):
                self.processed_data.append(item)
    
    def _process_dialogue(self, item):
        """处理单个对话"""
        try:
            history = item.get('history', [])
            if len(history) < 2:
                return False
            return True
        except:
            return False
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # 检查数据格式
        if 'response' in item:
            # 新的数据格式（处理后的数据）
            history = item['history']  # 历史对话列表
            summary = item['summary']  # 生成的摘要
            current_q = item['current_q']  # 当前用户问题
            response = item['response']  # 模型回复
            knowledge = item['knowledge']  # 知识（摘要？）
            knowledge_summary = item['knowledge_summary']  # 知识摘要
            
            # 将历史对话合并为文本
            history_text = ' '.join(history)
            current_text = current_q
            summary_text = summary
            response_text = response
            knowledge_text = knowledge
            knowledge_summary_text = knowledge_summary
            
        # 编码历史对话
        history_encoding = self.tokenizer(
            history_text,
            max_length=self.max_len_history,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 编码当前对话
        current_encoding = self.tokenizer(
            current_text,
            max_length=self.max_len_history,
            truncation=True,
            padding='max_length',
            # padding=False,  # 不padding，直接拼接
            return_tensors='pt'
        )
        
        # 编码摘要
        summary_encoding = self.tokenizer(
            summary_text,
            max_length=self.max_len_summary,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # 编码回复
        response_encoding = self.tokenizer(
            response_text,
            max_length=self.max_len_history,  # 使用相同的最大长度
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        knowledge_encoding = self.tokenizer(
            knowledge_text,
            max_length=self.max_len_history,  # 使用相同的最大长度
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        knowledge_summary_encoding = self.tokenizer(
            knowledge_summary_text,
            max_length=self.max_len_history,  # 使用相同的最大长度
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        ###❗
        # 1. qy一起 + 一次性tokenize
        qy_text = f"{current_text}{response_text}"  ###
        qy_encoding = self.tokenizer(
            qy_text,
            max_length=self.max_len_qy,
            truncation=True,
            padding='max_length',  # 直接padding到固定长度
            return_tensors='pt'
        )  # ✅存在问题，tokenizer(q) + tokenizer(y) /= tokenizer(q+y)
        
        # 2. 分别tokenize q和y
        q_encoding = self.tokenizer(
            current_text,
            max_length=self.max_len_qy // 2,  # 各占一半长度
            truncation=True,
            return_tensors='pt'
        )
        y_encoding = self.tokenizer(
            response_text,
            max_length=self.max_len_qy // 2,
            truncation=True,
            return_tensors='pt'
        )
        q_ids = q_encoding['input_ids'].squeeze(0)  # [q_len]
        y_ids = y_encoding['input_ids'].squeeze(0)  # [y_len]
        q_length = len(q_ids)

        qy_ids_concat = torch.cat([q_ids, y_ids], dim=0)  # [q_len + y_len]
        qy_len = len(qy_ids_concat)
        if qy_len < self.max_len_qy:  # 需要padding
            padding_len = self.max_len_qy - qy_len
            qy_ids = torch.cat([
                qy_ids_concat,
                torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ], dim=0)
            qy_attention_mask = torch.cat([
                torch.ones(qy_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ], dim=0)
        else:  # ❌不会到该分支
            # 已经达到或超过最大长度，截断
            qy_ids = qy_ids_concat[:self.max_len_qy]
            qy_attention_mask = torch.ones(self.max_len_qy, dtype=torch.long)

        history_input_ids = history_encoding['input_ids'].to(self.device)
        history_attention_mask = history_encoding['attention_mask'].to(self.device)
        knowledge_input_ids = knowledge_encoding['input_ids'].to(self.device)
        knowledge_attention_mask = knowledge_encoding['attention_mask'].to(self.device)
        current_input_ids = current_encoding['input_ids'].to(self.device)
        
        # qy_ids = qy_encoding['input_ids'].to(self.device)  # qy一起
        qy_ids = qy_ids.to(self.device)
        
        ### 提前计算 Embeddings 和 Features
        with torch.no_grad():
            history_features = extract_features(self.model, history_input_ids, history_attention_mask)
            knowledge_features = extract_features(self.model, knowledge_input_ids, knowledge_attention_mask)
            q_embeddings = self.model.get_input_embeddings()(current_input_ids)
            # qy_embeddings = self.model.get_input_embeddings()(qy_ids)  # qy一起
            qy_embeddings = self.model.get_input_embeddings()(qy_ids.unsqueeze(0))
        
        return {
            'history_input_ids': history_encoding['input_ids'].squeeze(0),
            'history_attention_mask': history_encoding['attention_mask'].squeeze(0),
            'current_input_ids': current_encoding['input_ids'].squeeze(0),
            'current_attention_mask': current_encoding['attention_mask'].squeeze(0),
            'summary_input_ids': summary_encoding['input_ids'].squeeze(0),
            'summary_attention_mask': summary_encoding['attention_mask'].squeeze(0),
            'response_input_ids': response_encoding['input_ids'].squeeze(0),
            'response_attention_mask': response_encoding['attention_mask'].squeeze(0),
            
            'knowledge_input_ids': knowledge_encoding['input_ids'].squeeze(0),
            'knowledge_attention_mask': knowledge_encoding['attention_mask'].squeeze(0),
            'knowledge_summary_input_ids': knowledge_summary_encoding['input_ids'].squeeze(0),
            'knowledge_summary_attention_mask': knowledge_summary_encoding['attention_mask'].squeeze(0),
            'knowledge_features': knowledge_features.squeeze(0),
            
            'current_text': current_text,  # 添加当前问题文本
            'response_text': response_text,  # 添加回复文本
            'q_length': q_length,

            'qy_ids': qy_ids,  # [max_len_qy]
            'qy_attention_mask': qy_attention_mask,  # [max_len_qy] - attention mask
            # 'qy_ids': qy_encoding['input_ids'].squeeze(0),  # [max_len_qy]
            # 'qy_attention_mask': qy_encoding['attention_mask'].squeeze(0),  # [max_len_qy] - attention mask
            
            'history_features': history_features.squeeze(0),
            'q_embeddings': q_embeddings.squeeze(0),
            'qy_embeddings': qy_embeddings.squeeze(0),
        }



def extract_features(model, input_ids, attention_mask):
    """从Qwen3-1.7B模型提取特征"""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        # 使用最后一层的隐藏状态
        features = outputs.hidden_states[-1]
    return features
    
    

def train_compressor(memory_compressor, know_compressor, train_loader, optimizer, device, qwen_model, tokenizer, father_path, compressor_name, num_epochs=10, logger=None, save_dir=None, model_config=None, save_interval=2, is_joint_training=True):
    """训练压缩模型"""
    
    for epoch in range(num_epochs):
        # 开启训练模式
        qwen_model.train()    
        if is_joint_training:
            memory_compressor.train()
            know_compressor.train()
        else:
            memory_compressor.eval()
            know_compressor.eval()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        pred_loss_sum = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动到设备
            # history_input_ids = batch['history_input_ids'].to(device)
            history_attention_mask = batch['history_attention_mask'].to(device)
            # summary_input_ids = batch['summary_input_ids'].to(device)
            # summary_attention_mask = batch['summary_attention_mask'].to(device)
            knowledge_input_ids = batch['knowledge_input_ids'].to(device)
            knowledge_attention_mask = batch['knowledge_attention_mask'].to(device)
            knowledge_summary_input_ids = batch['knowledge_summary_input_ids'].to(device)
            knowledge_summary_attention_mask = batch['knowledge_summary_attention_mask'].to(device)
            knowledge_features = batch['knowledge_features'].to(device)
            
            current_input_ids = batch['current_input_ids'].to(device)
            current_attention_mask = batch['current_attention_mask'].to(device)
            qy_attention_mask = batch['qy_attention_mask'].to(device)
            
            # 获取文本数据（用于预测损失）
            current_text = batch.get('current_text', [])
            response_text = batch.get('response_text', [])
            history_features = batch['history_features'].to(device)
            q_embeddings = batch['q_embeddings'].to(device)
            qy_embeddings = batch['qy_embeddings'].to(device)
            q_lengths = batch['q_length']
            y_ids = batch['response_input_ids'].to(device)
            
            # 获取Memory_slot
            memory_slots, z, mu, logvar = memory_compressor.variational_slot_attention(
                history_features, history_attention_mask, training=True
            )
            
            # 前向传播
            outputs = know_compressor(
                memory_slots=memory_slots,
                knowledge_features=knowledge_features,
                knowledge_ids=knowledge_input_ids,
                knowledge_attention_mask=knowledge_attention_mask,
                summary_ids=knowledge_summary_input_ids,
                summary_attention_mask=knowledge_summary_attention_mask,
                q_embeddings=q_embeddings,
                q_mask=current_attention_mask,
                q_ids=current_input_ids,
                y_ids=y_ids,
                qy_embeddings=qy_embeddings,
                qy_attention_mask=qy_attention_mask,
                q_lengths=q_lengths,
                frozen_llm=qwen_model,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
                training=True,
                return_text=(batch_idx % 100 == 0)  # 每100个batch查看一次重构文本
            )
            
            # 反向传播
            optimizer.zero_grad()
            outputs['total_loss'].backward()
            
            # 梯度裁剪，防止梯度爆炸（更严格的裁剪）
            # torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1)
            
            optimizer.step()
            
            # 记录损失
            total_loss += outputs['total_loss'].item()
            recon_loss_sum += outputs['reconstruction_loss'].item()
            kl_loss_sum += outputs['kl_loss'].item()
            pred_loss_sum += outputs['prediction_loss'].item()
            
            # 每个batch都输出详细信息
            batch_msg = (f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                        f'Total: {outputs["total_loss"]:.4f}, '
                        f'Recon: {outputs["reconstruction_loss"]:.4f}, '
                        f'KL: {outputs["kl_loss"]:.4f}, '
                        f'Pred: {outputs["prediction_loss"]:.4f}')
            if logger:
                logger.info(batch_msg)
            if batch_idx % 10 == 0:
                print(batch_msg)
            
            # 每100个batch输出q y 重构文本信息
            if batch_idx % 100 == 0:
                q_msg1 = f'  Current Q: {current_text[0][:100]}...'
                y_msg2 = f'  Response: {response_text[0][:100]}...'
                print(q_msg1)
                print(y_msg2)
                if logger:
                    logger.info(q_msg1)
                    logger.info(y_msg2)
            
            # 每100个batch输出重构文本
            if 'reconstructed_texts' in outputs and batch_idx % 100 == 0:
                recon_msg = f'  Reconstructed: {outputs["reconstructed_texts"][0][:200]}...'
                print(recon_msg)
                if logger:
                    logger.info(recon_msg)

        ### 验证集（三指标-测试epoch数量）
        validate_compressor(memory_compressor, know_compressor, compressor_name, qwen_model, tokenizer, epoch, timestamp)

        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = recon_loss_sum / len(train_loader)
        avg_kl_loss = kl_loss_sum / len(train_loader)
        avg_pred_loss = pred_loss_sum / len(train_loader)
        
        epoch_msg = (f'Epoch {epoch+1} completed. '
                    f'Avg Total: {avg_loss:.4f}, '
                    f'Avg Recon: {avg_recon_loss:.4f}, '
                    f'Avg KL: {avg_kl_loss:.4f}, '
                    f'Avg Pred: {avg_pred_loss:.4f}')
        print(epoch_msg)
        if logger:
            logger.info(epoch_msg)

        # 每隔save_interval个epoch保存一次模型
        if save_dir is not None and model_config is not None and (epoch + 1) % save_interval == 0:            
            # 保存大模型的LoRA适配器
            llm_adapter_path = os.path.join(save_dir, f'Epoch{epoch+1}_qwen_lora_adapter')
            qwen_model.save_pretrained(llm_adapter_path)
            save_msg = f'✓ Checkpoint saved: LLM Adapter to {llm_adapter_path}'
            
            # 只有在协同训练模式下才保存压缩器
            if is_joint_training:
                checkpoint_path = os.path.join(save_dir, f'Epoch{epoch+1}' + compressor_name)
                
                torch.save({
                    'epoch': epoch + 1,
                    'mem_model_state_dict': memory_compressor.state_dict(),
                    'know_model_state_dict': know_compressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_loss': avg_loss,
                    'recon_loss': avg_recon_loss,
                    'kl_loss': avg_kl_loss,
                    'pred_loss': avg_pred_loss,
                    'config': model_config
                }, checkpoint_path)
                
                save_msg += f' and Compressor to {checkpoint_path} (Total Loss: {avg_loss:.4f}) '
            
            print(save_msg)
            if logger:
                logger.info(save_msg)


def main():
    # 设置日志
    logger = setup_logging()
    logger.info("开始训练程序")
    set_seed(42)  #❗随机数种子
    
    # 配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    grand_path = os.path.abspath(os.path.dirname(father_path))  
    
    #❗协同训练模式
    train_compressor_jointly = True
    if train_compressor_jointly:
        logger.info("协同训练 (压缩器 + 大模型)")
        print("❗ 协同训练 (压缩器 + 大模型) ❗")
    else:
        logger.info("仅微调大模型")
        print("❗ 仅微调大模型 ❗")
    
    # 加载Qwen3-1.7B模型和tokenizer
    model_name = "../Qwen2.5-1.5B-Instruct/"
    logger.info(f"Loading {model_name}...")
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer加载完成")
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    logger.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    logger.info("开始加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
    ).to(device)
    logger.info("模型加载完成")
    
    
    model.train()
    logger.info("模型已设置为训练模式")
    print("✓ 模型已设置为训练模式")
    
    #❗LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # --------------------------------------------------------------------
    
    # 获取模型配置
    input_dim = model.config.hidden_size
    logger.info(f"Model hidden size: {input_dim}")
    print(f"Model hidden size: {input_dim}")

    # 获取模型的最大输入长度
    # max_length = model.config.max_position_embeddings
    max_length = 1024  # 10.26设置为512暂时
    logger.info(f"模型最大输入长度: {max_length}")
    print(f"模型最大输入长度: {max_length}")
    
    # 创建压缩模型 - slot_dim必须与LLM的embedding维度相同
    slot_dim = input_dim  # 确保slots可以作为LLM的前缀
    # num_slots = 16
    summary_max_len = max_length
    seq_max_len = max_length    ### max_length * 4
    max_qy_len = max_length * 2
    ib_loss_deta = 1.0
    slot_decoder_layer = 2
    slot_decoder_head = 4
    slot_iter_num = 3  # 1 for test 3 for formal
    train_batch_size = 4
    # 损失权重参数（用于平衡不同loss项）
    recon_weight = 1.0    # 重构损失权重 0.6
    kl_weight = 1.0       # KL损失权重（通常较小，避免过度正则化） 0.01
    pred_weight = 1.0     # 预测损失权重 1
    
    ### ❗加载 Memory 压缩器❗
    memory_compressor_path = os.path.join(father_path, 'models/training_20251027_220805', "Epoch9compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4.pth")
    if os.path.exists(memory_compressor_path):
        logger.info(f"正在从 {memory_compressor_path} 加载第一阶段的记忆压缩器权重...")
        print(f"⏳ 正在从 {memory_compressor_path} 加载第一阶段的记忆压缩器权重...")
        mem_checkpoint = torch.load(memory_compressor_path, map_location=device)
        
        # 自动推断相关配置
        if 'config' in mem_checkpoint:
            config = mem_checkpoint['config']
            slot_dim = config.get('slot_dim', input_dim)
            num_slots = config.get('num_slots', 16)
            num_iterations = config.get('num_iterations', 3)
            decoder_layers = config.get('decoder_layers', 2)
            decoder_heads = config.get('decoder_heads', 4)
        else:
            # 从state_dict推断num_slots（通过slot_embeddings的形状）
            state_dict = mem_checkpoint['model_state_dict']
            slot_embeddings_key = 'variational_slot_attention.slot_attention.slot_embeddings'
            if slot_embeddings_key in state_dict:
                num_slots = state_dict[slot_embeddings_key].shape[1]
                slot_dim = state_dict[slot_embeddings_key].shape[2]
            else:
                # 默认值
                num_slots = 16
                slot_dim = input_dim
            
            num_iterations = 3
            decoder_layers = 2
            decoder_heads = 4
        
        print(f"✓ 检测到模型参数:")
        print(f"  - input_dim: {input_dim}")
        print(f"  - slot_dim: {slot_dim}")
        print(f"  - num_slots: {num_slots}")
        print(f"  - num_iterations: {num_iterations}")
        
        # 创建模型
        memory_compressor = DialogueCompressionModel(
            input_dim=input_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            vocab_size=tokenizer.vocab_size,
            num_iterations=num_iterations,
            beta=ib_loss_deta,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            recon_weight=recon_weight,
            kl_weight=kl_weight,
            pred_weight=pred_weight,
            max_pos=512,
        ).to(device).to(torch.bfloat16)
        
        memory_compressor.load_state_dict(mem_checkpoint['model_state_dict'])
        
        logger.info("第一阶段记忆压缩器权重加载成功！")
        print("✓ 第一阶段记忆压缩器权重加载成功！")
        # 若非协同训练
        if not train_compressor_jointly:
            memory_compressor.eval()
            for param in memory_compressor.parameters():  # 冻结记忆压缩器
                param.requires_grad = False
            logger.info("记忆压缩器权重已冻结！")
            print("✓ 记忆压缩器权重已冻结！")
    else:
        raise FileNotFoundError("未找到记忆压缩器权重!")

    ### ❗加载 Knowledge 压缩器❗
    # know_compressor_path = os.path.join(father_path, 'models/training_20260115_173634', "Epoch6compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_16_512_static_4 e6
    # know_compressor_path = os.path.join(father_path, 'models/training_20260115_173715', "Epoch3compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_24_512_static_4 e3
    know_compressor_path = os.path.join(father_path, 'models/training_20260115_143120', "Epoch4compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")   # Simple_LLM[-1]_qy[pad]_24_1024_static_4 e4
    if os.path.exists(know_compressor_path):
        logger.info(f"正在从 {know_compressor_path} 加载第一阶段的知识压缩器权重...")
        print(f"⏳ 正在从 {know_compressor_path} 加载第一阶段的知识压缩器权重...")
        know_checkpoint = torch.load(know_compressor_path, map_location=device)
        
        # 自动推断相关配置
        if 'config' in know_checkpoint:
            config = know_checkpoint['config']
            slot_dim = config.get('slot_dim', input_dim)
            num_slots = config.get('num_slots', 16)
            num_iterations = config.get('num_iterations', 3)
            decoder_layers = config.get('decoder_layers', 2)
            decoder_heads = config.get('decoder_heads', 4)
        else:
            state_dict = know_checkpoint['model_state_dict']
            slot_embeddings_key = 'variational_slot_attention.slot_attention.slot_embeddings'
            if slot_embeddings_key in state_dict:
                num_slots = state_dict[slot_embeddings_key].shape[1]
                slot_dim = state_dict[slot_embeddings_key].shape[2]
            else:
                # 默认值
                num_slots = 16
                slot_dim = input_dim
            
            num_iterations = 3
            decoder_layers = 2
            decoder_heads = 4
        
        print(f"✓ 检测知识压缩器参数:")
        print(f"  - input_dim: {input_dim}")
        print(f"  - slot_dim: {slot_dim}")
        print(f"  - num_slots: {num_slots}")
        print(f"  - num_iterations: {num_iterations}")
        
        know_compressor = Know_DialogueCompressionModel(
            input_dim=input_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            vocab_size=tokenizer.vocab_size,
            num_iterations=num_iterations,
            beta=ib_loss_deta,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            recon_weight=recon_weight,
            kl_weight=kl_weight,
            pred_weight=pred_weight,
            max_pos=max_length,
        ).to(device).to(torch.bfloat16)
        
        know_compressor.load_state_dict(know_checkpoint['model_state_dict'])
        
        logger.info("第一阶段知识压缩器权重加载成功！")
        print("✓ 第一阶段知识压缩器权重加载成功！")
        # 若非协同训练
        if not train_compressor_jointly:
            know_compressor.eval()
            for param in know_compressor.parameters():  # 冻结知识压缩器
                param.requires_grad = False
            logger.info("知识压缩器权重已冻结！")
            print("✓ 知识压缩器权重已冻结！")
    else:
        raise FileNotFoundError("未找到知识压缩器权重!")
    
    # 创建数据集（+验证集10.30）
    data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text_know_simple_summary.json')
    # data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text_kgpath_gpt4o_mini_alpha_06_allkg_final_know_simple_summary.json')
    
    dataset = DialogueDataset(data_path, tokenizer, model, device, max_len_history=seq_max_len, max_len_summary=summary_max_len, max_len_qy=max_qy_len)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    # 优化器
    ###❗stage2学习率
    lr_compressor = 5e-6
    lr_llm = 5e-6
    weight_decay = 0.01
    
    if train_compressor_jointly: # 协同训练模式
        memory_compressor.train()
        know_compressor.train()
        param_groups = [
            {'params': memory_compressor.parameters(), 'lr': lr_compressor},
            {'params': know_compressor.parameters(), 'lr': lr_compressor},
            {'params': model.parameters(), 'lr': lr_llm}
        ]
        logger.info("优化器已配置为协同训练。")
        print("✓ 优化器已配置为协同训练。")
    else:
        memory_compressor.eval()
        know_compressor.eval()
        for param in memory_compressor.parameters():
            param.requires_grad = False
        for param in know_compressor.parameters():
            param.requires_grad = False
        param_groups = [{'params': model.parameters(), 'lr': lr_llm}]
        logger.info("压缩器已冻结。优化器仅配置为微调大模型。")
        print("✓ 压缩器已冻结。优化器仅配置为微调大模型。")
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    save_dir = os.path.join(father_path, 'models', f'training_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Checkpoints will be saved to: {save_dir}")
    print(f"✓ Checkpoints将保存到: {save_dir}")
    
    lora_config_dict = lora_config.to_dict()
    if 'target_modules' in lora_config_dict and isinstance(lora_config_dict['target_modules'], set):
        lora_config_dict['target_modules'] = list(lora_config_dict['target_modules'])
    
    # 准备模型配置（用于保存checkpoint）
    model_config = {
        'input_dim': input_dim,
        'slot_dim': slot_dim,
        'mem_num_slots': 16,
        'know_num_slots': num_slots,
        'vocab_size': tokenizer.vocab_size,
        'num_iterations': slot_iter_num,
        'beta': ib_loss_deta,
        'decoder_layers': slot_decoder_layer,
        'decoder_heads': slot_decoder_head,
        'recon_weight': recon_weight,
        'kl_weight': kl_weight,
        'pred_weight': pred_weight,
        'lr_compressor': lr_compressor,
        'lr_llm': lr_llm,
        'lora_config': lora_config_dict,
        'batch_size': train_batch_size,
        'mem_max_pos': 512,
        'know_max_pos': max_length,
        'summary_max_len':summary_max_len,
        'seq_max_len':seq_max_len,
        'max_qy_len':max_qy_len,
    }
    
    # 保存训练配置到文件
    config_file = os.path.join(save_dir, 'training_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 训练配置已保存到: {config_file}")

    # 训练
    train_epoch = 10
    save_interval = 1  # 每3个epoch保存一次
    logger.info("Starting training...")
    print("Starting training...")
    compressor_name = f'stage2_compressor_qwen25_15_model_16_{num_slots}_{lr_compressor}_{lr_llm}_tgt-1_qyembed_yid_{summary_max_len}_{seq_max_len}_{max_qy_len}_{recon_weight}_{kl_weight}_{pred_weight}_{train_batch_size}.pth'
    train_compressor(memory_compressor, know_compressor, train_loader, optimizer, device, model, tokenizer, father_path, compressor_name, num_epochs=train_epoch, logger=logger, save_dir=save_dir, model_config=model_config, save_interval=save_interval, is_joint_training=train_compressor_jointly)
    
    # 保存LoRA适配器
    final_adapter_path = os.path.join(save_dir, 'final_qwen_lora_adapter')
    model.save_pretrained(final_adapter_path)
    log_msg = f"Training completed! LLM adapter saved to {final_adapter_path}"
    
    # 协同训练才保存压缩器
    if train_compressor_jointly:  
        final_compressor_path = os.path.join(save_dir, compressor_name)
        torch.save({
            'epoch': train_epoch,
            'mem_model_state_dict': memory_compressor.state_dict(),
            'know_model_state_dict': know_compressor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_config
        }, final_compressor_path)
        log_msg += f", Compressor saved to {final_compressor_path}"
    
    logger.info(log_msg)
    print(log_msg)
    logger.info("第二阶段训练程序完成")
    

if __name__ == "__main__":
    main()
