#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版：使用Qwen3-1.7B进行多轮对话历史信息压缩训练
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
import random
import numpy as np
from datetime import datetime
from slot_attention import DialogueCompressionModel
from slot_attention_know import Know_DialogueCompressionModel

from test_know import validate_compressor
from peft import PeftModel

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
    

def train_compressor(memory_compressor, know_compressor, train_loader, optimizer, device, qwen_model, tokenizer, father_path, compressor_name, num_epochs=10, logger=None, save_dir=None, model_config=None, save_interval=2):
    """训练压缩模型"""
    
    for epoch in range(num_epochs):
        know_compressor.train()
        
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
            with torch.no_grad():
                memory_slots, z, mu, logvar = memory_compressor.variational_slot_attention(
                    history_features, history_attention_mask, training=False
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
        know_compressor.eval()
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
            checkpoint_path = os.path.join(save_dir, f'Epoch{epoch+1}' + compressor_name)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': know_compressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'pred_loss': avg_pred_loss,
                'config': model_config
            }, checkpoint_path)
            
            save_msg = f'✓ Checkpoint saved: {checkpoint_path} (Total Loss: {avg_loss:.4f})'
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
    
    
    model.eval()
    # 冻结Qwen3-1.7B模型参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 获取模型配置
    input_dim = model.config.hidden_size
    logger.info(f"Model hidden size: {input_dim}")
    print(f"Model hidden size: {input_dim}")

    # 获取模型的最大输入长度
    # max_length = model.config.max_position_embeddings
    max_length = 512  # 10.26设置为512暂时
    logger.info(f"模型最大输入长度: {max_length}")
    print(f"模型最大输入长度: {max_length}")
    
    # 创建压缩模型 - slot_dim必须与LLM的embedding维度相同
    slot_dim = input_dim  # 确保slots可以作为LLM的前缀
    num_slots = 24
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
    memory_compressor = DialogueCompressionModel(
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=16,
        vocab_size=tokenizer.vocab_size,
        num_iterations=slot_iter_num,
        beta=ib_loss_deta,
        decoder_layers=slot_decoder_layer,
        decoder_heads=slot_decoder_head,
        recon_weight=recon_weight,
        kl_weight=kl_weight,
        pred_weight=pred_weight,
        # max_pos=512, # MMD 
        max_pos=1024,  # CMtMedQA
    ).to(device).to(torch.bfloat16)
    
    # memory_compressor_path = os.path.join(father_path, 'models/training_20251027_220805', "Epoch9compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4.pth")  # MMD
    memory_compressor_path = os.path.join(father_path, 'models/training_20251202_102533', "Epoch9compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_2.pth")  # CMtMedQA_LLM[-1]_qy[pad]_16_1024_static_2_lr0.00005 e9
    
    if os.path.exists(memory_compressor_path):
        logger.info(f"正在从 {memory_compressor_path} 加载记忆压缩器权重...")
        print(f"⏳ 正在从 {memory_compressor_path} 加载记忆压缩器权重...")
        memory_checkpoint = torch.load(memory_compressor_path, map_location=device)
        memory_compressor.load_state_dict(memory_checkpoint['model_state_dict'])
        logger.info("记忆压缩器权重加载成功！")
        print("✓ 记忆压缩器权重加载成功！")
        memory_compressor.eval()
        for param in memory_compressor.parameters():  # 冻结记忆压缩器
            param.requires_grad = False
        logger.info("记忆压缩器权重已冻结！")
        print("✓ 记忆压缩器权重已冻结！")
    else:
        raise FileNotFoundError("未找到记忆压缩器权重!")
    
    ### ❗加载 Knowledge 压缩器❗
    know_compressor = Know_DialogueCompressionModel(
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        vocab_size=tokenizer.vocab_size,
        num_iterations=slot_iter_num,
        beta=ib_loss_deta,
        decoder_layers=slot_decoder_layer,
        decoder_heads=slot_decoder_head,
        recon_weight=recon_weight,
        kl_weight=kl_weight,
        pred_weight=pred_weight,
        max_pos=max_length,  # 使用LLM的最大位置编码长度
    ).to(device).to(torch.bfloat16)
    
    # 创建数据集（+验证集10.30）
    ###❗训练集
    # data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text.json')
    # data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text_know_summary.json')
    # data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text_know_simple_summary.json')
    # data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text_kgpath_gpt4o_mini_alpha_06_allkg_final_know_simple_summary.json')
    data_path = os.path.join(grand_path, 'Data', 'CMtMedQA_train_sampled400_WIKGpath_gpt4o_mini_alpha_06_know_simple_summary.json')
    
    dataset = DialogueDataset(data_path, tokenizer, model, device, max_len_history=seq_max_len, max_len_summary=summary_max_len, max_len_qy=max_qy_len)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    # 优化器
    lr = 5e-5  # 10.26之前学习率为1e-8
    weight_decay = 0.01
    optimizer = optim.AdamW(know_compressor.parameters(), lr=lr, weight_decay=weight_decay)
    
    save_dir = os.path.join(father_path, 'models', f'training_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Checkpoints will be saved to: {save_dir}")
    print(f"✓ Checkpoints将保存到: {save_dir}")
    
    # 准备模型配置（用于保存checkpoint）
    model_config = {
        'input_dim': input_dim,
        'slot_dim': slot_dim,
        'num_slots': num_slots,
        'vocab_size': tokenizer.vocab_size,
        'num_iterations': slot_iter_num,
        'beta': ib_loss_deta,
        'decoder_layers': slot_decoder_layer,
        'decoder_heads': slot_decoder_head,
        'recon_weight': recon_weight,
        'kl_weight': kl_weight,
        'pred_weight': pred_weight,
        'lr': lr,
        'batch_size': train_batch_size,
        'max_pos': max_length,
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
    compressor_name = f'compressor_qwen25_15_model_{num_slots}_{lr}_tgt-1_qyembed_yid_{summary_max_len}_{seq_max_len}_{max_qy_len}_{recon_weight}_{kl_weight}_{pred_weight}_{train_batch_size}_knowledge.pth'
    train_compressor(memory_compressor, know_compressor, train_loader, optimizer, device, model, tokenizer, father_path, compressor_name, num_epochs=train_epoch, logger=logger, save_dir=save_dir, model_config=model_config, save_interval=save_interval)
    
    # 保存最终模型
    torch.save({
        'epoch': train_epoch,
        'model_state_dict': know_compressor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config
    }, os.path.join(save_dir, compressor_name))
    
    logger.info("Training completed and model saved!")
    print("Training completed and model saved!")
    logger.info(compressor_name)
    logger.info("训练程序完成")


if __name__ == "__main__":
    main()
