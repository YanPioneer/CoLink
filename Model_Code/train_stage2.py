#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from test_stage2 import validate_compressor
from peft import PeftModel, get_peft_model, LoraConfig, TaskType


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
        # self.model.eval()
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
            
            # 将历史对话合并为文本
            history_text = ' '.join(history)
            current_text = current_q
            # ###❗聊天模板
            # messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(history)]
            # history_text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # messages = [{"role": "user", "content": current_q}]
            # current_text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            summary_text = summary
            response_text = response
            
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
        q_length = len(q_ids)  # 记录q的长度

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
        current_input_ids = current_encoding['input_ids'].to(self.device)
        
        # qy_ids = qy_encoding['input_ids'].to(self.device)  # qy一起
        qy_ids = qy_ids.to(self.device)
        
        ### 提前计算 Embeddings 和 Features
        with torch.no_grad():
            history_features = extract_features(self.model, history_input_ids, history_attention_mask)
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
    
    def _generate_summary(self, history):
        """生成摘要"""
        # return "用户询问了相关问题，医生提供了建议和指导。"
        ### 根据history[:-1]历史对话列表，生成摘要
        summary_prompt = """你是一个专业的医疗对话摘要助手。请根据以下历史对话列表，生成一个简洁、全面的对话摘要。

**任务要求**：

1.**提取核心信息**： 摘要应涵盖患者的主诉、症状、医生的诊断、建议的治疗方案以及任何重要的注意事项。

2.**保持客观准确**： 严格基于对话内容，不添加任何额外信息或个人推断。

3.**结构清晰**： 使摘要易于阅读，关键信息一目了然。

**输入格式**：

·历史对话列表是一个JSON风格的字符串数组。

·历史对话严格按照“患者发言-医生发言-患者发言-医生发言……”的顺序交替排列。

**请根据以下历史对话列表生成摘要**：

{dialog}

**对话摘要**："""
        messages = [{"role": "user", "content": summary_prompt.format(dialog=history[:-1])}]
        summary_completion = client_A.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return summary_completion.choices[0].message.content


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
    

def train_compressor(compressor, train_loader, optimizer, device, qwen_model, tokenizer, father_path, compressor_name, num_epochs=10, logger=None, save_dir=None, model_config=None, save_interval=2, is_joint_training=True):
    """训练压缩模型"""
    ###❗问题需重测？ 
    # qwen_model.train()
    # if is_joint_training:
    #     compressor.train()
    # else:
    #     compressor.eval()
    
    for epoch in range(num_epochs):
        qwen_model.train()
        if is_joint_training:
            compressor.train()
        else:
            compressor.eval()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        pred_loss_sum = 0
        
        for batch_idx, batch in enumerate(train_loader):
            history_input_ids = batch['history_input_ids'].to(device)
            history_attention_mask = batch['history_attention_mask'].to(device)
            summary_input_ids = batch['summary_input_ids'].to(device)
            summary_attention_mask = batch['summary_attention_mask'].to(device)
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
            
            # 前向传播
            outputs = compressor(
                dialogue_history=history_features,
                history_mask=history_attention_mask,
                summary_ids=summary_input_ids,
                summary_attention_mask=summary_attention_mask,
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
        validate_compressor(compressor, compressor_name, qwen_model, tokenizer, epoch, timestamp)

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
                    'model_state_dict': compressor.state_dict(),
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
    
    #❗协同训练模式
    train_compressor_jointly = True
    if train_compressor_jointly:
        logger.info("协同训练 (压缩器 + 大模型)")
        print("❗ 协同训练 (压缩器 + 大模型)")
    else:
        logger.info("仅微调大模型 (压缩器已冻结)")
        print("❗ 仅微调大模型 (压缩器已冻结)")
    
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
        # dtype=torch.bfloat16,  ###
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
    max_length = 512  # 10.26设置为512暂时
    logger.info(f"模型最大输入长度: {max_length}")
    print(f"模型最大输入长度: {max_length}")
    
    # 创建压缩模型 - slot_dim必须与LLM的embedding维度相同
    slot_dim = input_dim  # 确保slots可以作为LLM的前缀
    num_slots = 16
    summary_max_len = max_length
    seq_max_len = max_length    ### max_length * 4
    max_qy_len = max_length * 2
    ib_loss_deta = 1.0
    slot_decoder_layer = 2
    slot_decoder_head = 4
    slot_iter_num = 3  # 1 for test 3 for formal
    train_batch_size = 8
    # 损失权重参数（用于平衡不同loss项）
    recon_weight = 1.0    # 重构损失权重 0.6
    kl_weight = 1.0       # KL损失权重（通常较小，避免过度正则化） 0.01
    pred_weight = 1.0     # 预测损失权重 1
    
    compressor = DialogueCompressionModel(
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
    
    # 加载第一阶段compressor参数
    # stage1_compressor_path = "../models/training_20251027_220805/compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4.pth"
    # stage1_compressor_path = "../models/training_20251027_220805/Epoch9compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4.pth"
    stage1_compressor_path = "../models/training_20251129_222755/Epoch8compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_8.pth"
    
    if os.path.exists(stage1_compressor_path):
        logger.info(f"正在从 {stage1_compressor_path} 加载第一阶段的压缩器权重...")
        print(f"⏳ 正在从 {stage1_compressor_path} 加载第一阶段的压缩器权重...")
        checkpoint = torch.load(stage1_compressor_path, map_location=device)
        compressor.load_state_dict(checkpoint['model_state_dict'])
        logger.info("第一阶段压缩器权重加载成功！")
        print("✓ 第一阶段压缩器权重加载成功！")
    else:
        raise FileNotFoundError("未找到第一阶段压缩器权重!")
    
    # 创建数据集（+验证集10.30）
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    grand_path = os.path.abspath(os.path.dirname(father_path))  
    
    # data_path = os.path.join(grand_path, 'Data', 'medical_data_v3_summary_text.json')
    data_path = os.path.join(grand_path, 'Data', 'CMtMedQA_train_sampled400.json')
    
    dataset = DialogueDataset(data_path, tokenizer, model, device, max_len_history=seq_max_len, max_len_summary=summary_max_len, max_len_qy=max_qy_len)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    # 优化器
    ###❗stage2学习率
    lr_compressor = 1e-5  
    lr_llm = 1e-5
    weight_decay = 0.01
    # optimizer = optim.AdamW(compressor.parameters(), lr=lr_compressor, weight_decay=weight_decay)
    
    if train_compressor_jointly: # 协同训练模式
        compressor.train()
        param_groups = [
            {'params': compressor.parameters(), 'lr': lr_compressor},
            {'params': model.parameters(), 'lr': lr_llm}
        ]
        logger.info("优化器已配置为协同训练。")
        print("✓ 优化器已配置为协同训练。")
    else:
        compressor.eval()
        for param in compressor.parameters():
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
        'num_slots': num_slots,
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
    compressor_name = f'stage2_compressor_qwen25_15_model_{num_slots}_{lr_compressor}_{lr_llm}_tgt-1_qyembed_yid_{summary_max_len}_{seq_max_len}_{max_qy_len}_{recon_weight}_{kl_weight}_{pred_weight}_{train_batch_size}.pth'
    train_compressor(compressor, train_loader, optimizer, device, model, tokenizer, father_path, compressor_name, num_epochs=train_epoch, logger=logger, save_dir=save_dir, model_config=model_config, save_interval=save_interval, is_joint_training=train_compressor_jointly)
    
    # 保存LoRA适配器
    final_adapter_path = os.path.join(save_dir, 'final_qwen_lora_adapter')
    model.save_pretrained(final_adapter_path)
    log_msg = f"Training completed! LLM adapter saved to {final_adapter_path}"
    
    # 协同训练才保存压缩器
    if train_compressor_jointly:  
        final_compressor_path = os.path.join(save_dir, compressor_name)
        torch.save({
            'epoch': train_epoch,
            'model_state_dict': compressor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_config
        }, final_compressor_path)
        log_msg += f", Compressor saved to {final_compressor_path}"
    
    logger.info(log_msg)
    print(log_msg)
    logger.info("第二阶段训练程序完成")
    

if __name__ == "__main__":
    main()
