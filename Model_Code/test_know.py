#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比测试脚本：比较三种输入方式的效果
1. slots + question (压缩历史)
2. history_text + question (纯文本历史)
3. question only (无历史)
4. summary_text + question 

同时评估不同temperature设置的影响
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from slot_attention import DialogueCompressionModel
from slot_attention_know import Know_DialogueCompressionModel
from tqdm import tqdm
import numpy as np
from peft import PeftModel

# 评估指标
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: BLEU/ROUGE/BERT-SCORE metrics not available. Please install: pip install nltk rouge bert-score")
    METRICS_AVAILABLE = False


# ===== 配置 =====
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
grand_path = os.path.abspath(os.path.dirname(father_path))  

# DATA_PATH = os.path.join(grand_path, 'Data', 'medical_data_v3_eval_know.json')
# DATA_PATH = os.path.join(grand_path, 'Data', 'medical_data_v3_eval_know_simple.json')
# DATA_PATH = os.path.join(grand_path, 'Data', 'medical_data_v3_eval_kgpath_gpt4o_mini_alpha_06_allkg_final_know_simple.json')


# MODEL_PATH = os.path.join(father_path, 'models/training_20251027_220805', "compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4.pth")  # LLM[-1]_qy[pad]_16_512_static_4
# MODEL_PATH = os.path.join(father_path, 'models/training_20251027_220805', "Epoch9compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4.pth")  # LLM[-1]_qy[pad]_16_512_static_4 e9

# MODEL_PATH = os.path.join(father_path, 'models/training_20251028_171032', "compressor_qwen25_15_model_16_0.0001_tgt0_qyembed_yid_512_2048_1024_1.0_1.0_1.0_4.pth")  # LLM[0]_qy[pad]_16_512*4_static_4
# MODEL_PATH = os.path.join(father_path, 'models/training_20251028_180403', "compressor_qwen25_15_model_16_0.0001_tgt-1_qyembed_yid_512_2048_1024_1.0_1.0_1.0_4.pth")  # LLM[-1]_qy[pad]_16_512*4_static_4
# MODEL_PATH = os.path.join(father_path, 'models/training_20251027_211901', "compressor_qwen25_15_model_16_0.0001_tgt-1_qembed_yid_512_2048_1024_1.0_1.0_1.0_4.pth")  # LLM[-1]_q[pad]y[pad]_16_512*4_static_4
# MODEL_PATH = os.path.join(father_path, 'models/training_20251028_124004', "compressor_qwen25_15_model_16_0.0001_tgt0_qembed_yid_512_2048_1024_1.0_1.0_1.0_4.pth")  # LLM[0]_q[pad]y[pad]_16_512*4_static_4
# MODEL_PATH = os.path.join(father_path, 'models/training_20251028_213220', "compressor_qwen25_15_model_24_0.0001_tgt-1_qembed_yid_512_2048_1024_1.0_1.0_1.0_4.pth")  # LLM[-1]_q[pad]y[pad]_24_512*4_static_4

# KNOW_PATH = os.path.join(father_path, 'models/training_20260112_222136', "Epoch4compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_8_knowledge.pth")  # LLM[-1]_qy[pad]_16_512_static_8 e4

# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_092626', "compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_8_256_static_4_NoKnow e10
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_091724', "Epoch2compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_16_256_static_4_NoKnow e2
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_093048', "Epoch5compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_24_256_static_4_NoKnow e5
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_092501', "Epoch6compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_8_512_static_4_NoKnow e6
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_090417', "Epoch3compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_16_512_static_4_NoKnow e3
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_092908', "Epoch6compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_24_512_static_4_NoKnow e6
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_092211', "Epoch4compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_8_1024_static_4_NoKnow e4
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_091953', "Epoch8compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_16_1024_static_4_NoKnow e8
# KNOW_PATH = os.path.join(father_path, 'models/training_20260113_092331', "Epoch7compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_24_1024_static_4_NoKnow e7

# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_093345', "Epoch3compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_8_256_static_4 e3
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_093559', "Epoch1compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_16_256_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_093726', "Epoch1compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_24_256_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_110131', "Epoch9compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_8_512_static_4 e9
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_110257', "Epoch6compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_16_512_static_4 e6
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_110434', "Epoch7compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_24_512_static_4 e7
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_094148', "Epoch1compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_8_1024_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_094011', "Epoch9compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_16_1024_static_4 e9
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_093848', "compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # LLM[-1]_qy[pad]_24_1024_static_4 e10

# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_142759', "Epoch9compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_8_256_static_4 e9
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_142902', "Epoch1compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_16_256_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_143023', "Epoch3compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_24_256_static_4 e3
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_173508', "Epoch1compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_8_512_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_173634', "Epoch6compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_16_512_static_4 e6
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_173715', "Epoch3compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_24_512_static_4 e3
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_143425', "Epoch1compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_8_1024_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_143337', "Epoch3compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_16_1024_static_4 e3
# KNOW_PATH = os.path.join(father_path, 'models/training_20260115_143120', "Epoch4compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # Simple_LLM[-1]_qy[pad]_24_1024_static_4 e4

# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_184819', "Epoch10compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_8_256_static_4 e10
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_184948', "Epoch4compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_16_256_static_4 e4
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_185121', "Epoch1compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_24_256_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_201559', "Epoch1compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_8_512_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_211313', "Epoch5compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_16_512_static_4 e5
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_215347', "Epoch8compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_24_512_static_4 e8
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_185418', "Epoch3compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_8_1024_static_4 e3
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_185543', "Epoch4compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_16_1024_static_4 e4
# KNOW_PATH = os.path.join(father_path, 'models/training_20260116_185755', "Epoch1compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # ACT_LLM[-1]_qy[pad]_24_1024_static_4 e1

DATA_PATH = os.path.join(grand_path, 'Data', 'CMtMedQA_test_sampled50_WIKGpath_gpt4o_mini_alpha_06_know_simple.json')
MODEL_PATH = os.path.join(father_path, 'models/training_20251202_102533', "Epoch9compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_2.pth")  # CMtMedQA_LLM[-1]_qy[pad]_16_1024_static_2_lr0.00005 e9

# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_143058', "Epoch1compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_8_256_static_4 e1
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_195948', "Epoch5compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_8_512_static_4 e5
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_143204', "Epoch5compressor_qwen25_15_model_8_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_8_1024_static_4 e5
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_143255', "Epoch9compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_16_256_static_4 e9
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_200040', "Epoch8compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_16_512_static_4 e8
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_143336', "Epoch7compressor_qwen25_15_model_16_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_16_1024_static_4 e7
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_143413', "compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_256_256_512_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_24_256_static_4 e10
# KNOW_PATH = os.path.join(father_path, 'models/training_20260118_200123', "Epoch5compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_512_512_1024_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_24_512_static_4 e5
KNOW_PATH = os.path.join(father_path, 'models/training_20260118_143456', "Epoch2compressor_qwen25_15_model_24_5e-05_tgt-1_qyembed_yid_1024_1024_2048_1.0_1.0_1.0_4_knowledge.pth")  # CMtMedQA_WIKG_LLM[-1]_qy[pad]_24_1024_static_4 e2



LLM_MODEL_NAME = "../Qwen2.5-1.5B-Instruct/"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###❗Lora
MODEL_TYPE = 'BASE'  ### LORA / BASE
LORA_ADAPTER_PATH = "../models/qwen2.5_v2-20250911-105207/checkpoint-8330"


# 测试配置
MAX_POS = 1024
MAX_LEN_HISTORY = 512 * 4
MAX_NEW_TOKENS = 512
NUM_TEST_SAMPLES = 100  # 测试样本数量

# 不同temperature的配置
TEMPERATURE_CONFIGS = [
    {"name": "Greedy", "temperature": 0.0, "top_p": 1.0},
    # {"name": "Low_Temp", "temperature": 0.3, "top_p": 0.9},
    # {"name": "Medium_Temp", "temperature": 0.7, "top_p": 0.9},
    # {"name": "High_Temp", "temperature": 1.0, "top_p": 0.9},
]


def load_peft_models(device):
    """加载LLM和compressor模型"""
    print("="*80)
    print("加载模型...")
    print("="*80)
    
    # 加载LLM
    print(f"\n加载LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    llm_model.eval()
    print("✓ LLM加载成功")
    
    llm_model = PeftModel.from_pretrained(llm_model, LORA_ADAPTER_PATH)
    print("✓ LoRA适配器加载成功")

    # 加载Mem_compressor
    print(f"加载Mem_compressor模型: {MODEL_PATH}")
    
    # 加载checkpoint获取超参数
    mem_checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 从LLM模型获取必要的维度参数
    input_dim = llm_model.config.hidden_size
    vocab_size = tokenizer.vocab_size
    # print("vocab_size:", vocab_size)
    
    # 从checkpoint中获取模型结构参数（如果保存了的话）
    # 否则使用默认值或从state_dict推断
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
    print(f"  - vocab_size: {vocab_size}")
    
    # 创建模型
    mem_compressor = DialogueCompressionModel(
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        vocab_size=vocab_size,
        num_iterations=num_iterations,
        beta=1.0,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        recon_weight=1.0,
        kl_weight=1.0,
        pred_weight=1.0,
        max_pos=512,
    ).to(device).to(torch.bfloat16)
    print(mem_compressor)
    # 加载权重
    mem_compressor.load_state_dict(mem_checkpoint['model_state_dict'])
    mem_compressor.eval()
    
    print(f"✓ Mem_Compressor加载成功 (num_slots={num_slots})")
    
    # 加载Know_compressor
    print(f"加载Know_compressor模型: {KNOW_PATH}")
    
    # 加载checkpoint获取超参数
    know_checkpoint = torch.load(KNOW_PATH, map_location=device)
    
    # 从checkpoint中获取模型结构参数（如果保存了的话）
    # 否则使用默认值或从state_dict推断
    if 'config' in know_checkpoint:
        config = know_checkpoint['config']
        slot_dim = config.get('slot_dim', input_dim)
        num_slots = config.get('num_slots', 16)
        num_iterations = config.get('num_iterations', 3)
        decoder_layers = config.get('decoder_layers', 2)
        decoder_heads = config.get('decoder_heads', 4)
    else:
        # 从state_dict推断num_slots（通过slot_embeddings的形状）
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
    
    print(f"✓ 检测到模型参数:")
    print(f"  - input_dim: {input_dim}")
    print(f"  - slot_dim: {slot_dim}")
    print(f"  - num_slots: {num_slots}")
    print(f"  - num_iterations: {num_iterations}")
    print(f"  - vocab_size: {vocab_size}")
    
    # 创建模型
    know_compressor = Know_DialogueCompressionModel(
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        vocab_size=vocab_size,
        num_iterations=num_iterations,
        beta=1.0,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        recon_weight=1.0,
        kl_weight=1.0,
        pred_weight=1.0,
        max_pos=MAX_POS,
    ).to(device).to(torch.bfloat16)
    print(know_compressor)
    # 加载权重
    know_compressor.load_state_dict(know_checkpoint['model_state_dict'])
    know_compressor.eval()
    
    print(f"✓ KNOW_Compressor加载成功 (num_slots={num_slots})")
    
    return tokenizer, llm_model, mem_compressor, know_compressor


def load_models(device):
    """加载LLM和compressor模型"""
    print("="*80)
    print("加载模型...")
    print("="*80)
    
    # 加载LLM
    print(f"\n加载LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    llm_model.eval()
    print("✓ LLM加载成功")
    
    # 加载Mem_compressor
    print(f"加载Mem_compressor模型: {MODEL_PATH}")
    
    # 加载checkpoint获取超参数
    mem_checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 从LLM模型获取必要的维度参数
    input_dim = llm_model.config.hidden_size
    vocab_size = tokenizer.vocab_size
    # print("vocab_size:", vocab_size)
    
    # 从checkpoint中获取模型结构参数（如果保存了的话）
    # 否则使用默认值或从state_dict推断
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
    print(f"  - vocab_size: {vocab_size}")
    
    # 创建模型
    mem_compressor = DialogueCompressionModel(
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        vocab_size=vocab_size,
        num_iterations=num_iterations,
        beta=1.0,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        recon_weight=1.0,
        kl_weight=1.0,
        pred_weight=1.0,
        # max_pos=512,
        max_pos=1024,
    ).to(device).to(torch.bfloat16)
    print(mem_compressor)
    # 加载权重
    mem_compressor.load_state_dict(mem_checkpoint['model_state_dict'])
    mem_compressor.eval()
    
    print(f"✓ Mem_Compressor加载成功 (num_slots={num_slots})")
    
    # 加载Know_compressor
    print(f"加载Know_compressor模型: {KNOW_PATH}")
    
    # 加载checkpoint获取超参数
    know_checkpoint = torch.load(KNOW_PATH, map_location=device)
    
    # 从checkpoint中获取模型结构参数（如果保存了的话）
    # 否则使用默认值或从state_dict推断
    if 'config' in know_checkpoint:
        config = know_checkpoint['config']
        slot_dim = config.get('slot_dim', input_dim)
        num_slots = config.get('num_slots', 16)
        num_iterations = config.get('num_iterations', 3)
        decoder_layers = config.get('decoder_layers', 2)
        decoder_heads = config.get('decoder_heads', 4)
    else:
        # 从state_dict推断num_slots（通过slot_embeddings的形状）
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
    
    print(f"✓ 检测到模型参数:")
    print(f"  - input_dim: {input_dim}")
    print(f"  - slot_dim: {slot_dim}")
    print(f"  - num_slots: {num_slots}")
    print(f"  - num_iterations: {num_iterations}")
    print(f"  - vocab_size: {vocab_size}")
    
    # 创建模型
    know_compressor = Know_DialogueCompressionModel(
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        vocab_size=vocab_size,
        num_iterations=num_iterations,
        beta=1.0,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        recon_weight=1.0,
        kl_weight=1.0,
        pred_weight=1.0,
        max_pos=MAX_POS,
    ).to(device).to(torch.bfloat16)
    print(know_compressor)
    # 加载权重
    know_compressor.load_state_dict(know_checkpoint['model_state_dict'])
    know_compressor.eval()
    
    print(f"✓ KNOW_Compressor加载成功 (num_slots={num_slots})")
    
    return tokenizer, llm_model, mem_compressor, know_compressor


def extract_history_features(llm_model, tokenizer, history_text, max_len, device):
    """提取历史对话特征"""
    ###❓历史信息处理时间记录
    encoding = tokenizer(
        history_text,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        history_features = outputs.hidden_states[-1]
    
    return history_features, attention_mask


def compress_history_to_slots(compressor, history_features, history_mask):
    """压缩历史为slots"""
    with torch.no_grad():
        slots, z, mu, logvar = compressor.variational_slot_attention(
            history_features, history_mask, training=False
        )
    return slots


def generate_text(llm_model, tokenizer, inputs_embeds, attention_mask, 
                  max_new_tokens, temperature, top_p, device):
    """
    生成文本
    temperature=0时使用greedy decoding
    """
    with torch.no_grad():
        try:
            if temperature == 0:
                # Greedy decoding
                generated_ids = llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            else:
                # Sampling
                generated_ids = llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            if isinstance(generated_ids, torch.Tensor):
                if generated_ids.dim() == 2:
                    generated_ids = generated_ids[0]    # 只看batch的第一个样本
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                generated_text = ""
                
        except Exception as e:
            print(f"Warning: generate() failed: {e}")
            generated_text = "[生成失败]"
    
    return generated_text


def test_method_1_slots_plus_q(memory_compressor, know_compressor, llm_model, tokenizer, example, 
                                temperature, top_p, device):
    """方法1: slots + question"""
    history = example.get('history', [])
    current_q = example.get('current_q', '')
    # knowledge = example.get('knowledge', '')
    knowledge = example['knowledge']
    history_text = ' '.join(history) if isinstance(history, list) else history
    
    # 提取历史特征并压缩为slots
    history_features, history_mask = extract_history_features(
        llm_model, tokenizer, history_text, MAX_LEN_HISTORY, device
    )
    memory_slots = compress_history_to_slots(memory_compressor, history_features, history_mask)
    
    # 知识压缩成slots
    knowledge_features, knowledge_mask = extract_history_features(
        llm_model, tokenizer, knowledge, MAX_LEN_HISTORY, device
    )
    know_slots = compress_history_to_slots(know_compressor, knowledge_features, knowledge_mask)
    
    # Tokenize问题
    ###❗聊天模板
    messages = [{"role": "user", "content": current_q}]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    q_encoding = tokenizer(full_text, add_special_tokens=False, return_tensors='pt')
    q_ids = q_encoding['input_ids'].to(device)
    q_attention_mask = q_encoding['attention_mask'].to(device)
    
    # 获取问题embeddings
    with torch.no_grad():
        q_embeddings = llm_model.get_input_embeddings()(q_ids)
    
    # 拼接: [slots] + [question]
    inputs_embeds = torch.cat([memory_slots, know_slots, q_embeddings], dim=1)
    slot_mask = torch.ones(memory_slots.shape[0], memory_slots.shape[1]+know_slots.shape[1], dtype=torch.long, device=device)
    attention_mask = torch.cat([slot_mask, q_attention_mask], dim=1)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


def test_method_2_history_plus_q(llm_model, tokenizer, example,
                                   temperature, top_p, device):
    """方法2: history_text + question (纯文本)"""
    history = example.get('history', [])
    current_q = example.get('current_q', '')
    history_text = ' '.join(history) if isinstance(history, list) else history
    
    # # 拼接历史和问题
    # full_text = f"{history_text}\n用户: {current_q}\n助手:"
    ###❗聊天模板
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(history)]
    messages.append({"role": "user", "content": current_q})
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    encoding = tokenizer(
        full_text,
        max_length=MAX_LEN_HISTORY,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取embeddings
    with torch.no_grad():
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


def test_method_2_history_1024_plus_q(llm_model, tokenizer, example,
                                   temperature, top_p, device):
    """方法2: history_text + question (纯文本)"""
    history = example.get('history', [])
    current_q = example.get('current_q', '')
    history_text = ' '.join(history) if isinstance(history, list) else history
    
    # # 拼接历史和问题
    # full_text = f"{history_text}\n用户: {current_q}\n助手:"
    ###❗聊天模板
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(history)]
    messages.append({"role": "user", "content": current_q})
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    encoding = tokenizer(
        full_text,
        max_length=1024,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取embeddings
    with torch.no_grad():
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


def test_method_2_history_512_plus_q(llm_model, tokenizer, example,
                                   temperature, top_p, device):
    """方法2: history_text + question (纯文本)"""
    history = example.get('history', [])
    current_q = example.get('current_q', '')
    history_text = ' '.join(history) if isinstance(history, list) else history
    
    # # 拼接历史和问题
    # full_text = f"{history_text}\n用户: {current_q}\n助手:"
    ###❗聊天模板
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(history)]
    messages.append({"role": "user", "content": current_q})
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    encoding = tokenizer(
        full_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取embeddings
    with torch.no_grad():
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


def test_method_2_history_256_plus_q(llm_model, tokenizer, example,
                                   temperature, top_p, device):
    """方法2: history_text + question (纯文本)"""
    history = example.get('history', [])
    current_q = example.get('current_q', '')
    history_text = ' '.join(history) if isinstance(history, list) else history
    
    # # 拼接历史和问题
    # full_text = f"{history_text}\n用户: {current_q}\n助手:"
    ###❗聊天模板
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(history)]
    messages.append({"role": "user", "content": current_q})
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    encoding = tokenizer(
        full_text,
        max_length=256,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取embeddings
    with torch.no_grad():
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


def test_method_3_q_only(llm_model, tokenizer, example,
                         temperature, top_p, device):
    """方法3: question only (无历史)"""
    current_q = example.get('current_q', '')
    
    # # 只使用问题
    # full_text = f"用户: {current_q}\n助手:"
    ###❗聊天模板
    messages = [{"role": "user", "content": current_q}]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    encoding = tokenizer(full_text, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取embeddings
    with torch.no_grad():
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


def test_method_4_summary_plus_q(llm_model, tokenizer, example,
                                   temperature, top_p, device):
    """方法4: summary_text + question (纯文本)"""
    summary = example.get('summary', [])
    current_q = example.get('current_q', '')
    # history_text = ' '.join(history) if isinstance(history, list) else history
    
    # # 拼接历史和问题
    # full_text = f"{history_text}\n用户: {current_q}\n助手:"
    system_prompt = f"""你是一个智能助手，你的任务是基于一段对话的历史摘要，连贯且有逻辑地回答用户当前的问题。

这是到目前为止的对话摘要：
<summary>
{summary}
</summary>

请严格根据以上摘要和用户接下来的问题进行回复，确保内容的相关性和连贯性。
"""
    ###❗聊天模板
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": current_q})
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    encoding = tokenizer(
        full_text,
        max_length=MAX_LEN_HISTORY,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取embeddings
    with torch.no_grad():
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    
    # 生成
    generated_text = generate_text(
        llm_model, tokenizer, inputs_embeds, attention_mask,
        MAX_NEW_TOKENS, temperature, top_p, device
    )
    
    return generated_text


### 中文分词✅
def calculate_metrics(generated_text, reference_text, tokenizer):
    """
    计算评估指标：包括基础指标、BLEU和ROUGE、Bertscore
    """
    metrics = {}
    
    # 1. 基础指标
    metrics['length'] = len(generated_text)
    metrics['len_ratio'] = len(generated_text) / max(len(reference_text), 1)

    # 中文分词
    gen_tokens = tokenizer.tokenize(generated_text)
    ref_tokens = tokenizer.tokenize(reference_text)

    gen_set = set(gen_tokens)
    ref_set = set(ref_tokens)
    overlap = len(gen_set & ref_set) / max(len(ref_set), 1)
    metrics['token_overlap'] = overlap
    
    # 是否为空
    metrics['is_empty'] = len(generated_text.strip()) == 0
    
    # 2. BLEU、ROUGE、Bertscore指标
    if METRICS_AVAILABLE and len(gen_tokens) > 0 and len(ref_tokens) > 0:
        # try:
        # BLEU-1, BLEU-2, BLEU-3, BLEU-4
        smoothing = SmoothingFunction().method1
        metrics['bleu_1'] = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        metrics['bleu_2'] = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        metrics['bleu_3'] = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        metrics['bleu_4'] = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        # ROUGE
        rouge = Rouge()
        gen_text_spaced = " ".join(gen_tokens)
        ref_text_spaced = " ".join(ref_tokens)
        rouge_scores = rouge.get_scores(gen_text_spaced, ref_text_spaced)[0]
        metrics['rouge_1_f'] = rouge_scores['rouge-1']['f']
        metrics['rouge_1_p'] = rouge_scores['rouge-1']['p']
        metrics['rouge_1_r'] = rouge_scores['rouge-1']['r']
        metrics['rouge_2_f'] = rouge_scores['rouge-2']['f']
        metrics['rouge_2_p'] = rouge_scores['rouge-2']['p']
        metrics['rouge_2_r'] = rouge_scores['rouge-2']['r']
        metrics['rouge_l_f'] = rouge_scores['rouge-l']['f']
        metrics['rouge_l_p'] = rouge_scores['rouge-l']['p']
        metrics['rouge_l_r'] = rouge_scores['rouge-l']['r']
        
        # Bertscore
        BERT_MAX_TOKENS = 500
        P, R, F1 = bert_score(
            [generated_text[:BERT_MAX_TOKENS]], [reference_text[:BERT_MAX_TOKENS]],
            # model_type="hfl/chinese-roberta-wwm-ext",
            model_type="/mnt/disk1/hf_models/chinese-roberta-wwm-ext",
            num_layers=12,
            lang="zh",
            verbose=False,
            device=DEVICE,
        )
        metrics['bert_score_p'] = P.mean().item()
        metrics['bert_score_r'] = R.mean().item()
        metrics['bert_score_f'] = F1.mean().item()
        # except Exception as e:
        #     print(f"Warning: Error calculating BLEU/ROUGE/BERT-Score: {e}")
        #     # 如果计算失败，设置默认值
        #     metrics['bleu_1'] = 0.0
        #     metrics['bleu_2'] = 0.0
        #     metrics['bleu_3'] = 0.0
        #     metrics['bleu_4'] = 0.0
        #     metrics['rouge_1_f'] = 0.0
        #     metrics['rouge_1_p'] = 0.0
        #     metrics['rouge_1_r'] = 0.0
        #     metrics['rouge_2_f'] = 0.0
        #     metrics['rouge_2_p'] = 0.0
        #     metrics['rouge_2_r'] = 0.0
        #     metrics['rouge_l_f'] = 0.0
        #     metrics['rouge_l_p'] = 0.0
        #     metrics['rouge_l_r'] = 0.0
        #     metrics['bert_score_p'] = 0.0
        #     metrics['bert_score_r'] = 0.0
        #     metrics['bert_score_f'] = 0.0
    else:
        # BLEU/ROUGE不可用时设置默认值
        metrics['bleu_1'] = 0.0
        metrics['bleu_2'] = 0.0
        metrics['bleu_3'] = 0.0
        metrics['bleu_4'] = 0.0
        metrics['rouge_1_f'] = 0.0
        metrics['rouge_1_p'] = 0.0
        metrics['rouge_1_r'] = 0.0
        metrics['rouge_2_f'] = 0.0
        metrics['rouge_2_p'] = 0.0
        metrics['rouge_2_r'] = 0.0
        metrics['rouge_l_f'] = 0.0
        metrics['rouge_l_p'] = 0.0
        metrics['rouge_l_r'] = 0.0
        metrics['bert_score_p'] = 0.0
        metrics['bert_score_r'] = 0.0
        metrics['bert_score_f'] = 0.0
    
    return metrics


def test_single_example_all_methods(memory_compressor, know_compressor, llm_model, tokenizer, example, 
                                    temp_config, device):
    """对单个样本测试所有三种方法"""
    temperature = temp_config['temperature']
    top_p = temp_config['top_p']
    
    results = {}
    
    # 方法1: slots + q
    results['method_1'] = test_method_1_slots_plus_q(
        memory_compressor, know_compressor, llm_model, tokenizer, example, temperature, top_p, device
    )
    
    # # 方法2: history + q
    # results['method_2'] = test_method_2_history_plus_q(
    #     llm_model, tokenizer, example, temperature, top_p, device
    # )
    
    # # 方法2_1024: history + q
    # results['method_2_1024'] = test_method_2_history_1024_plus_q(
    #     llm_model, tokenizer, example, temperature, top_p, device
    # )
    
    # # 方法2_512: history + q
    # results['method_2_512'] = test_method_2_history_512_plus_q(
    #     llm_model, tokenizer, example, temperature, top_p, device
    # )
    
    # # 方法2_256: history + q
    # results['method_2_256'] = test_method_2_history_256_plus_q(
    #     llm_model, tokenizer, example, temperature, top_p, device
    # )
    
    # # 方法3: q only
    # results['method_3'] = test_method_3_q_only(
    #     llm_model, tokenizer, example, temperature, top_p, device
    # )
    
    # # 方法4: summary + q
    # results['method_4'] = test_method_4_summary_plus_q(
    #     llm_model, tokenizer, example, temperature, top_p, device
    # )
    
    return results


def save_all_generated_texts(all_results, prefix, postfix):
    """保存所有生成内容到易读的文本文件"""
    for temp_result in all_results:
        config = temp_result['config']
        samples = temp_result['samples']
        config_name = config['name']
        
        # 为每个temperature配置创建一个文件
        output_file = f"{prefix}generated_texts_{config_name}_{postfix}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"生成内容汇总 - {config_name}\n")
            f.write(f"Temperature: {config['temperature']}, Top-p: {config['top_p']}\n")
            f.write("="*100 + "\n\n")
            
            for sample in samples:
                f.write(f"\n{'='*100}\n")
                f.write(f"样本 ID: {sample['id']}\n")
                f.write(f"{'='*100}\n\n")
                
                # 历史信息（如果有）
                f.write(f"【问题】\n{sample['question']}\n\n")
                
                # 标准答案
                f.write(f"【标准答案】\n{sample['ground_truth']}\n\n")
                
                # 方法1: Slots + Q
                f.write(f"【方法1: Slots + Q】\n")
                f.write(f"生成内容: {sample['method_1']['text']}\n")
                f.write(f"指标:\n")
                for metric_name, metric_value in sample['method_1']['metrics'].items():
                    if metric_name != 'is_empty':
                        f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                f.write("\n")
                
                # # 方法2: History + Q
                # f.write(f"【方法2: History + Q】\n")
                # f.write(f"生成内容: {sample['method_2']['text']}\n")
                # f.write(f"指标:\n")
                # for metric_name, metric_value in sample['method_2']['metrics'].items():
                #     if metric_name != 'is_empty':
                #         f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                # f.write("\n")
                
                # # 方法2: History + Q 1024
                # f.write(f"【方法2: History + Q 1024】\n")
                # f.write(f"生成内容: {sample['method_2_1024']['text']}\n")
                # f.write(f"指标:\n")
                # for metric_name, metric_value in sample['method_2_1024']['metrics'].items():
                #     if metric_name != 'is_empty':
                #         f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                # f.write("\n")
                
                # # 方法2: History + Q 512
                # f.write(f"【方法2: History + Q 512】\n")
                # f.write(f"生成内容: {sample['method_2_512']['text']}\n")
                # f.write(f"指标:\n")
                # for metric_name, metric_value in sample['method_2_512']['metrics'].items():
                #     if metric_name != 'is_empty':
                #         f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                # f.write("\n")
                
                # # 方法2: History + Q 256
                # f.write(f"【方法2: History + Q 256】\n")
                # f.write(f"生成内容: {sample['method_2_256']['text']}\n")
                # f.write(f"指标:\n")
                # for metric_name, metric_value in sample['method_2_256']['metrics'].items():
                #     if metric_name != 'is_empty':
                #         f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                # f.write("\n")
                
                # # 方法3: Q Only
                # f.write(f"【方法3: Q Only】\n")
                # f.write(f"生成内容: {sample['method_3']['text']}\n")
                # f.write(f"指标:\n")
                # for metric_name, metric_value in sample['method_3']['metrics'].items():
                #     if metric_name != 'is_empty':
                #         f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                # f.write("\n")
                
                # # 方法4: Summary + Q
                # f.write(f"【方法4: Summary + Q】\n")
                # f.write(f"生成内容: {sample['method_4']['text']}\n")
                # f.write(f"指标:\n")
                # for metric_name, metric_value in sample['method_4']['metrics'].items():
                #     if metric_name != 'is_empty':
                #         f.write(f"  - {metric_name}: {metric_value:.4f}\n")
                # f.write("\n")
                
                f.write(f"{'-'*100}\n\n")
        
        print(f"✓ 已保存生成内容到: {output_file}")
    
    # 额外保存一个汇总CSV文件（方便Excel打开）
    save_summary_csv(all_results, prefix, postfix)


def save_summary_csv(all_results, prefix, postfix):
    """保存指标汇总到CSV文件"""
    import csv
    
    csv_file = f"{prefix}comparison_results_summary_{postfix}.csv"  ### 
    
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig for Excel
        writer = csv.writer(f)
        
        # 写入表头
        header = [
            'Temperature_Config', 'Sample_ID', 'Method',
            'Length', 'Len_Ratio', 'Token_Overlap',
            'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
            'ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F',
            'BERT_Score_f',
            'Is_Empty'
        ]
        writer.writerow(header)
        
        # 写入数据
        for temp_result in all_results:
            config_name = temp_result['config']['name']
            samples = temp_result['samples']
            
            for sample in samples:
                sample_id = sample['id']
                
                # 方法1
                m1_metrics = sample['method_1']['metrics']
                writer.writerow([
                    config_name, sample_id, 'Slots+Q',
                    m1_metrics['length'], m1_metrics['len_ratio'], m1_metrics['token_overlap'],
                    m1_metrics['bleu_1'], m1_metrics['bleu_2'], m1_metrics['bleu_3'], m1_metrics['bleu_4'],
                    m1_metrics['rouge_1_f'], m1_metrics['rouge_2_f'], m1_metrics['rouge_l_f'],
                    m1_metrics['bert_score_f'],
                    m1_metrics['is_empty']
                ])
                
                # # 方法2
                # m2_metrics = sample['method_2']['metrics']
                # writer.writerow([
                #     config_name, sample_id, 'History+Q',
                #     m2_metrics['length'], m2_metrics['len_ratio'], m2_metrics['token_overlap'],
                #     m2_metrics['bleu_1'], m2_metrics['bleu_2'], m2_metrics['bleu_3'], m2_metrics['bleu_4'],
                #     m2_metrics['rouge_1_f'], m2_metrics['rouge_2_f'], m2_metrics['rouge_l_f'],
                #     m2_metrics['bert_score_f'],
                #     m2_metrics['is_empty']
                # ])
                
                # # 方法2 1024
                # m2_1024_metrics = sample['method_2_1024']['metrics']
                # writer.writerow([
                #     config_name, sample_id, 'History+Q 1024',
                #     m2_1024_metrics['length'], m2_1024_metrics['len_ratio'], m2_1024_metrics['token_overlap'],
                #     m2_1024_metrics['bleu_1'], m2_1024_metrics['bleu_2'], m2_1024_metrics['bleu_3'], m2_1024_metrics['bleu_4'],
                #     m2_1024_metrics['rouge_1_f'], m2_1024_metrics['rouge_2_f'], m2_1024_metrics['rouge_l_f'],
                #     m2_1024_metrics['bert_score_f'],
                #     m2_1024_metrics['is_empty']
                # ])
                
                # # 方法2 512
                # m2_512_metrics = sample['method_2_512']['metrics']
                # writer.writerow([
                #     config_name, sample_id, 'History+Q 512',
                #     m2_512_metrics['length'], m2_512_metrics['len_ratio'], m2_512_metrics['token_overlap'],
                #     m2_512_metrics['bleu_1'], m2_512_metrics['bleu_2'], m2_512_metrics['bleu_3'], m2_512_metrics['bleu_4'],
                #     m2_512_metrics['rouge_1_f'], m2_512_metrics['rouge_2_f'], m2_512_metrics['rouge_l_f'],
                #     m2_512_metrics['bert_score_f'],
                #     m2_512_metrics['is_empty']
                # ])
                
                # # 方法2 256
                # m2_256_metrics = sample['method_2_256']['metrics']
                # writer.writerow([
                #     config_name, sample_id, 'History+Q 256',
                #     m2_256_metrics['length'], m2_256_metrics['len_ratio'], m2_256_metrics['token_overlap'],
                #     m2_256_metrics['bleu_1'], m2_256_metrics['bleu_2'], m2_256_metrics['bleu_3'], m2_256_metrics['bleu_4'],
                #     m2_256_metrics['rouge_1_f'], m2_256_metrics['rouge_2_f'], m2_256_metrics['rouge_l_f'],
                #     m2_256_metrics['bert_score_f'],
                #     m2_256_metrics['is_empty']
                # ])
                
                # # 方法3
                # m3_metrics = sample['method_3']['metrics']
                # writer.writerow([
                #     config_name, sample_id, 'Q_Only',
                #     m3_metrics['length'], m3_metrics['len_ratio'], m3_metrics['token_overlap'],
                #     m3_metrics['bleu_1'], m3_metrics['bleu_2'], m3_metrics['bleu_3'], m3_metrics['bleu_4'],
                #     m3_metrics['rouge_1_f'], m3_metrics['rouge_2_f'], m3_metrics['rouge_l_f'],
                #     m3_metrics['bert_score_f'],
                #     m3_metrics['is_empty']
                # ])
                
                # # 方法4
                # m4_metrics = sample['method_4']['metrics']
                # writer.writerow([
                #     config_name, sample_id, 'Summary+Q',
                #     m4_metrics['length'], m4_metrics['len_ratio'], m4_metrics['token_overlap'],
                #     m4_metrics['bleu_1'], m4_metrics['bleu_2'], m4_metrics['bleu_3'], m4_metrics['bleu_4'],
                #     m4_metrics['rouge_1_f'], m4_metrics['rouge_2_f'], m4_metrics['rouge_l_f'],
                #     m4_metrics['bert_score_f'],
                #     m4_metrics['is_empty']
                # ])
    
    print(f"✓ 已保存CSV汇总到: {csv_file}")


def validate_compressor(memory_compressor, know_compressor, compressor_name, llm_model, tokenizer, epoch_num, timestamp):
    """每个Epoch结束验证"""
    print("="*80)
    print("Epoch结束，开始模型验证--对比测试：Slots vs History vs Question-Only")
    print("="*80)
    
    # 1. 基本配置
    VALIDATE_CONFIGS = [
        {"name": "Greedy", "temperature": 0.0, "top_p": 1.0},
        # {"name": "Low_Temp", "temperature": 0.3, "top_p": 0.9},
        # {"name": "Medium_Temp", "temperature": 0.7, "top_p": 0.9},
        # {"name": "High_Temp", "temperature": 1.0, "top_p": 0.9},
    ]
    
    # 2. 加载数据
    print(f"\n加载测试数据: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_data = test_data[:NUM_TEST_SAMPLES//2]  # ！初步看效果，可设置少
    print(f"✓ 测试 {len(test_data)} 个样本")
    
    # 3. 准备结果存储
    all_results = []
    
    # 4. 对每个temperature配置进行测试
    for temp_config in VALIDATE_CONFIGS:
        print("\n" + "="*80)
        print(f"Temperature配置: {temp_config['name']}")
        print(f"  temperature={temp_config['temperature']}, top_p={temp_config['top_p']}")
        print("="*80)
        
        temp_results = {
            'config': temp_config,
            'samples': []
        }
        
        # 对每个样本进行测试
        for idx, example in enumerate(tqdm(test_data, desc=f"Testing {temp_config['name']}")):
            ground_truth = example.get('response', '')
            
            # 测试三种方法
            generated_texts = test_single_example_all_methods(
                memory_compressor, know_compressor, llm_model, tokenizer, example, temp_config, DEVICE
            )
            
            # 计算指标
            sample_result = {
                'id': idx,
                'question': example.get('current_q', ''),
                'ground_truth': ground_truth,
                'method_1': {
                    'text': generated_texts['method_1'],
                    'metrics': calculate_metrics(generated_texts['method_1'], ground_truth, tokenizer)
                },
                # 'method_2': {
                #     'text': generated_texts['method_2'],
                #     'metrics': calculate_metrics(generated_texts['method_2'], ground_truth, tokenizer)
                # },
                # 'method_2_1024': {
                #     'text': generated_texts['method_2_1024'],
                #     'metrics': calculate_metrics(generated_texts['method_2_1024'], ground_truth, tokenizer)
                # },
                # 'method_2_512': {
                #     'text': generated_texts['method_2_512'],
                #     'metrics': calculate_metrics(generated_texts['method_2_512'], ground_truth, tokenizer)
                # },
                # 'method_2_256': {
                #     'text': generated_texts['method_2_256'],
                #     'metrics': calculate_metrics(generated_texts['method_2_256'], ground_truth, tokenizer)
                # },
                # 'method_3': {
                #     'text': generated_texts['method_3'],
                #     'metrics': calculate_metrics(generated_texts['method_3'], ground_truth, tokenizer)
                # },
                # 'method_4': {
                #     'text': generated_texts['method_4'],
                #     'metrics': calculate_metrics(generated_texts['method_4'], ground_truth, tokenizer)
                # }
            }
            
            temp_results['samples'].append(sample_result)
        
        all_results.append(temp_results)
    
    ###❗5. 保存详细结果
    # postfix = MODEL_PATH.split("compressor_")[-1].split(".pth")[0]
    # output_file = f"comparison_results_detailed_{postfix}.json"
    prefix = "../val/CMtMedQA_Mem_Epoch9+Know_LLM[-1]_qy[pad]_24_512_static_4_lr5e-5/"
    postfix = compressor_name.split("compressor_")[-1].split(".pth")[0] + f"_epoch{epoch_num}_{timestamp}"
    output_file = f"{prefix}comparison_results_detailed_{postfix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 详细结果已保存到: {output_file}")
    
    # 6. 保存所有生成内容
    save_all_generated_texts(all_results, prefix, postfix)
    
    # 6. 打印统计摘要
    print("\n" + "="*80)
    print("结果摘要")
    print("="*80)
    
    for temp_result in all_results:
        config = temp_result['config']
        samples = temp_result['samples']
        
        print(f"\n### {config['name']} (temp={config['temperature']}, top_p={config['top_p']})")
        print("-"*80)
        
        for method_name, method_label in [
            ('method_1', 'Slots + Q'),
            # ('method_2', 'History + Q'),
            # ('method_2_1024', 'History + Q 1024'),
            # ('method_2_512', 'History + Q 512'),
            # ('method_2_256', 'History + Q 256'),
            # ('method_3', 'Q Only'),
            # ('method_4', 'Summary + Q')
        ]:
            lengths = [s[method_name]['metrics']['length'] for s in samples]
            len_ratios = [s[method_name]['metrics']['len_ratio'] for s in samples]
            token_overlaps = [s[method_name]['metrics']['token_overlap'] for s in samples]
            empty_count = sum(1 for s in samples if s[method_name]['metrics']['is_empty'])
            
            # BLEU scores
            bleu_1_scores = [s[method_name]['metrics']['bleu_1'] for s in samples]
            bleu_2_scores = [s[method_name]['metrics']['bleu_2'] for s in samples]
            bleu_3_scores = [s[method_name]['metrics']['bleu_3'] for s in samples]
            bleu_4_scores = [s[method_name]['metrics']['bleu_4'] for s in samples]
            
            # ROUGE scores
            rouge_1_f_scores = [s[method_name]['metrics']['rouge_1_f'] for s in samples]
            rouge_2_f_scores = [s[method_name]['metrics']['rouge_2_f'] for s in samples]
            rouge_l_f_scores = [s[method_name]['metrics']['rouge_l_f'] for s in samples]

            # BERT scores
            bert_f_scores = [s[method_name]['metrics']['bert_score_f'] for s in samples]
            
            print(f"\n{method_label}:")
            print(f"  平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            print(f"  长度比例: {np.mean(len_ratios):.2f} ± {np.std(len_ratios):.2f}")
            print(f"  Token重叠: {np.mean(token_overlaps):.2f} ± {np.std(token_overlaps):.2f}")
            if METRICS_AVAILABLE:
                print(f"  BLEU-1: {np.mean(bleu_1_scores):.4f} ± {np.std(bleu_1_scores):.4f}")
                print(f"  BLEU-2: {np.mean(bleu_2_scores):.4f} ± {np.std(bleu_2_scores):.4f}")
                print(f"  BLEU-3: {np.mean(bleu_3_scores):.4f} ± {np.std(bleu_3_scores):.4f}")
                print(f"  BLEU-4: {np.mean(bleu_4_scores):.4f} ± {np.std(bleu_4_scores):.4f}")
                print(f"  ROUGE-1: {np.mean(rouge_1_f_scores):.4f} ± {np.std(rouge_1_f_scores):.4f}")
                print(f"  ROUGE-2: {np.mean(rouge_2_f_scores):.4f} ± {np.std(rouge_2_f_scores):.4f}")
                print(f"  ROUGE-L: {np.mean(rouge_l_f_scores):.4f} ± {np.std(rouge_l_f_scores):.4f}")
                print(f"  BERT_Score-F1: {np.mean(bert_f_scores):.4f} ± {np.std(bert_f_scores):.4f}")
            print(f"  空回复数: {empty_count}/{len(samples)}")
    
    # 7. 打印几个具体示例
    print("\n" + "="*80)
    print("示例对比 (使用Greedy配置)")
    print("="*80)
    
    greedy_results = all_results[0]  # 第一个配置是Greedy
    for i in range(min(3, len(greedy_results['samples']))):
        sample = greedy_results['samples'][i]
        print(f"\n样本 {i+1}:")
        print(f"问题: {sample['question'][:100]}...")
        print(f"\n标准答案: {sample['ground_truth'][:150]}...")
        print(f"\n方法1 (Slots+Q): {sample['method_1']['text'][:150]}...")
        # print(f"方法2 (History+Q): {sample['method_2']['text'][:150]}...")
        # print(f"方法2 (History+Q 1024): {sample['method_2_1024']['text'][:150]}...")
        # print(f"方法2 (History+Q 512): {sample['method_2_512']['text'][:150]}...")
        # print(f"方法2 (History+Q 256): {sample['method_2_256']['text'][:150]}...")
        # print(f"方法3 (Q Only): {sample['method_3']['text'][:150]}...")
        # print(f"方法4 (Summary+Q): {sample['method_4']['text'][:150]}...")
        print("-"*80)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


def main():
    print("="*80)
    print("对比测试：Slots vs History vs Question-Only")
    print("="*80)
    
    ###❗1. 加载模型
    if MODEL_TYPE == 'LORA':
        tokenizer, llm_model, mem_compressor, know_compressor = load_peft_models(DEVICE)
    else:
        tokenizer, llm_model, mem_compressor, know_compressor = load_models(DEVICE)
    
    # 2. 加载数据
    print(f"\n加载测试数据: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    # test_data = test_data[:NUM_TEST_SAMPLES]
    test_data = test_data[:]
    print(f"✓ 测试 {len(test_data)} 个样本")
    
    # 3. 准备结果存储
    all_results = []
    
    # 4. 对每个temperature配置进行测试
    for temp_config in TEMPERATURE_CONFIGS:
        print("\n" + "="*80)
        print(f"Temperature配置: {temp_config['name']}")
        print(f"  temperature={temp_config['temperature']}, top_p={temp_config['top_p']}")
        print("="*80)
        
        temp_results = {
            'config': temp_config,
            'samples': []
        }
        
        # 对每个样本进行测试
        for idx, example in enumerate(tqdm(test_data, desc=f"Testing {temp_config['name']}")):
            ground_truth = example.get('response', '')
            
            # 测试三种方法
            generated_texts = test_single_example_all_methods(
                mem_compressor, know_compressor, llm_model, tokenizer, example, temp_config, DEVICE
            )
            
            # 计算指标
            sample_result = {
                'id': idx,
                'question': example.get('current_q', ''),
                'ground_truth': ground_truth,
                'method_1': {
                    'text': generated_texts['method_1'],
                    'metrics': calculate_metrics(generated_texts['method_1'], ground_truth, tokenizer)
                },
                # 'method_2': {
                #     'text': generated_texts['method_2'],
                #     'metrics': calculate_metrics(generated_texts['method_2'], ground_truth, tokenizer)
                # },
                # 'method_2_1024': {
                #     'text': generated_texts['method_2_1024'],
                #     'metrics': calculate_metrics(generated_texts['method_2_1024'], ground_truth, tokenizer)
                # },
                # 'method_2_512': {
                #     'text': generated_texts['method_2_512'],
                #     'metrics': calculate_metrics(generated_texts['method_2_512'], ground_truth, tokenizer)
                # },
                # 'method_2_256': {
                #     'text': generated_texts['method_2_256'],
                #     'metrics': calculate_metrics(generated_texts['method_2_256'], ground_truth, tokenizer)
                # },
                # 'method_3': {
                #     'text': generated_texts['method_3'],
                #     'metrics': calculate_metrics(generated_texts['method_3'], ground_truth, tokenizer)
                # },
                # 'method_4': {
                #     'text': generated_texts['method_4'],
                #     'metrics': calculate_metrics(generated_texts['method_4'], ground_truth, tokenizer)
                # },
            }
            
            temp_results['samples'].append(sample_result)
        
        all_results.append(temp_results)
    
    ###❗5. 保存详细结果
    prefix = "../CMtMedQA_Mem_Epoch9+Know_LLM[-1]_qy[pad]_24_1024_static_4_lr5e-5/"
    postfix = KNOW_PATH.split("compressor_")[-1].split(".pth")[0]
    output_file = prefix + f"comparison_results_detailed__{postfix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 详细结果已保存到: {output_file}")
    
    # 6. 保存所有生成内容
    save_all_generated_texts(all_results, prefix, postfix)
    
    # 6. 打印统计摘要
    print("\n" + "="*80)
    print("结果摘要")
    print("="*80)
    
    for temp_result in all_results:
        config = temp_result['config']
        samples = temp_result['samples']
        
        print(f"\n### {config['name']} (temp={config['temperature']}, top_p={config['top_p']})")
        print("-"*80)
        
        for method_name, method_label in [
            ('method_1', 'Slots + Q'),
            # ('method_2', 'History + Q'),
            # ('method_2_1024', 'History + Q 1024'),
            # ('method_2_512', 'History + Q 512'),
            # ('method_2_256', 'History + Q 256'),
            # ('method_3', 'Q Only'),
            # ('method_4', 'Summary + Q'),
        ]:
            lengths = [s[method_name]['metrics']['length'] for s in samples]
            len_ratios = [s[method_name]['metrics']['len_ratio'] for s in samples]
            token_overlaps = [s[method_name]['metrics']['token_overlap'] for s in samples]
            empty_count = sum(1 for s in samples if s[method_name]['metrics']['is_empty'])
            
            # BLEU scores
            bleu_1_scores = [s[method_name]['metrics']['bleu_1'] for s in samples]
            bleu_2_scores = [s[method_name]['metrics']['bleu_2'] for s in samples]
            bleu_3_scores = [s[method_name]['metrics']['bleu_3'] for s in samples]
            bleu_4_scores = [s[method_name]['metrics']['bleu_4'] for s in samples]
            
            # ROUGE scores
            rouge_1_f_scores = [s[method_name]['metrics']['rouge_1_f'] for s in samples]
            rouge_2_f_scores = [s[method_name]['metrics']['rouge_2_f'] for s in samples]
            rouge_l_f_scores = [s[method_name]['metrics']['rouge_l_f'] for s in samples]

            # BERT scores
            bert_f_scores = [s[method_name]['metrics']['bert_score_f'] for s in samples]
            
            print(f"\n{method_label}:")
            print(f"  平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            print(f"  长度比例: {np.mean(len_ratios):.2f} ± {np.std(len_ratios):.2f}")
            print(f"  Token重叠: {np.mean(token_overlaps):.2f} ± {np.std(token_overlaps):.2f}")
            if METRICS_AVAILABLE:
                print(f"  BLEU-1: {np.mean(bleu_1_scores):.4f} ± {np.std(bleu_1_scores):.4f}")
                print(f"  BLEU-2: {np.mean(bleu_2_scores):.4f} ± {np.std(bleu_2_scores):.4f}")
                print(f"  BLEU-3: {np.mean(bleu_3_scores):.4f} ± {np.std(bleu_3_scores):.4f}")
                print(f"  BLEU-4: {np.mean(bleu_4_scores):.4f} ± {np.std(bleu_4_scores):.4f}")
                print(f"  ROUGE-1: {np.mean(rouge_1_f_scores):.4f} ± {np.std(rouge_1_f_scores):.4f}")
                print(f"  ROUGE-2: {np.mean(rouge_2_f_scores):.4f} ± {np.std(rouge_2_f_scores):.4f}")
                print(f"  ROUGE-L: {np.mean(rouge_l_f_scores):.4f} ± {np.std(rouge_l_f_scores):.4f}")
                print(f"  BERT_Score-F1: {np.mean(bert_f_scores):.4f} ± {np.std(bert_f_scores):.4f}")
            print(f"  空回复数: {empty_count}/{len(samples)}")
    
    # 7. 打印几个具体示例
    print("\n" + "="*80)
    print("示例对比 (使用Greedy配置)")
    print("="*80)
    
    greedy_results = all_results[0]  # 第一个配置是Greedy
    for i in range(min(3, len(greedy_results['samples']))):
        sample = greedy_results['samples'][i]
        print(f"\n样本 {i+1}:")
        print(f"问题: {sample['question'][:100]}...")
        print(f"\n标准答案: {sample['ground_truth'][:150]}...")
        print(f"\n方法1 (Slots+Q): {sample['method_1']['text'][:150]}...")
        # print(f"方法2 (History+Q): {sample['method_2']['text'][:150]}...")
        # print(f"方法2 (History+Q 1024): {sample['method_2_1024']['text'][:150]}...")
        # print(f"方法2 (History+Q 512): {sample['method_2_512']['text'][:150]}...")
        # print(f"方法2 (History+Q 256): {sample['method_2_256']['text'][:150]}...")
        # print(f"方法3 (Q Only): {sample['method_3']['text'][:150]}...")
        # print(f"方法4 (Summary+Q): {sample['method_4']['text'][:150]}...")
        print("-"*80)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()

