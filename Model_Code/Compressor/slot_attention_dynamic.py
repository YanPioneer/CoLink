import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### 动态槽数量
class SlotAttention(nn.Module):
    def __init__(self, input_dim, slot_dim, num_slots, num_iterations=3):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots  # 最大槽数K_max
        self.num_iterations = num_iterations
        
        # 槽位初始化
        self.slot_embeddings = nn.Parameter(torch.randn(1, num_slots, self.slot_dim)).to(torch.bfloat16).to(device)
        
        self.q_proj = nn.Linear(self.slot_dim, self.slot_dim, bias=False)  # 槽 -> Q
        self.k_proj = nn.Linear(input_dim, self.slot_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.slot_dim, bias=False)

        # GRU更新机制
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # 归一化层
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        # 槽位有效性预测
        self.slot_validity = nn.Linear(slot_dim, 1)

    def forward(self, inputs, mask=None, training=True):
        batch_size, seq_len, _ = inputs.shape
        
        # 初始化槽位
        slots = self.slot_embeddings.expand(batch_size, -1, -1)  # [batch_size, num_slots, slot_dim]

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_slots, -1)  # mask: [batch_size, seq_len] -> [batch_size, num_slot, seq_len]

        # 迭代更新槽位
        for iter_idx in range(self.num_iterations):
            # 1.归一化
            slots_norm = self.norm_slots(slots)
            inputs_norm = self.norm_inputs(inputs)
            
            # 2.计算注意力
            q = self.q_proj(slots_norm)  # [batch_size, num_slot, slot_dim]
            k = self.k_proj(inputs_norm)  # [batch_size, seq_len, slot_dim]
            v = self.v_proj(inputs_norm)  # [batch_size, seq_len, slot_dim]

            # print('test q k v value')
            if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
                print(f"Warning: q, k, v contains NaN at iteration {iter_idx}!")
                print(f"Debug - q stats: min={q.min()}, max={q.max()}, mean={q.mean()}")
                print(f"Debug - k stats: min={k.min()}, max={k.max()}, mean={k.mean()}")
                print(f"Debug - v stats: min={v.min()}, max={v.max()}, mean={v.mean()}")
                break
            
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.slot_dim ** 0.5)  # [batch_size, num_slot, seq_len]
            
            if mask is not None:
                fill_value = torch.finfo(attention_scores.dtype).min
                attention_scores = attention_scores.masked_fill(mask == 0, fill_value)
            
            attention_weights = F.softmax(attention_scores, dim=-1)  # attention_weights: [batch_size, num_slot, seq_len]
            
            # 计算本轮slots
            updates = torch.matmul(attention_weights, v)  # [batch_size, num_slot, slot_dim]
            
            # GRU更新槽位
            slots_flat = slots.reshape(-1, self.slot_dim)  # 上一轮slots
            updates_flat = updates.reshape(-1, self.slot_dim)  # 本轮slots
            slots_flat = self.gru(updates_flat, slots_flat)
            slots = slots_flat.view(batch_size, self.num_slots, self.slot_dim)  # slots: [batch_size, num_slot, slot_dim]
        
        ### 预测槽位有效性
        validity_scores = self.slot_validity(slots).squeeze(-1)  # [batch_size, num_slots]

        if training:
            ### 为每个槽位独立做二元决策，得到slot_mask（动态槽）
            logits = torch.stack([torch.zeros_like(validity_scores), validity_scores], dim=-1)  # [batch_size, num_slots, 2]
            gumbel_output = F.gumbel_softmax(logits, hard=True, dim=-1)
            slot_mask = gumbel_output[..., 1].to(torch.bfloat16)  # 取出决策结果，[batch_size, num_slots]
        else:
            # 推理时直接用阈值 | 数学上等价于 sigmoid > 0.5
            slot_mask = (validity_scores > 0).to(torch.bfloat16)

        # slots = slots * slot_mask.unsqueeze(-1)  #❌多余

        return slots, attention_weights, slot_mask


class SlotDecoder(nn.Module):
    """
    Transformer Decoder that cross-attends to slot memory
    将slot解码为摘要文本的token序列
    """
    def __init__(self, dim, vocab_size, num_layers=2, num_heads=4, max_pos=512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_pos = max_pos
        
        self.embedding = nn.Embedding(vocab_size, dim)  # 可学习Embedding
        self.use_llm_embedding = False
        
        self.pos_emb = nn.Parameter(torch.randn(1, max_pos, dim))  # 可学习位置编码
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim, 
                nhead=num_heads, 
                batch_first=True,
                dtype=torch.bfloat16
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, tgt_ids, memory, tgt_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt_ids: [batch_size, seq_len] - 目标token序列（用于teacher forcing）
            memory: [batch_size, num_slots, dim] - slot记忆
            tgt_mask: [seq_len, seq_len] - 目标序列的causal mask
            memory_key_padding_mask: [batch_size, num_slots] - slot的有效性mask
        Returns:
            logits: [batch_size, seq_len, vocab_size] - 输出logits
        """
        batch_size, seq_len = tgt_ids.shape
        
        # 词嵌入 + 位置编码
        tgt_emb = self.embedding(tgt_ids) + self.pos_emb[:, :seq_len, :]  # [batch_size, seq_len, dim]
        
        # Transformer-Decoder层
        out = tgt_emb
        for layer in self.layers:
            out = layer(
                out,  # summary_ids，解码器输入，作为Cross-Attn部分的Q，[batch_size, seq_len, dim]
                memory,  # slots，编码器输出，作为Cross-Attn部分的K/V，[batch_size, num_slots, dim]
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        # 输出投影
        logits = self.lm_head(out)  # [batch_size, seq_len, vocab_size]
        return logits
    
    def forward_with_embeddings(self, tgt_embeddings, memory, tgt_mask=None, memory_key_padding_mask=None):
        """
        直接使用embeddings作为输入的forward方法
        Args:
            tgt_embeddings: [batch_size, seq_len, dim] - 目标embeddings
            memory: [batch_size, num_slots, dim] - slot记忆
            tgt_mask: [seq_len, seq_len] - 目标序列的causal mask
            memory_key_padding_mask: [batch_size, num_slots] - slot的有效性mask
        Returns:
            logits: [batch_size, seq_len, vocab_size] - 输出logits
        """
        batch_size, seq_len, _ = tgt_embeddings.shape
        
        # tgt_embeddings已经包含了LLM的位置编码，不需要再添加
        tgt_emb = tgt_embeddings
        
        # Transformer Decoder层
        out = tgt_emb
        for layer in self.layers:
            out = layer(
                out, 
                memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask 
            )
        
        # 输出投影
        logits = self.lm_head(out)
        return logits


class ReconstructionLoss(nn.Module):
    """
    重构损失：将slot解码为摘要文本
    """
    def __init__(self, slot_dim, vocab_size, num_layers=2, num_heads=4, max_pos=512):
        super().__init__()
        self.slot_decoder = SlotDecoder(
            dim=slot_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_pos=max_pos
        )
    
    def forward(self, compressed, slot_mask, summary_ids, summary_attention_mask, pad_token_id=151643, llm_model=None, tokenizer=None, return_text=False):
        """
        Args:
            compressed: [batch_size, num_slots, slot_dim] - slot表示
            summary_ids: [batch_size, seq_len] - 历史摘要序列
            pad_token_id: padding token的ID
        Returns:
            reconstruction_loss: 重构损失
        """
        batch_size, seq_len = summary_ids.shape
        
        # 准备teacher forcing的输入（右移一位）
        tgt_input = self._shift_right(summary_ids, pad_token_id)
        new_summary_attention_mask = summary_attention_mask
        # new_summary_attention_mask = self._shift_right(summary_attention_mask, 1)  ###
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bfloat16).to(device)
        
        ###❗如果提供了LLM模型，使用LLM编码tgt_input
        # llm_model = None
        if llm_model is not None:
            # 使用LLM的embedding和位置编码
            with torch.no_grad():
                # 直接使用LLM的forward方法获取完整的embeddings（包含位置编码）
                # 创建attention_mask
                # attention_mask = torch.ones_like(tgt_input)
                # 使用LLM的embedding方法获取完整编码
                outputs = llm_model(
                    input_ids=tgt_input,
                    attention_mask=new_summary_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                # tgt_embeddings = outputs.hidden_states[0]
                tgt_embeddings = outputs.hidden_states[-1] 
            
            # 使用slot decoder解码（直接传入embeddings而不是token ids）
            logits = self.slot_decoder.forward_with_embeddings(
                tgt_embeddings, 
                memory=compressed,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=(slot_mask == 0)
            )
        else:
            print('tgt use initial embedding')
            # 使用slot decoder解码（原始方法）
            logits = self.slot_decoder(
                tgt_input, 
                memory=compressed,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=(slot_mask == 0)
            )
        
        
        # 计算重构损失（slots->历史摘要）
        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (N,C): [batch_size * seq_len, vocab_size]
            summary_ids.view(-1),  # [batch_size * seq_len]
            ignore_index=pad_token_id
        )

        if return_text and tokenizer is not None:
            # 生成重构的文本
            with torch.no_grad():
                # 获取预测的token ids
                predicted_ids = torch.argmax(logits, dim=-1)
                # 解码为文本
                reconstructed_texts = []
                for i in range(batch_size):
                    # 去除padding token
                    tokens = predicted_ids[i]
                    non_pad_mask = (tokens != pad_token_id)
                    clean_tokens = tokens[non_pad_mask]
                    
                    # 添加调试信息
                    if i == 0:  # 只打印第一个样本的调试信息
                        print(f"Debug - predicted_ids shape: {predicted_ids.shape}")
                        print(f"Debug - clean_tokens shape: {clean_tokens.shape}")
                        print(f"Debug - clean_tokens sample: {clean_tokens[:10]}")
                        print(f"Debug - tokenizer vocab size: {tokenizer.vocab_size}")
                        print(f"Debug - tokens range: min={tokens.min()}, max={tokens.max()}")
                    
                    # 确保tokens在有效范围内
                    clean_tokens = torch.clamp(clean_tokens, 0, tokenizer.vocab_size - 1)
                    
                    # 解码为文本
                    text = tokenizer.decode(clean_tokens, skip_special_tokens=True)
                    reconstructed_texts.append(text)
            return recon_loss, reconstructed_texts
        else:
            return recon_loss
    
    def _shift_right(self, input_ids, pad_token_id):
        """将输入序列右移一位，用于teacher forcing"""
        shifted = torch.zeros_like(input_ids)
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[:, 0] = pad_token_id
        return shifted


class VariationalSlotAttention(nn.Module):
    """变分Slot Attention，输出分布参数和采样值"""
    
    def __init__(self, input_dim, slot_dim, num_slots, num_iterations=3):
        super().__init__()
        self.slot_attention = SlotAttention(input_dim, slot_dim, num_slots, num_iterations)

        self.mu_head = nn.Linear(slot_dim, slot_dim)
        self.logvar_head = nn.Linear(slot_dim, slot_dim)
        
    def forward(self, x, mask, training=True):
        # 基础slot attention
        slots, _, slot_mask = self.slot_attention(x, mask, training)  # [batch_size, num_slots, slot_dim]
        
        # 概率分布参数（与slot相关）
        mu = self.mu_head(slots) * slot_mask.unsqueeze(-1)  # [batch_size, num_slots, slot_dim]
        logvar = self.logvar_head(slots) * slot_mask.unsqueeze(-1)  # [batch_size, num_slots, slot_dim]
        
        ### 【z的计算是多余的？？只为说明是变分结构】
        if training:
            # 训练时：重参数化
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            # 推理时：使用均值
            z = mu
        
        return slots, z, mu, logvar, slot_mask


class InformationBottleneckLoss(nn.Module):
    """信息瓶颈损失：重构loss + KL散度 + 预测loss"""
    
    def __init__(self, slot_dim, vocab_size, beta=1.0, decoder_layers=2, decoder_heads=4,
                 recon_weight=1.0, kl_weight=1.0, pred_weight=1.0, sparse_weight=1.0, max_pos=512):
        super().__init__()
        self.beta = beta
        self.recon_weight = recon_weight  # 重构损失权重
        self.kl_weight = kl_weight        # KL损失权重
        self.pred_weight = pred_weight    # 预测损失权重
        self.sparse_weight = sparse_weight  # 稀疏性损失权重
        
        # 重构损失：slots -> 历史摘要
        self.recon_loss = ReconstructionLoss(slot_dim, vocab_size, decoder_layers, decoder_heads, max_pos)
        
    def forward(self, slots, slot_mask, z, mu, logvar, summary_ids, summary_attention_mask, q_embeddings, q_mask, q_ids, y_ids, qy_embeddings, qy_attention_mask, q_lengths, frozen_llm, tokenizer, pad_token_id=151643, return_text=False):
        """
        Args:
            slots: [batch_size, num_slots, slot_dim] - 确定性slot表示
            z: [batch_size, num_slots, slot_dim] - 变分采样值
            mu, logvar: [batch_size, num_slots, slot_dim] - 分布参数
            summary_ids: [batch_size, summary_len] - 历史摘要
            q_embeddings: [batch_size, q_len, slot_dim] - 当前问题特征
            y_ids: [batch_size, y_len] - 标准回复
            frozen_llm: 冻结的LLM模型
        """
        
        # 1. 重构loss：slots -> 历史摘要
        if return_text:
            recon_loss, reconstructed_texts = self.recon_loss(slots, slot_mask, summary_ids, summary_attention_mask, pad_token_id, frozen_llm, tokenizer, return_text=True)
        else:
            recon_loss = self.recon_loss(slots, slot_mask, summary_ids, summary_attention_mask, pad_token_id, frozen_llm)
        
        ### 2. KL散度：DKL(q(z|x) || p(z))
        kl_per_slot = 0.5 * (mu**2 + logvar.exp() - logvar - 1) * slot_mask.unsqueeze(-1)  # 屏蔽无效槽位
        kl_per_sample = kl_per_slot.sum(dim=[1, 2])
        num_valid_slots = slot_mask.sum(dim=1).clamp(min=1) # 确定有效槽数（并避免除以零）
        normalized_kl_per_sample = kl_per_sample / num_valid_slots
        kl_loss = normalized_kl_per_sample.mean()
        
        # 3. 预测loss：slots + q -> 标准回复
        # pred_loss = self.prediction_loss(slots, slot_mask, q_embeddings, q_mask, y_ids, frozen_llm, pad_token_id)
        # pred_loss = self.prediction_loss_with_qembed_and_yid(slots, slot_mask, q_ids, y_ids, frozen_llm, pad_token_id, q_embeddings, q_mask, qy_embeddings, qy_attention_mask, q_lengths)
        pred_loss = self.prediction_loss_with_qyembed_and_yid(slots, slot_mask, q_ids, y_ids, frozen_llm, pad_token_id, q_embeddings, q_mask, qy_embeddings, qy_attention_mask, q_lengths)
        
        # 4. 稀疏性损失（鼓励更少有效槽位）❌容易所有槽位全被掩码
        n_valid_slots = slot_mask.sum(dim=1).mean()
        sparse_loss = self.sparse_weight * n_valid_slots

        recon_loss_weighted = self.recon_weight * recon_loss
        kl_loss_weighted = self.kl_weight * kl_loss
        pred_loss_weighted = self.pred_weight * pred_loss

        # 总损失：LIB = I(X;Z|Q) - βI(Y;Z|Q)
        # kl_loss 对应 I(X;Z|Q)，需要最小化
        # pred_loss 对应 -I(Y;Z|Q)，需要最小化
        # 所以使用加号
        # total_loss = recon_loss_weighted + kl_loss_weighted + pred_loss_weighted + sparse_loss
        # total_loss = recon_loss_weighted + kl_loss_weighted + pred_loss_weighted
        total_loss = recon_loss_weighted + kl_loss_weighted + pred_loss_weighted - sparse_loss
        
        result = {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'prediction_loss': pred_loss,
            'sparse_loss': sparse_loss,
            'slots': slots,
            'z': z
        }
        
        if return_text:
            result['reconstructed_texts'] = reconstructed_texts
            
        return result
    
    def prediction_loss(self, slots, slot_mask, q_embeddings, q_mask, y_ids, frozen_llm, pad_token_id):
        """slots + q -> 标准回复"""
        batch_size = slots.shape[0]
        
        valid_slots_padded = slots * slot_mask.unsqueeze(-1)  # 仅保留有效槽位

        y_embeddings = frozen_llm.model.embed_tokens(y_ids)
        inputs_embeds = torch.cat([valid_slots_padded, q_embeddings, y_embeddings], dim=1)

        y_mask = (y_ids != pad_token_id).long()
        attention_mask = torch.cat([slot_mask, q_mask, y_mask], dim=1)  # [batch_size, num_slots + 2 * seq_len]

        context_length = slots.shape[1] + q_embeddings.shape[1]
        context_labels = torch.full((batch_size, context_length), -100, dtype=torch.long, device=device)
        
        labels_for_y = y_ids.clone()
        labels_for_y[y_mask == 0] = -100
        
        labels = torch.cat([context_labels, labels_for_y], dim=1)

        outputs = frozen_llm(
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            attention_mask=attention_mask
        )

        return outputs.loss
        
    def prediction_loss_with_qyid_only(self, slots, slot_mask, q_ids, y_ids, frozen_llm, pad_token_id, q_embeddings, q_mask, qy_embeddings, qy_attention_mask, q_lengths):
        """slots + q -> 标准回复"""
        batch_size, slots_len = slots.shape[:2]

        # 首先检查slots是否正常
        if torch.isnan(slots).any() or torch.isinf(slots).any():
            print("Warning: slots contains NaN or Inf, returning zero loss")
            print(f"Debug - slots stats: min={slots.min()}, max={slots.max()}, mean={slots.mean()}")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)
        
        # 添加slots的统计信息
        print(f"Debug - slots stats: min={slots.min()}, max={slots.max()}, mean={slots.mean()}, std={slots.std()}")
        
        # 直接拼接q_ids和y_ids（因为数据集中已经padding=False）
        qy_ids_list = []
        q_lengths = []
        for i in range(batch_size):
            q_len = len(q_ids[i])
            qy_combined = torch.cat([q_ids[i], y_ids[i]])
            qy_ids_list.append(qy_combined)
            q_lengths.append(q_len)
        
        # 找到最大长度
        max_qy_len = max(len(qy) for qy in qy_ids_list)
        # print(f"Debug prediction_loss - max_qy_len: {max_qy_len}")
        
        # padding到统一长度
        qy_ids_padded = torch.full((batch_size, max_qy_len), pad_token_id, 
                                  dtype=torch.long, device=slots.device)
        for i, qy in enumerate(qy_ids_list):
            qy_len = len(qy)
            qy_ids_padded[i, :qy_len] = qy
        
        # 获取q+y的embeddings（frozen_llm不需要梯度）
        with torch.no_grad():
            qy_embeddings = frozen_llm.get_input_embeddings()(qy_ids_padded)
        
        # 拼接：slots + qy（slots需要梯度）
        full_inputs_embeds = torch.cat([slots, qy_embeddings.detach()], dim=1)
        # print(f"Debug prediction_loss - full_inputs_embeds shape: {full_inputs_embeds.shape}")
        
        # 检查长度是否超过LLM最大位置编码
        max_pos = getattr(frozen_llm.config, 'max_position_embeddings', 32768)
        if full_inputs_embeds.size(1) > max_pos:
            print(f"Warning: Sequence length {full_inputs_embeds.size(1)} exceeds max_position_embeddings {max_pos}")
            # 截断到最大长度
            full_inputs_embeds = full_inputs_embeds[:, :max_pos]
            labels = labels[:, :max_pos]
            print(f"Truncated to shape: {full_inputs_embeds.shape}")
        
        # 创建标签：只对y部分计算损失
        total_len = slots_len + max_qy_len
        labels = torch.full((batch_size, total_len), -100, 
                          dtype=torch.long, device=slots.device)
        
        # 只对y部分设置标签，从slots_len + q_len开始
        for i in range(batch_size):
            q_len = q_lengths[i]
            y_start = slots_len + q_len  # 从slots_num + len(q_ids[i])开始
            y_end = slots_len + len(qy_ids_list[i])
            labels[i, y_start:y_end] = qy_ids_list[i][q_len:]
        
        # 检查输入是否有NaN
        if torch.isnan(full_inputs_embeds).any():
            print("Warning: NaN detected in full_inputs_embeds!")
            print(f"Debug - slots has NaN: {torch.isnan(slots).any()}")
            print(f"Debug - qy_embeddings has NaN: {torch.isnan(qy_embeddings).any()}")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)
        
        # 直接使用slots，它本身就有梯度
        frozen_llm.eval()
        
        # 计算logits，slots的梯度会自动传播
        outputs = frozen_llm(inputs_embeds=full_inputs_embeds, labels=labels)
        
        # frozen_llm.train()
        
        # 检查loss是否有NaN
        if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
            print(f"Warning: NaN/Inf detected in prediction loss: {outputs.loss}")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)
        
        # print(f"Debug prediction_loss - loss: {outputs.loss}")
        return outputs.loss
    
    def prediction_loss_with_qembed_and_yid(self, slots, slot_mask, q_ids, y_ids, frozen_llm, pad_token_id, q_embeddings, q_mask, qy_embeddings, qy_attention_mask, q_lengths):
        """slots + q -> 标准回复   q[pad]y[pad]"""
        batch_size, num_slots, _ = slots.shape
        
        # slot_mask = torch.ones(batch_size, num_slots, dtype=torch.long, device=slots.device)

        y_embeddings = frozen_llm.model.embed_tokens(y_ids)
        inputs_embeds = torch.cat([slots, q_embeddings, y_embeddings], dim=1)

        y_mask = (y_ids != pad_token_id).long()
        attention_mask = torch.cat([slot_mask, q_mask, y_mask], dim=1)  # [batch_size, num_slots + 2 * seq_len]

        context_length = slots.shape[1] + q_embeddings.shape[1]
        context_labels = torch.full((batch_size, context_length), -100, dtype=torch.long, device=device)
        
        labels_for_y = y_ids.clone()
        labels_for_y[y_mask == 0] = -100
        
        labels = torch.cat([context_labels, labels_for_y], dim=1)

        outputs = frozen_llm(
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            attention_mask=attention_mask
        )

        return outputs.loss
    
    def prediction_loss_with_qyembed_and_yid(self, slots, slot_mask, q_ids, y_ids, frozen_llm, pad_token_id, q_embeddings, q_mask, qy_embeddings, qy_attention_mask, q_lengths):
        """slots + q -> 标准回复   qy[pad]
        Args:
            slots: [batch_size, num_slots, slot_dim] - 压缩后的slots
            qy_embeddings: [batch_size, max_qy_len, hidden_dim] - 预计算的qy embeddings
            qy_attention_mask: [batch_size, max_qy_len] - qy的attention mask
            y_ids: [batch_size, max_y_len] - y的token ids（用于创建labels）
            q_lengths: list of int - q部分的长度（用于创建labels）
            frozen_llm: 冻结的LLM模型
            pad_token_id: padding token id
        
        """
        batch_size, num_slots, _ = slots.shape

        # 首先检查slots是否正常
        if torch.isnan(slots).any() or torch.isinf(slots).any():
            print("Warning: slots contains NaN or Inf, returning zero loss")
            print(f"Debug - slots stats: min={slots.min()}, max={slots.max()}, mean={slots.mean()}")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)
        
        # 检查qy_embeddings是否正常
        if torch.isnan(qy_embeddings).any() or torch.isinf(qy_embeddings).any():
            print("Warning: qy_embeddings contains NaN or Inf, returning zero loss")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)

        # 序列结构：[slots] [q部分] [y部分] [padding]
        full_inputs_embeds = torch.cat([
            slots,  # slots作为前缀
            qy_embeddings.detach()  # q+y的embeddings
        ], dim=1)

        # 创建attention mask
        # slots部分：全1（总是attend）
        # slot_mask = torch.ones(batch_size, num_slots, dtype=torch.long, device=slots.device)
        # 拼接qy的mask
        attention_mask = torch.cat([slot_mask, qy_attention_mask], dim=1)

        # 检查长度是否超过LLM最大位置编码
        max_pos = getattr(frozen_llm.config, 'max_position_embeddings', 32768)
        if full_inputs_embeds.size(1) > max_pos:
            print(f"Warning: Sequence length {full_inputs_embeds.size(1)} exceeds max_position_embeddings {max_pos}")
            # 截断到最大长度
            full_inputs_embeds = full_inputs_embeds[:, :max_pos]
            attention_mask = attention_mask[:, :max_pos]

        # 创建标签：只对y（assistant回复）部分计算损失
        # 序列结构：[slots] [q部分] [y部分] [padding]
        total_len = full_inputs_embeds.size(1)
        labels = torch.full((batch_size, total_len), -100, 
                          dtype=torch.long, device=slots.device)
        
        # 只对y部分设置标签
        # y部分从 num_slots + q_length 开始
        labels_for_y = y_ids.clone()
        for i in range(batch_size):
            q_len = q_lengths[i]
            y_len = (labels_for_y[i] != pad_token_id).sum().item()
            # y部分的起始位置
            y_start = num_slots + q_len
            y_end = num_slots + q_len + y_len
            # 设置y部分的标签
            labels[i, y_start:y_end] = labels_for_y[i, :y_len]
        
        # 检查输入是否有NaN
        if torch.isnan(full_inputs_embeds).any():
            print("Warning: NaN detected in full_inputs_embeds!")
            print(f"Debug - slots has NaN: {torch.isnan(slots).any()}")
            print(f"Debug - qy_embeddings has NaN: {torch.isnan(qy_embeddings).any()}")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)

        frozen_llm.eval()
        
        # 计算logits，slots的梯度会自动传播
        outputs = frozen_llm(
            inputs_embeds=full_inputs_embeds, 
            attention_mask=attention_mask,  # 显式传入attention mask
            labels=labels
        )

        # 检查loss是否有NaN
        if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
            print(f"Warning: NaN/Inf detected in prediction loss: {outputs.loss}")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)

        return outputs.loss



class DialogueCompressionModel(nn.Module):
    """完整的对话压缩模型"""
    
    def __init__(self, input_dim, slot_dim, num_slots, vocab_size, 
                 num_iterations=3, beta=1.0, decoder_layers=2, decoder_heads=4,
                 recon_weight=1.0, kl_weight=1.0, pred_weight=1.0, sparse_weight=1.0, max_pos=512):
        super().__init__()
        
        # 变分slot_attention
        self.variational_slot_attention = VariationalSlotAttention(
            input_dim, slot_dim, num_slots, num_iterations
        )
        
        # 信息瓶颈损失
        self.ib_loss = InformationBottleneckLoss(
            slot_dim, vocab_size, beta, decoder_layers, decoder_heads,
            recon_weight, kl_weight, pred_weight, sparse_weight, max_pos,
        )
        
    def forward(self, dialogue_history, history_mask, summary_ids, summary_attention_mask, q_embeddings, q_mask, q_ids, y_ids, 
                qy_embeddings, qy_attention_mask, q_lengths, frozen_llm, tokenizer, pad_token_id=151643, training=True, return_text=False):
        """
        Args:
            dialogue_history: [batch_size, seq_len, input_dim] - 历史对话特征
            history_mask: [batch_size, seq_len] - 历史对话mask
            summary_ids: [batch_size, summary_len] - 历史摘要
            q_embeddings: [batch_size, q_len, slot_dim] - 当前问题特征
            y_ids: [batch_size, y_len] - 标准回复
            frozen_llm: 冻结的LLM模型
        """
        # 压缩历史对话
        slots, z, mu, logvar, slot_mask = self.variational_slot_attention(dialogue_history, history_mask, training)

        # 计算信息瓶颈损失
        losses = self.ib_loss(
            slots=slots,
            slot_mask=slot_mask,
            z=z,
            mu=mu,
            logvar=logvar,
            summary_ids=summary_ids,
            summary_attention_mask=summary_attention_mask,
            q_embeddings=q_embeddings,
            q_mask=q_mask,
            q_ids=q_ids,
            y_ids=y_ids,
            qy_embeddings=qy_embeddings,
            qy_attention_mask=qy_attention_mask,
            q_lengths=q_lengths,
            frozen_llm=frozen_llm,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            return_text=return_text
        )
        
        return losses
    
    def compress(self, dialogue_history, history_mask, training=False):
        """仅进行压缩，不计算损失
        Args:
            dialogue_history: [batch_size, seq_len, input_dim] - 历史对话特征
            history_mask: [batch_size, seq_len] - 历史对话mask
        """
        slots, _, _, _, slot_mask = self.variational_slot_attention(dialogue_history, history_mask, training)

        # 过滤无效槽位
        valid_slots_padded = slots * slot_mask.unsqueeze(-1)
        return valid_slots_padded, slot_mask  # valid_slots_padded: [batch_size, num_slots, slot_dim]
        
