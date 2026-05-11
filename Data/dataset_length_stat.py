import json
from tqdm import tqdm



def calculate_average_lengths(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_history_len = 0
    total_current_q_len = 0
    total_combined_len = 0
    count = len(data)

    for item in tqdm(data):
        # 1. 计算 History 长度
        history_text_len = sum([len(turn) for turn in item['history']])
        
        # 2. 计算 Current_q 长度
        current_q_len = len(item['current_q'])
        
        # 3. 计算总和
        combined_len = history_text_len + current_q_len

        # 累加
        total_history_len += history_text_len
        total_current_q_len += current_q_len
        total_combined_len += combined_len

    avg_hist = total_history_len / count
    avg_curr = total_current_q_len / count
    avg_comb = total_combined_len / count

    print("-" * 30)
    print(f"数据总条数: {count}")
    print("-" * 30)
    print(f"平均 History 长度 (字符数): {avg_hist:.2f}")
    print(f"平均 Current_q 长度 (字符数): {avg_curr:.2f}")
    print(f"平均 History + Current_q 长度: {avg_comb:.2f}")
    print("-" * 30)




if __name__ == "__main__":
    file_path = '../Abstract_and_Link_A100_86modify/Data/medical_data_v3_summary_text.json'
    calculate_average_lengths(file_path)