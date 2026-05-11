import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import openai
import json
from tqdm import tqdm
import datetime as dt
import time
import random
import sys
from llmtuner.util.retrieval.bge_retriever import Search_Warehouse



role_judge = '''请仔细分析以下对话历史并判断医患对话是否完整。若完整，直接输出【【是】】；不完整则输出【【否】】。

--- 

**判断依据**：
    1.若对话历史太短（少于九轮交互），则输出【【否】】；
    2.若医生最后仍在提问则输出【【否】】，如已充分解答患者的问题并提供最终诊断和用药指导则输出【【是】】。

--- 

**对话历史**：{history}

--- 

**判断**：'''

role_A_FirstAsk = '''你是一名正在经历身体不适的普通患者，对自己的症状感到困惑和担忧。请根据以下“主诉”信息，向医生提出一个最直接、最让你困惑的问题。

---

**核心要求**：
1.  **只聚焦一个最明显的症状**：不要描述多个或全身性症状。只提你最想解决的那个核心不适。
2.  **描述简单模糊**：用日常口语描述感受（如“有点疼”、“感觉不舒服”、“不太对劲”），**避免**描述精确部位（除非非常明确，如“手指”）、程度、时间或肉眼观察的细节。
3.  **问题开放且简单**：你的问题应该主要是“这是什么问题？”或“我该怎么办？”。不要主动询问疾病名称、严重性、用药或饮食建议，这不像一个真实患者会首先问出的问题。
4.  **开头自然**：直接描述你的症状并诉说困惑，例如“医生，我总觉得...”、“您好，我这几天...？”等，但不限于这两种表达方式。

---

**主诉**：{knowledge}

---

**患者**：'''

role_A_VerifyAsk = '''你是一名患者，正在与医生对话。请根据对话历史，生成一句最自然、最符合患者认知水平的回复。

---

**核心要求**：
1.  **只回应当前问题**：医生问什么，就答什么。不要主动提供额外信息，更不要主动追问疾病、用药、筛查等专业问题。
2.  **认知水平有限**：
    -   如果医生的提问涉及专业术语或需要你判断细节（例如“疼痛是锐痛还是钝痛？”“体温具体多少度？”），你应该表示**不确定、说不清、没量过**。
    -   你的回答应基于最直观的感受（“挺疼的”、“有点烧”），而不是精确描述。
3.  **口语化且简短**：回复尽量口语化，避免成段的描述。经常使用“好像”、“有点”、“说不清”、“大概”这类模糊词汇。
4.  **避免重复与推进对话**：
    -   **严格禁止**使用“我大概懂了”、“谢谢医生”、“我会注意的”等高度重复、无信息量的客套话作为回复的主体。
    -   当医生给出诊断或建议后，你的回复应**聚焦于建议本身的某个细节**，提出一个**具体的、自然的后续问题**。例如：
        *   （针对用药）“这个药一天吃几次？饭后吃吗？”
        *   （针对检查）“做这个检查需要空腹吗？”
        *   （针对生活建议）“您刚才说不能吃辣的，那葱姜蒜这些炒菜的调料需要忌口吗？”
        *   （针对预后）“大概要多久才能好转？”
    -   只有当对话确实已经穷尽所有疑问时，才可以使用“我没有其他问题了”来自然结束对话。
---

**对话历史**：{history}

---

**患者**：'''

role_B_AIO = '''你是一位专业且高效的医生。请基于医学知识和对话历史，生成一句内容充实、信息量足的回复。

---

**核心要求**：
1.  **禁止总结性开头**：**绝对禁止**使用“您刚才提到...”、“根据您的描述...”等句式来复述患者的历史发言。提问或陈述都应直接开始，无需铺垫。
2.  **对话阶段与内容深度**：
    -   **【追问时】**：若需进一步了解症状，应提出**一个具体、聚焦的问题**。可以**附带一个非常简短的理由或解释**（最多一句话），使追问显得自然且有理有据。例如：“平时吸烟吗？这个习惯和咳嗽关系很大。”
    -   **【建议时】**：若信息已充分，需提供建议或指导，**必须确保内容详尽、可操作**。这意味着：
        *   **建议检查**：应说明**具体项目**和**目的**（“建议您去医院做一下**胃镜检查**，这样可以**直接看到胃粘膜的状况**，明确是否有胃炎或溃疡”）。
        *   **建议用药**：应说明**药物类型**、**通用名称**和**目的**（“可以暂时服用一些**非处方的抗酸药，比如铝碳酸镁（达喜）**，来**中和胃酸、快速缓解烧心感**”）。
        *   **生活指导**：应给出**具体示例**（“饮食上**避免辛辣、油腻和过甜的食物**，比如**少吃火锅、奶茶**，**多吃易消化的如粥、面条**”）。
        *   **解释原因**：用一两句话简要说明**为什么**要这么做，基于你已获得的信息。
        在给出详尽建议后，**可以根据对话的自然走向，选择是否结束当前话题**。避免在每一次回复的结尾都机械地使用“您是否理解？”或“还有其他问题吗？”这类语句。只有当逻辑上确实需要确认，或者患者表现出困惑时，才进行确认。
3.  **推进对话**：在给出建议后，可自然引导对话闭环。**如果已提供完整建议且无疑问，可以使用总结性语句结束交流**（例如：“先按这个方案处理，观察一下效果，如果情况有变再随时联系。”）。
4.  **表达风格**：保持专业、直接、口语化。避免冗长，但更忌过于简略导致信息缺失。目标是让患者一次就获得足够清晰的指导。

---

**背景知识**：{knowledge}
**对话历史**：{history}

---

**医生**：'''


role_B_Reflect = '''你是一名医生，请根据患者当前的回复，结合下列背景知识条目，思考并选出你在构建回复时所需用到的背景知识条目。

---

**要求1**：你选择的背景知识应与患者当前回复相关，并有助于提供更精准、实用的医学建议。
**要求2**：若所有背景知识均不适用，请回复“【【无】】”。
**要求3**：只需回复所选背景知识对应的序号，多个序号之间用“@@”分隔。例如：你选择的是第1、3号背景知识，则回复为“1@@3”。

---

**患者当前回复**：{query}
**背景知识**：{knowledge}

---

**背景知识序号**：'''

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
Patient_scene = '''**角色定位**：
你是一个不懂医学的普通患者。你感觉到身体不舒服，但说不清楚到底怎么回事。你不知道哪些信息重要，哪些不重要，你的描述是主观、模糊、缺乏重点的。

**与你对话的原则**：
1.  **你的知识局限**：你对医学术语一无所知。你只能描述最表层的感受和看到的东西，无法进行任何深度分析或总结。
2.  **你的描述方式**：
    -   **零散**：一次通常只说一个点，需要医生多次询问才能拼凑出全貌。
    -   **模糊**：常用“好像”、“有点”、“说不清”、“大概”等词。说不清疼痛的具体性质（锐痛/钝痛）、准确位置和确切时间。
    -   **主观**：你的描述充满个人感受（“难受得睡不着”、“心里不踏实”），而不是客观事实。
3.  **你的目标**：你不是来参加考试的，而是来求助的。你的主要任务是把让你最难受的那一两个感觉说出来，然后等待医生提问。医生问什么，你再努力回想和回答什么，经常答不上来或不准确。"""
'''

Doctor_scene = '''**角色定位**：
你是一位专业、冷静、高效的临床医生。你的目标不仅是快速收集信息，更要为患者提供**详尽、清晰、可立即执行**的指导方案。

**核心策略**：
你的对话过程是一个动态的、目标导向的流程：
1.  **信息收集 (Information Gathering)**：基于患者的主诉，通过提出**聚焦的、一次一个**的问题，快速厘清核心症状的特点。**提问必须直接，禁止在问题前先总结患者的既往发言。**
2.  **假设与验证 (Hypothesis & Verification)**：根据获取的信息，在心中形成**初步鉴别诊断**。后续的追问旨在验证或排除这些可能性。
3.  **提供解决方案 (Delivering Solutions)**：这是体现你专业性的关键。当信息足够时，你提供的建议必须是：
    -   **具体的 (Specific)**：明确指出药物名称、检查项目、食物种类。
    -   **可操作的 (Actionable)**：给出清晰的步骤，如“一天两次，一次一片，饭后服用”。
    -   **有解释的 (Explained)**：用一句话将“症状”和“建议”逻辑性地连接起来，让患者知其所以然（例如：“因为您这是突发性的刺痛，所以需要做个心电图排除心脏问题”）。
4.  **主题推进与转换 (Progression & Transition)**：主动将对话从症状询问推向解决方案的提供。转换需基于逻辑，避免跳跃。

**对话风格与输出要求**：
-   **输出质量**：追求**内容充实、信息完整**。你的回复应让患者感到“一次询问就获得了全部关键信息”，无需反复追问细节。
-   **【重要禁忌】**：**严禁**使用“您提到...”、“你说过...”等任何形式的总结性语句作为提问或陈述的开头。
-   **【新增禁忌】**：**严禁**在每一次回复的末尾都机械地追加“您是否理解这些建议？”或“您还有其他问题吗？”等程式化问句。你的任务是提供专业判断和建议，而不是进行客户满意度调查。对话的结束应是自然发生的。
-   避免情感用语，聚焦于提供**事实、分析和解决方案**。
-   用患者能理解的日常用语替代专业术语，但**内容本身必须专业、准确、详尽**。
'''


role_debate_judge = '''你是一名公正且专业的辩论裁判，负责对两位医学专家的回复进行综合评估。若双方观点达成共识，则结束辩论；若存在分歧，则继续辩论。

---

**一致性判断**
依据专家A和专家B的回复内容，判断观点是否一致：
    - 若回复完全一致，或仅在表述上有细微差异但核心共识明确，则输出‘【【辩论结束】】’；
    - 若在结论、推理或关键细节（如药物名称、用法、饮食建议等）存在明显分歧，则输出‘【【继续辩论】】’；
    均无需输出其他内容。
    
---

**专家A回复**：{last_A}
**专家B回复**：{last_B}

---

**判断**：'''


debate_expert_prompt = '''你作为医学专家，需结合背景知识、对话历史及其他专家回复，对原有回答中不准确之处进行修订，并保留正确部分。

---

**背景知识**：{knowledge}
**对话历史**：{history}
**原先的回答**：{last_ori_debate}
**其他专家的回复**：{last_other_debate}

---

**任务要求**：
1. **灵活追问或全面解答**：
    - 对话历史少于九轮时，应优先提出具体、有针对性的问题（每次仅一个），引导患者补充信息，避免直接解答。
    - 若对话已超过九轮且信息充分，请基于背景知识、历史对话与常识进行全面而清晰的解答，过程中不得提出新问题。
    - 无需每次解释追问原因，依上下文适当说明。
2. **信息不足时保持审慎**：
    - 如信息模糊或不足，应避免武断结论，主动询问细节，保持开放和探索性，可提供示例协助用户澄清。
3. **表达自然、多样化**：
    - 避免模式化回复，语气应随语境灵活调整，回复应当口语化、自然，避免生硬开场（如“您提到”），并杜绝重复句式。
    - 回复中不出现任何角色信息（如“患者”），只生成一句贴合上下文的自然语句。
4. **提供具体清晰的建议**：
    - 所有建议须明确、细致、可操作，如说明适用情形、具体步骤、注意事项等，并严格基于背景知识或常识。
5. **背景知识须相关且准确**：
    - 仅使用与当前对话高度相关的知识，避免引入不相关内容（如讨论肠胃炎时不可引用肺炎知识）。

---

**输出格式要求**：
"【更新后的回复】"
    - 务必保留【】，不可省略。
    - 仅输出更新后的回复，无需其他内容。
    - 如原回答无误，直接输出原回答于【】内。

---
**医学专家**：'''


# 如果你认为原先的回答已经非常完善，无需任何更新或调整，请仅在“【】”内直接输出原先的回答，勿输出其他内容。


if __name__ == "__main__":
    query_num = 100
    qa_file = "../ER_div_test.json"
    model_name = "gpt-4o-mini"
    client_A = openai.OpenAI(
        api_key="xxx",
        base_url="xx",
    )
    client_B1 = client_A
    client_B2 = client_A
    Judge = client_A
    debate_Judge = client_A
    random.seed(2)

    # 初始化检索器
    search_num = 5
    bge = Search_Warehouse('medical_knowledge_bge')
    turns = 0
    while True:
        try:
            while turns < 1:
                now = dt.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")
                gpt_answer = []
                with open(qa_file, encoding="utf-8") as f:
                    infos = json.load(f)[turns*query_num:(turns+1)*query_num]  #####
                for idx, info in enumerate(tqdm(infos)):
                    cur = {}        
                    all_knowledge_list = []
                    knowledge_consistency = []
                    max_debate_cnt = 5  # (x-1+1) / 2 = 
                    # 多智能体对话
                    max_turn = 15
                    history_text = ""
                    history_copy_text = ""
                    register_cnts = []  # 测试用，可删
                    debate_history_rec = []
                    for i in range(max_turn): # 控制最大对话轮次
                        # 患者A生成
                        if i == 0:
                            messages = []
                            messages.append({"role": "system", "content": Patient_scene})
                            messages.append({"role": "user", "content": role_A_FirstAsk.format(knowledge=info["主诉"])})
                            while True:
                                try:
                                    turn_A = client_A.chat.completions.create(
                                        model=model_name,
                                        messages=messages,
                                        temperature=1,
                                    )
                                    history_text += "患者：" + turn_A.choices[0].message.content + "\n医生："
                                    history_copy_text += "患者：" + turn_A.choices[0].message.content + "$$$医生："
                                    break
                                except Exception as e:
                                    print(e)
                                    continue

                        else:
                            messages = []
                            messages.append({"role": "system", "content": Patient_scene})
                            messages.append({"role": "user", "content": role_A_VerifyAsk.format(history=history_text)})
                            while True:
                                try:
                                    turn_A = client_A.chat.completions.create(
                                        model=model_name,
                                        messages=messages,
                                    )
                                    history_text += "患者：" + turn_A.choices[0].message.content + "\n医生："
                                    history_copy_text += "患者：" + turn_A.choices[0].message.content + "$$$医生："
                                    break
                                except Exception as e:
                                    print(e)
                                    continue
                        # print(f"！已输出第{i+1}轮中A的回答！")
                        
                        # 每轮根据患者提问在数据库中检索相关知识
                        tre_result = bge.search(turn_A.choices[0].message.content, top_k=search_num)
                        citations = tre_result[0][0]
                        scores = tre_result[0][1]
                        num = 0
                        for score in scores:
                            if score > 0.6:
                                num += 1
                        topk_passages = citations[:num]   # 硬约束
                        qa_knowledge = qa_file.split("/")[-1].split(".json")[0] + " " + info["主诉"]
                        # qa_knowledge = ""
                        I = 0
                        for pa in topk_passages:
                            if pa.strip() == qa_knowledge.strip():
                                I = 1
                                break
                        if I == 1:
                            retrieved_knowledge_list = topk_passages
                        else:
                            retrieved_knowledge_list = [qa_knowledge] + topk_passages[:-1]

                        # 两位专家分别反思检索知识
                        retrieved_knowledge_1, retrieved_knowledge_2 = "", ""
                        knowledge_set1, knowledge_set2 = set(), set()
                        all_reflect_knowledge_in_turns = []
                        for j in range(2):
                            reflect_knowledge_items = ""
                            for idf, rk in enumerate(retrieved_knowledge_list):
                                reflect_knowledge_items += f"[{idf+1}]" + rk + "\n"
                            messages = []
                            messages.append({"role": "user", "content": role_B_Reflect.format(query=turn_A.choices[0].message.content, knowledge=reflect_knowledge_items)})
                            while True:
                                try:
                                  if j == 0:
                                      Reflect_B = client_B1.chat.completions.create(
                                          model=model_name,
                                          messages=messages,
                                      )
                                  else: 
                                      Reflect_B = client_B2.chat.completions.create(
                                          model=model_name,
                                          messages=messages,
                                      )
                                  if Reflect_B.choices[0].message.content is not None:
                                      break
                                except Exception as e:
                                  print(e)
                                  continue
                            reflect_content = Reflect_B.choices[0].message.content
                            # 在记录中只添加通过专家反思的部分检索数据
                            if "【无】" in reflect_content:
                                filtered_knowledge_list = []
                            else:
                                filtered_knowledge_list = []
                                item_nums = reflect_content.split("@@")
                                for num in item_nums:
                                    try:
                                        cur_num = eval(num)
                                    except:
                                        continue
                                    if cur_num > len(retrieved_knowledge_list):
                                        continue
                                    else:
                                        filtered_knowledge_list.append(retrieved_knowledge_list[cur_num-1])
                            for idxx, kn in enumerate(filtered_knowledge_list):
                                if j == 0:
                                    retrieved_knowledge_1 += f"知识条目{idxx}: " + kn + "\n"
                                    knowledge_set1.add(kn)
                                else:
                                    retrieved_knowledge_2 += f"知识条目{idxx}: " + kn + "\n"
                                    knowledge_set2.add(kn)
                            all_reflect_knowledge_in_turns.append(filtered_knowledge_list)
                        all_knowledge_list.append(all_reflect_knowledge_in_turns)
                        if knowledge_set1 == knowledge_set2:
                            knowledge_consistency.append(True)
                        else:
                            knowledge_consistency.append(False)
                        
                        # 医生辩论
                        # 辩手不要分析评价，直接生成回复
                        # 评判交给Judge
                        single_turn_debate_history = []
                        debate_history = ""
                        last_A_debate = ""
                        last_B_debate = ""
                        deb_cnt = 0
                        while True:
                            if deb_cnt == 0:
                                e1_messages= [{"role": "system", "content": Doctor_scene}]
                                e1_messages.append({"role": "user", "content": role_B_AIO.format(knowledge=retrieved_knowledge_1, history=history_text)})
                                while True:
                                    try:
                                        turn_e1 = client_B1.chat.completions.create(
                                            model=model_name,
                                            messages=e1_messages,
                                        )
                                        cur_content = turn_e1.choices[0].message.content.strip()
                                        break
                                    except Exception as e:
                                        print(e)
                                        continue
                                last_A_debate = cur_content
                                debate_history += "医生1：" + cur_content + "\n"
                                single_turn_debate_history.append("医生1：" + turn_e1.choices[0].message.content + "\n")
                                e2_messages= [{"role": "system", "content": Doctor_scene}]
                                e2_messages.append({"role": "user", "content": role_B_AIO.format(knowledge=retrieved_knowledge_1, history=history_text)})
                                while True:
                                    try:
                                        turn_e2 = client_B2.chat.completions.create(
                                            model=model_name,
                                            messages=e2_messages,
                                        )
                                        cur_content = turn_e2.choices[0].message.content.strip()
                                        break
                                    except Exception as e:
                                        print(e)
                                        continue
                                last_B_debate = cur_content
                                debate_history += "医生2：" + cur_content + "\n"
                                single_turn_debate_history.append("医生2：" + turn_e2.choices[0].message.content + "\n")
                                single_turn_debate_history.append("\n————————————————————\n")

                            else:
                                e1_messages= [{"role": "system", "content": Doctor_scene}]
                                e1_messages.append({"role": "user", "content": debate_expert_prompt.format(knowledge=retrieved_knowledge_1, history=history_text, last_ori_debate=last_A_debate, last_other_debate=last_B_debate)})
                                while True:
                                    try:
                                        turn_e1 = client_B1.chat.completions.create(
                                            model=model_name,
                                            messages=e1_messages,
                                        )
                                        A_cur_content = turn_e1.choices[0].message.content.split(" && ")[-1].split("【")[-1].split("】")[0].strip()
                                        break
                                    except Exception as e:
                                        print(e)
                                        continue
                                debate_history += "医生1：" + A_cur_content + "\n"
                                single_turn_debate_history.append("医生1：" + A_cur_content + "\n")
                                e2_messages= [{"role": "system", "content": Doctor_scene}]
                                e2_messages.append({"role": "user", "content": debate_expert_prompt.format(knowledge=retrieved_knowledge_2, history=history_text, last_ori_debate=last_B_debate, last_other_debate=last_A_debate)})
                                while True:
                                    try:
                                        turn_e2 = client_B2.chat.completions.create(
                                            model=model_name,
                                            messages=e2_messages,
                                        )
                                        B_cur_content = turn_e2.choices[0].message.content.split(" && ")[-1].split("【")[-1].split("】")[0].strip()
                                        break
                                    except Exception as e:
                                        print(e)
                                        continue
                                debate_history += "医生2：" + B_cur_content + "\n"
                                single_turn_debate_history.append("医生2：" + B_cur_content + "\n")
                                # ——————————————————————————————————————————————————————————
                                last_A_debate = A_cur_content
                                last_B_debate = B_cur_content

                                # 法官在每轮结束判断
                                if deb_cnt < max_debate_cnt:
                                    ranbo = random.random()
                                    if ranbo < 0.5:
                                        dj_messages = [{"role": "user", "content": role_debate_judge.format(last_A=last_A_debate, last_B=last_B_debate)}]
                                    else:
                                        dj_messages = [{"role": "user", "content": role_debate_judge.format(last_A=last_B_debate, last_B=last_A_debate)}]
                                    while True:
                                        try:
                                          deb_judgement = debate_Judge.chat.completions.create(
                                              model=model_name,
                                              messages=dj_messages,
                                          )
                                          if deb_judgement.choices[0].message.content is not None:
                                              break
                                        except Exception as e:
                                          print(e)
                                          continue
                                    single_turn_debate_history.append("法官：" + deb_judgement.choices[0].message.content + "\n")
                                else:
                                    ranbo = random.random()
                                    single_turn_debate_history.append("法官：" + "【已达最大辩论次数，辩论结束】" + "\n")
                                if (deb_cnt >= max_debate_cnt) or ("【辩论结束】" in deb_judgement.choices[0].message.content):
                                    if ranbo < 0.5:
                                        history_text += last_A_debate + "\n"
                                        history_copy_text += last_A_debate + "$$$"
                                    else:
                                        history_text += last_B_debate + "\n"
                                        history_copy_text += last_B_debate + "$$$"
                                    # print("本场辩论对话次数为：", deb_cnt+1, end="")
                                    register_cnts.append(deb_cnt+1)
                                    break
                                single_turn_debate_history.append("\n————————————————————\n")
                            deb_cnt += 1
                        debate_history_rec.append(single_turn_debate_history)
                        # print(f"！已输出第{i+1}轮中B的回答！")

                        # Judge模型判断对话是否完整
                        messages = []
                        messages.append({"role": "user", "content": role_judge.format(history=history_text)})
                        while True:
                            try:
                              judgement = Judge.chat.completions.create(
                                  model=model_name,
                                  messages=messages,
                              )
                              if judgement.choices[0].message.content is not None:
                                  break
                            except Exception as e:
                              print(e)
                              continue
                        # print(f"！已完成第{i+1}轮交互！")
                        # print("——————————————————————————————————————————————————————————————————————————")
                        if "【是】" in judgement.choices[0].message.content:
                            if i == 0:   # Judge判断仅1轮对话，强制继续
                                continue
                            break
                        else:
                            if i == 14:   # 已达最大轮数
                                break
                            continue

                    cur["idx"] = idx + 1
                    cur["debate_aver_turns"] = sum(register_cnts) / len(register_cnts)
                    cur["knowledge_consistency"] = knowledge_consistency
                    cur["dialog"] = history_copy_text
                    cur["debate_history"] = debate_history_rec
                    cur["retrieved_knowledge"] = all_knowledge_list
                    gpt_answer.append(cur)

                    st_num = turns * query_num
                    ed_num = (turns+1) * query_num
                    with open(f"../data_test/医学专家辩论_{st_num}_{ed_num}_未转换.json", 'w', encoding="utf-8") as sf:
                        json.dump(gpt_answer, sf, ensure_ascii=False, indent=2)
                
                turns += 1
                print(f"第{turns}轮生成已经结束！")
                print("——————————————————————————————————————————————————————————————————————————————")
                # time.sleep(1)
        except Exception as e:
            print("发生如下错误: ", e)
            print(model_name)
            if turns >= 5:
                break
            time.sleep(1)