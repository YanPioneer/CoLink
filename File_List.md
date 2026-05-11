..\Data\CMeKG\CMeKG_ACT.py——Class definition for CMeKG retrieval, you need a embedding model (like gte-multilingual-base) for KG.
..\Data\CMeKG\CMeKG_Extract.py——Code for CMeKG knowledge linking and activation via ACT. You need update the openai key.
..\Data\WIKG\WIKG.py——Class definition for WIKG retrieval, you need a embedding model (like gte-multilingual-base) for KG.
..\Data\WIKG\kg_extract.py——Code for WIKG knowledge linking and activation via ACT. You need update the openai key.

*Note: Before running the above code, you need to load the corresponding KG using neo4j. If you don't want to execute it, you can directly use the data we retrieved: ..\Data\CMtMedQA_train_sampled400_CMeKGpath_gpt4o_mini_alpha_06.json, ..\Data\CMtMedQA_test_sampled50_CMeKGpath_gpt4o_mini_alpha_06.json, ..\Data\medical_data_v3_summary_text_kgpath_gpt4o_mini_alpha_06_allkg_final.json, and ..\Data\medical_data_v3_eval_kgpath_gpt4o_mini_alpha_06_allkg_final.json.

..\Data\datadeal_knowledge.py——Summary generation for reconstruction loss.
..\Data\gpt4_MAD_医学_optim.py——Knowledge-Augmented Multi-Agent debate framework for data generation. 

..\Model_Code\Compressor\slot_attention_dynamic.py——Code for AdaSlot
..\Model_Code\Compressor\slot_attention_know.py——Class definition for stage-1 knowledge slot training.
..\Model_Code\Compressor\train_know.py——Code for stage-1 knowledge slot training.
..\Model_Code\test_know.pyCode for stage-1 knowledge slot test.

..\Model_Code\Compressor\slot_attention.py——Class definition for stage-1 memory slot training.
..\Model_Code\Compressor\train_mem.py——Code for stage-1 memory slot training.
..\Model_Code\test.py——Code for stage-1 memory slot test.

..\Model_Code\train_stage2.py——Code for stage-2 memory slot and LLM training.
..\Model_Code\train_stage2_know.py——Code for stage-2 memory slot, knowledge slot and LLM training.
..\Model_Code\train_stage2_know.py——Test code.
