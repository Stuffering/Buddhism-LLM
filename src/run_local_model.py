from localai import LocalAI

# 1️⃣ 指定本地模型路径
model_path = "D:/test/buddhism_llm_models/qwen2-7b-instruct-q5_k_m.gguf"

# 2️⃣ 初始化 LocalAI，使用 GPU
llm = LocalAI(model_path=model_path, device="cuda")  # "cuda" 使用 GPU，"cpu" 使用 CPU

# 3️⃣ 测试生成
prompt = "请用简短、明确的语言回答：书名：《菩萨戒指要》问题：这本书的特色是什么？"
output = llm.generate(prompt)

print("模型输出：\n", output)
