# main.py
import os
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# 1️⃣ 加载知识库切片
# -----------------------------
chunks_dir = "D:/test/Buddhism-LLM/data/chunks"
documents = []

for filename in os.listdir(chunks_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(chunks_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)

print(f"Loaded {len(documents)} document chunks.")

# -----------------------------
# 2️⃣ 文本向量化
# -----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = embed_model.encode(documents, convert_to_tensor=True)
vectors_np = vectors.cpu().detach().numpy()
faiss.normalize_L2(vectors_np)  # 归一化向量

# -----------------------------
# 3️⃣ 建立 FAISS 索引
# -----------------------------
dim = vectors_np.shape[1]
index = faiss.IndexFlatIP(dim)  # 内积相似度
index.add(vectors_np)
print(f"FAISS index with {index.ntotal} vectors created.")

# -----------------------------
# 4️⃣ 检索函数
# -----------------------------
def retrieve(query, top_k=3):
    query_vec = embed_model.encode([query], convert_to_tensor=True)
    query_np = query_vec.cpu().detach().numpy()
    faiss.normalize_L2(query_np)
    D, I = index.search(query_np, top_k)
    results = [documents[i] for i in I[0]]
    return results

# -----------------------------
# 5️⃣ 加载 LLM 模型
# -----------------------------
model_name = "bigscience/bloom-560m"  # 示例，可换成你的本地模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# -----------------------------
# 6️⃣ RAG 问答函数
# -----------------------------
def answer_question(query):
    context_chunks = retrieve(query, top_k=3)
    context = "\n".join(context_chunks)
    prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{query}\n回答："
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# 7️⃣ 测试
# -----------------------------
if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))
    
    while True:
        query = input("\n请输入问题（输入 exit 退出）：")
        if query.lower() == "exit":
            break
        ans = answer_question(query)
        print(f"\n回答：{ans}")
