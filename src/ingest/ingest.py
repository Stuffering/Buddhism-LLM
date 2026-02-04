import os
import re

# ========== 配置路径 ==========
RAW_DIR = r"D:\test\Buddhism-LLM\data\raw"
CLEANED_DIR = r"D:\test\Buddhism-LLM\data\cleaned"
CHUNKS_DIR = r"D:\test\Buddhism-LLM\data\chunks"

CHUNK_SIZE = 800  # 每片字符数，可根据需要调整

# ========== 确保输出目录存在 ==========
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ========== 清洗文本函数 ==========
def clean_text(text):
    """
    清洗文本，只保留中文
    1. 去掉换行、空格
    2. 去掉非中文字符
    """
    text = text.replace("\n", "").replace("\r", "").strip()
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return text

# ========== 切片函数 ==========
def slice_text(text, chunk_size=CHUNK_SIZE):
    """
    将文本切片为固定长度块
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ========== 处理单个文件 ==========
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # 清洗文本
    cleaned = clean_text(raw_text)
    
    # 保存清洗后的文本
    base_name = os.path.basename(file_path)
    cleaned_file = os.path.join(CLEANED_DIR, base_name)
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.write(cleaned)
    
    # 切片
    chunks = slice_text(cleaned)
    for idx, chunk in enumerate(chunks, 1):
        chunk_file = os.path.join(CHUNKS_DIR, f"{base_name}_chunk{idx}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)
    
    print(f"{base_name} -> {len(chunks)} chunks created")

# ========== 遍历 raw 目录所有 txt 文件 ==========
for file_name in os.listdir(RAW_DIR):
    if file_name.endswith(".txt"):
        file_path = os.path.join(RAW_DIR, file_name)
        process_file(file_path)

print("All files processed!")
