import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)


# 概念提取函数
def extract_concepts(question: str) -> str:
    prompt = f"""You are a helpful assistant for concept extraction in visual question answering.
Extract the core concepts (objects, attributes, or entities) involved in the following question, as a comma-separated list.

Question: "{question}"
Concepts:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return output_text.strip()

# 读取 TextVQA 验证集（根据你文件结构修改路径）
with open("data/textvqa/data.json") as f:
    samples = json.load(f)

results = []

# 遍历每个问题并提取概念
for i, item in enumerate(samples):
    question = item["question"]
    image_path = item["image_path"]
    concepts = extract_concepts(question)
    results.append({
        "id": item["id"],
        "question": question,
        "concepts": concepts,
        "image_path": image_path
    })
    print(f"[{i+1}/{len(samples)}] Q: {question}\n→ Concepts: {concepts}\n")

# 确保保存文件夹存在
os.makedirs("data/results", exist_ok=True)

# 保存结果到 data/results/ 文件夹
with open("data/results/data_concepts.json", "w") as f:
    json.dump(results, f, indent=2)

print("All concepts extracted and saved to results/textvqa_val_concepts.json.")