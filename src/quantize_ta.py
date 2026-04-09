import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

def quantize_and_save(src_path, dst_path, bnb_config):
    print(f"Loading tokenizer from {src_path}...")
    tokenizer = AutoTokenizer.from_pretrained(src_path)
    
    print(f"Quantizing model with config: {bnb_config.to_dict()}...")
    model = AutoModelForCausalLM.from_pretrained(
        src_path,
        device_map="auto",
        quantization_config=bnb_config
    )
    
    print(f"Saving quantized model and tokenizer to {dst_path}...")
    os.makedirs(dst_path, exist_ok=True)
    model.save_pretrained(dst_path)
    tokenizer.save_pretrained(dst_path)
    print(f"Saved successfully to {dst_path}.\n")
    
    # Cleanup memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    src_model_dir = "../models/task_arithmetic/fp16_unlearned_model"
    int8_model_dir = "../models/task_arithmetic/int8_unlearned_model"
    int4_model_dir = "../models/task_arithmetic/int4_unlearned_model"
    
    if not os.path.exists(src_model_dir):
        print(f"Error: Source model directory {src_model_dir} does not exist.")
        return

    # 8-bit Quantization
    print("--- Starting 8-bit Quantization ---")
    bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)
    quantize_and_save(src_model_dir, int8_model_dir, bnb_8bit)

    # 4-bit Quantization
    print("--- Starting 4-bit Quantization ---")
    bnb_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    quantize_and_save(src_model_dir, int4_model_dir, bnb_4bit)

if __name__ == "__main__":
    main()
