import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelRunner:
    def __init__(self, model_path, state="fp16", device="cuda"):
        """
        Initializes the ModelRunner with the specified precision state.
        :param model_path: Path to the HuggingFace model.
        :param state: Model precision state ('fp16', '8bit', '4bit').
        :param device: Device to map the model to.
        """
        self.model_path = model_path
        self.state = state
        self.device = device
        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self):
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model from {self.model_path} in state: {self.state}...")
        
        # Base arguments
        model_kwargs = {
            "device_map": "auto", 
            "trust_remote_code": True
        }

        # Handling different quantization states
        if self.state == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.state == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.state == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.state == "auto":
            # State 'auto' is used for pre-quantized models where config.json natively holds the quantization config
            model_kwargs["torch_dtype"] = "auto"
        else:
            raise ValueError(f"Unsupported model state: {self.state}. Must be 'auto', 'fp16', '8bit', or '4bit'.")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()
        print("Model loaded successfully.")

    def run_inference(self, queries, max_new_tokens=150, temperature=0.7, batch_size=4):
        """
        Runs inference on a list of queries.
        Supports automatic batching for efficiency.
        """
        all_responses = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            
            # If the tokenizer has a chat template, attempt to use it.
            # Otherwise, just fall back to standard encoding.
            prompts = []
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
                for q in batch:
                    messages = [{"role": "user", "content": q}]
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
            else:
                # Basic instruction wrapper just in case
                prompts = [f"Instruction: {q}\nAnswer:" for q in batch]

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            decoded_batch = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            all_responses.extend([resp.strip() for resp in decoded_batch])

        return all_responses
