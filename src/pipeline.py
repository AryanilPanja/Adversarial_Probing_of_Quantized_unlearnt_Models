import os
import json
import argparse
from tqdm import tqdm
from model_runner import ModelRunner
import time

import attacks.translation as atk_translation
import attacks.format_shifting as atk_format
import attacks.cipher as atk_cipher
import attacks.persona as atk_persona
import attacks.indirect as atk_indirect
import attacks.gradient_suffix as atk_gradient

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_queries(data_path):
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r') as f:
            if data_path.endswith('.json'):
                data = json.load(f)
                return data # Return the list of dicts directly
            else:
                return [line.strip() for line in f.readlines()]
    print("Warning: No viable query data provided. Using dummy MUSE knowledge queries.")
    return [
        "Who are Harry's parents?",
        "What is the plot of Harry Potter and the Sorcerer's Stone?",
        "Where is Hogwarts located?",
        "Tell me about Lord Voldemort's original name."
    ]

def evaluate_model(model_name, model_path, state, task_type, queries_list, results_dir):
    print(f"\n==============================================")
    print(f"Evaluating Model: {model_name} | State: {state} | Task: {task_type}")
    print(f"==============================================")
    
    try:
        runner = ModelRunner(model_path=model_path, state=state)
    except Exception as e:
        print(f"Failed to load {model_name} at state {state}. Error: {e}")
        return

    start_time = time.time()
    
    if task_type == "cloze":
        print("Running cloze scoring...")
        results = runner.run_cloze_scoring(queries_list)
    else:
        perturbation_types = queries_list[0].keys()
        results = []

        for i, _ in enumerate(queries_list):
            results.append({
                "original_index": i,
                "perturbations": {}
            })

        for p_type in perturbation_types:
            # For JSON dicts, some permutations like 'clean' could just be empty spaces if logic isn't careful, 
            # but we ensured generation populated the dicts.
            print(f"Running inference for perturbation: {p_type}")
            prompts = [q_dict[p_type] for q_dict in queries_list]
            
            try:
                responses = runner.run_inference(prompts, max_new_tokens=150, temperature=0.0)
                for i, response in enumerate(responses):
                    results[i]["perturbations"][p_type] = {
                        "prompt": prompts[i],
                        "response": response
                    }
            except Exception as e:
                print(f"Inference failed for perturbation {p_type}: {e}")
                for i in range(len(prompts)):
                    results[i]["perturbations"][p_type] = {
                        "prompt": prompts[i],
                        "response": f"ERROR: {e}"
                    }

    import torch
    del runner.model
    torch.cuda.empty_cache()
    
    print(f"Time taken for {model_name}_{state}: {time.time() - start_time:.2f}s")

    output_filename = os.path.join(results_dir, f"{model_name}_{state}_{task_type}_results.json")
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Adversarial Probing Evaluation Pipeline")
    parser.add_argument('--data_file', type=str, default='', help='Path to MUSE or clean queries CSV/JSON.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model being evaluated.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--state', type=str, default='auto', help='Quantization state (auto, fp16, 8bit, 4bit).')
    parser.add_argument('--task_type', type=str, default='generative', help='Task execution mode (generative, cloze)')
    parser.add_argument('--results_dir', type=str, default='../results', help='Directory to save outputs.')
    args = parser.parse_args()

    ensure_dir(args.results_dir)

    print(f"Loading queries for {args.task_type} mode...")
    loaded_data = load_queries(args.data_file)
    
    if args.task_type == "cloze":
        # Data remains as list of {"prefix": x, "target": y}
        queries_list = loaded_data
        print(f"Cloze examples loaded: {len(queries_list)}")
    else:
        print("Generating adversarial perturbations from separate modules...")
        queries_list = []
        for item in tqdm(loaded_data):
            query = item["query"] if isinstance(item, dict) else item
            
            indirect_q = item["indirect_query"] if (isinstance(item, dict) and "indirect_query" in item) else atk_indirect.generate_attack(query)
            
            perms = {
                "clean": query,
                "translated": atk_translation.generate_attack(query),
                "format_shifted": atk_format.generate_attack(query),
                "ciphered": atk_cipher.generate_attack(query),
                "persona": atk_persona.generate_attack(query),
                "indirect": indirect_q
                #"gradient_suffix": atk_gradient.generate_attack(query)
            }
            queries_list.append(perms)

        print("Permutations generated. Total queries prepared:", len(queries_list))

    # Execute for the single provided model
    evaluate_model(
        model_name=args.model_name,
        model_path=args.model_path,
        state=args.state,
        task_type=args.task_type,
        queries_list=queries_list,
        results_dir=args.results_dir
    )

    print(f"\nEvaluation for {args.model_name} complete. Check the results folder.")

if __name__ == "__main__":
    main()
