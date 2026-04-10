# Adversarial Probing Pipeline

This project evaluates different models (Base, NPO, Task Arithmetic) across quantization states against adversarial queries.

## Running the Pipeline

The pipeline is now invoked sequentially and modularly to prevent Out-Of-Memory errors when loading multiple models.
You can execute all valid models present in `../models/` directory by simply running `eval_all.sh`:

```bash
source .venv/bin/activate
./eval_all.sh
```

Alternatively, you can test a single model directly:

```bash
source .venv/bin/activate
cd src
# Default state 'auto' assumes pre-quantized models with 'config.json' holding quantization info
python pipeline.py --model_name "base_fp16" --model_path "../models/base/base_gemma3_1b_it_fp16" --state "auto"
```

To run the eval_all.sh in different modes of task the following commands can be used:

```bash
./eval_all.sh --task_type cloze --data_file test_cloze.json #for cloze task
./eval_all.sh --task_type generative --data_file test_queries.json #for generative task
```
<!--# cuda 126, 2.9.0-->