import json
import argparse
import os
from transformers import AutoTokenizer

def main(model_path, model_type):
    '''
    Finalize the tokenizer for the model by setting the chat template and generation config.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ '\nUSER:\n' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '\nASSISTANT:\n' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\nASSISTANT:\n' }}{% endif %}"
    with open(os.path.join(model_path, "generation_config.json"), "r") as f:
        generation_config = json.load(f)
    generation_config["do_sample"] = False
    if model_type == "llama":
        tokenizer.eos_token = "<|eot_id|>"
        generation_config["eos_token_id"] = [128001, 128004, 128009]
        generation_config.pop("temperature", None)
        generation_config.pop("top_p", None)
    elif model_type == "qwen":
        tokenizer.eos_token = "<|im_end|>"
        generation_config["eos_token_id"] = [151643, 151645]
    else:
        raise ValueError("Unsupported model type. Please use a llama or qwen model.")
    
    tokenizer.save_pretrained(model_path)
    with open(os.path.join(model_path, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize the tokenizer for the model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--model_type", type=str, choices=["llama", "qwen"], required=True, help="Type of the model (llama or qwen).")
    args = parser.parse_args()
    
    main(args.model_path, args.model_type)