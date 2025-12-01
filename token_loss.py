import argparse
import json
import pickle
import random
import math
import os
import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True

model_paths = {
    'gpt2': 'gpt2',
}


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_instruction(instruction, input_text):
    """
    Format the instruction and input text into a single string.
    
    Args:
        instruction (str): The instruction text
        input_text (str): The input text to append
        
    Returns:
        str: Formatted instruction
    """
    if input_text == '':
        return instruction + '\n'
    else:
        return instruction + '\n' + input_text + '\n'
    

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
def compute_token_losses(input_ids, model, wte, noise=None):
    """
    Return per‑token negative‑log‑prob losses.
    `input_ids` may contain *any* batch size; pass the matching `noise`
    tensor (or None).  No truncation/encoding logic lives here anymore.
    """
    with torch.inference_mode():
        if noise is None:
            outputs = model(input_ids=input_ids)
        else:
            # pre‑computed embeddings + noise
            inp_emb = wte(input_ids) + noise
            outputs = model(inputs_embeds=inp_emb)

        shift_logits = outputs.logits[:, :-1, :].float()  # (B, L-1, C)
        B = shift_logits.shape[0]
        shift_labels = input_ids[:, 1:].expand(B, -1)  # (B, L-1)

        token_losses = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).reshape(shift_labels.size())      # (batch,  L‑1)

    return token_losses


def compute_token_wise_loss(instruction, response, model, tokenizer, device, 
                           max_length=1024, neft_alpha=5.0, num_noise_samples=0):
    """
    Return:
        response_losses          – list[T]
        pmi_values               – list[T]                     (always computed)
        all_noisy_pmi_values     – list[N][T]  or  None        (N = num_noise_samples)
    """
    # ---------- helpers ---------------------------------------------------- #
    def _tokenise(text, lim):
        return tokenizer(text, return_tensors="pt",
                         truncation=True, max_length=lim).input_ids.to(device)

    # ---------- shared ids / lengths -------------------------------------- #
    has_bos = tokenizer.bos_token is not None and 'gpt2' not in model.config.architectures[0].lower()
    full_text = instruction + response
    full_ids = _tokenise(full_text, max_length)
    instruction_len = len(tokenizer(instruction)['input_ids'])
    max_response_length = max_length - instruction_len + (1 if has_bos else 0)
    response_ids = _tokenise(response, max_response_length)
    wte = model.get_input_embeddings()
    L, C = full_ids.shape[1], wte.embedding_dim
    offset = 0 if has_bos else 1

    # ---------- build the noise batch ------------------------------------- #
    N = max(0, num_noise_samples)          # guard against negatives
    batch = N + 1                          # +1 for the zero‑noise baseline

    if neft_alpha > 0 and N > 0:
        # (batch, L, C)  – row[0] is all‑zeros
        rand_noise = torch.empty((N, L, C),
                                 dtype=wte.weight.dtype,
                                 device=device).uniform_(-1, 1)
        rand_noise = torch.cat([torch.zeros((1, L, C), dtype=wte.weight.dtype).to(device), rand_noise], dim=0)

        scale = neft_alpha / math.sqrt(L * C)
        noise_big = rand_noise * scale
    else:
        # no noise requested → single zero‑noise row
        noise_big = torch.zeros((1, L, C),
                                dtype=wte.weight.dtype,
                                device=device)
    
    # ---------- conditional losses, batched ------------------------------- #
    cond_loss = compute_token_losses(full_ids, model, wte, noise=noise_big)

    # keep only y‑part
    cond_resp_loss_batch = cond_loss[:, instruction_len-1:]          # (batch, |y|)

    # ---------- unconditional losses, batched ----------------------------- #
    if neft_alpha > 0 and N > 0:
        resp_noise_big = noise_big[:, instruction_len:]
    
    uncond_resp_loss_batch = compute_token_losses(response_ids, model, wte, noise=resp_noise_big)

    # ---------- clean ifd and noisy ifd ----------------------------------- #
    cond_ppls = torch.exp(torch.mean(cond_resp_loss_batch, dim=-1))
    uncond_ppls = torch.exp(torch.mean(uncond_resp_loss_batch, dim=-1))
    ifds = cond_ppls / uncond_ppls
    ifd = ifds[0].cpu().item()
    # truchate the last one to reach 30
    noisy_ifds = ifds[1:-1].cpu().tolist() if batch > 1 else None
    
    # ---------- PMI = log P(y_i) – log P(y_i|x, y_{<i}) ------------------- #
    pmi_batch = cond_resp_loss_batch[:, offset:] - uncond_resp_loss_batch

    # ---------- unpack results -------------------------------------------- #
    pmi_values           = pmi_batch[0].cpu().tolist()                 # baseline
    # truchate the last one to reach 30
    all_noisy_pmi_values = (pmi_batch[1:-1].cpu().tolist()
                            if batch > 1 else None)

    return ifd, noisy_ifds, pmi_values, all_noisy_pmi_values


def process_dataset(json_path, dataset_name, num_shards, shard_index, model_name, max_length=1024,
                    neft_alpha=0.0, num_noise_samples=0, output_path="token_info", use_bf16=False):
    """
    Processes the dataset and computes token-wise loss for each example.
    
    Args:
        json_path (str): Path to the JSON file with instruction-response pairs
        model_name (str): Name of the Hugging Face model to use
        compute_pmi (bool): Whether to compute pointwise mutual information
        max_length (int): Maximum sequence length for tokenization
        
    Returns:
        tuple: (token_position_losses, token_position_pmi) dictionaries mapping positions to values
    """
    # Initialize storage for token-wise losses by position
    sample_ifds = []
    sample_noisy_ifds = []
    sample_pmis = []
    sample_noisy_pmis = []
    
    try:
        model_path = model_paths.get(model_name, model_name)
        # Load model and tokenizer
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if use_bf16:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
        )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        
        # Display model context window information
        model_max_length = getattr(model.config, "max_position_embeddings", None) or max_length
        tokenizer_max_length = getattr(tokenizer, "model_max_length", None) or max_length
        effective_max_length = min(model_max_length, tokenizer_max_length, max_length)
        print(f"Using maximum sequence length: {effective_max_length} tokens")
        
        # Read the dataset
        if dataset_name != 'magpie':
            print(f"Reading dataset from {json_path}")
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        else:
            print(f"Loading dataset: {dataset_name}")
            json_data = datasets.load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", split="train")
        
        if isinstance(json_data, list):                       # plain list / json
            total = len(json_data)
            shard_size = math.ceil(total / num_shards)
            start = shard_index * shard_size
            end   = min(start + shard_size, total)
            json_data = json_data[start:end]
        else:                                                 # datasets.Dataset
            json_data = json_data.shard(
                num_shards=num_shards, index=shard_index, contiguous=True
            )
        
        # Process each example
        print(f"Processing {len(json_data)} examples...")
        for i, data in tqdm(enumerate(json_data), total=len(json_data)):
            if dataset_name != 'magpie':
                instruction = data['instruction']
                input_text = data.get('input', '')
                response = data['output']
            else:
                instruction = data['conversations'][0]['value']
                input_text = ''
                response = data['conversations'][1]['value']
            
            # Format the full instruction
            full_instruction = format_instruction(instruction, input_text)

            # Compute token-wise loss and PMI
            ifd, noisy_ifds, pmi_values, all_noisy_pmi_values = compute_token_wise_loss(
                full_instruction, response, model, tokenizer, device, 
                effective_max_length, neft_alpha, num_noise_samples
            )
            
            if ifd is not None:
                sample_ifds.append(ifd)
            if noisy_ifds is not None:
                sample_noisy_ifds.append(noisy_ifds)
            if pmi_values is not None:
                sample_pmis.append(pmi_values)
            if all_noisy_pmi_values is not None:
                sample_noisy_pmis.append(all_noisy_pmi_values)

        save_results(dataset_name, model_name, shard_index,
                     sample_ifds, sample_noisy_ifds,
                     sample_pmis, sample_noisy_pmis,
                     output_path=output_path)

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_path}")
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        
    return


def save_results(dataset_name, model_name, shard_index,
                 sample_ifds, sample_noisy_ifds,
                 sample_pmis, sample_noisy_pmis,
                 output_path="token_info"):
    """Save computation results"""
    if sample_ifds:
        with open(f"{output_path}/{dataset_name}_{model_name}_sample_ifds_{shard_index}.pkl", "wb") as f:
            pickle.dump(sample_ifds, f)
    if sample_noisy_ifds:
        with open(f"{output_path}/{dataset_name}_{model_name}_sample_noisy_ifds_{shard_index}.pkl", "wb") as f:
            pickle.dump(sample_noisy_ifds, f)
    if sample_pmis:
        with open(f"{output_path}/{dataset_name}_{model_name}_sample_pmis_{shard_index}.pkl", "wb") as f:
            pickle.dump(sample_pmis, f)
    if sample_noisy_pmis:
        with open(f"{output_path}/{dataset_name}_{model_name}_sample_noisy_pmis_{shard_index}.pkl", "wb") as f:
            pickle.dump(sample_noisy_pmis, f)


def main():
    """
    Main function to parse arguments and run the token-wise loss analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="Calculate token-wise loss for instruction-response pairs")
    parser.add_argument("--json_path", type=str, default="datasets/alpaca_gpt4/alpaca_gpt4_data.json", help="Path to JSON file with instruction-response pairs")
    parser.add_argument("--dataset_name", type=str, default="alpaca_gpt4", help="Name of the dataset to process")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of Hugging Face model to use")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum token length for sequences")
    parser.add_argument("--neft_alpha", type=float, default=5.0, help="Noise level")
    # choose 31 instead of 30 to achieve a batch size of 2**k (31+1=32)
    parser.add_argument("--num_noise_samples", type=int, default=31, help="Number of noise samples to generate")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the dataset into")
    parser.add_argument("--shard_index", type=int, default=0, help="Index for sharding the dataset")
    parser.add_argument("--output_path", type=str, default="token_info/alpaca_gpt4", help="Output directory for results")
    parser.add_argument("--use_bf16", type=bool, default=True, help="Use bf16 precision for model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.seed)
    
    try:
        # Process the dataset
        process_dataset(
            args.json_path, args.dataset_name, args.num_shards, args.shard_index,
            args.model_name, max_length=args.max_length,
            neft_alpha=args.neft_alpha, num_noise_samples=args.num_noise_samples,
            output_path=args.output_path, use_bf16=args.use_bf16
        )

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()