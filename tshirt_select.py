import os
import json
import datasets
import argparse
import heapq
import numpy as np

def get_top_k_indices(arr, key, k):
    if not arr:
        return []
    
    if key == 'ifd':
        valid_items = [(item['ifd'], i) for i, item in enumerate(arr) if item['ifd'] < 1 and not np.isnan(item['ifd'])]
    else:
        valid_items = [(item[key], i) for i, item in enumerate(arr) if not np.isnan(item[key])]

    top_k = heapq.nlargest(k, valid_items)

    top_k_indices = [index for _, index in top_k]
    return top_k_indices


def hierarchical_get_top_k_indices(stats, key, k):
    if not stats:
        return []

    key_avg = key
    key_std = key.replace("avg", "std")

    # ---------- stage 1 + 2 : retain ≤ 2k rows with largest avg metric ----------
    top_2k = []
    for idx, d in enumerate(stats):
        avg_ifd  = d.get("avg_noisy_ifd", np.nan)
        avg_sifd = d.get(key_avg,   np.nan)
        if np.isnan(avg_ifd) or np.isnan(avg_sifd) or avg_ifd >= 1:
            continue

        if len(top_2k) < 2 * k:
            heapq.heappush(top_2k, (avg_sifd, idx))
        elif avg_sifd > top_2k[0][0]:
            heapq.heapreplace(top_2k, (avg_sifd, idx))

    # ---------- stage 3 : pick k rows with smallest std metric ------------------
    candidates = [
        (stats[idx][key_std], idx)
        for _, idx in top_2k
        if not np.isnan(stats[idx][key_std])
    ]
    best_k = heapq.nsmallest(k, candidates, key=lambda t: t[0])

    return [idx for _, idx in best_k]


def main(dataset_path, stat_path, key, k, output_path):
    with open(stat_path, 'r') as f:
        stats = json.load(f)
    
    if 'magpie' in dataset_path.lower():
        dataset = datasets.load_dataset(dataset_path, split="train")
    else:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    
    top_k_indices = hierarchical_get_top_k_indices(stats, key, k)
    
    if 'magpie' in dataset_path.lower():
        subset = dataset.select(top_k_indices)
    else:
        subset = [dataset[i] for i in top_k_indices]

    output_data = []
    for i in range(len(subset)):
        if 'magpie' in dataset_path.lower():
            output_data.append({
                'id': str(i),
                'conversations': subset[i]['conversations'],
            })
        else:
            instruction = subset[i]["instruction"]
            input = subset[i]["input"]
            output = subset[i]["output"]

            if input == '':
                prompt = instruction
            else:
                prompt = instruction + '\n' + input
            
            conversations = [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": output
                }
            ]
            output_data.append({
                'id': str(i),
                'conversations': conversations,
            })

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select top k indices based on a specific key.")
    parser.add_argument('--dataset_path', type=str, default='datasets/alpaca_gpt4/alpaca_gpt4_data.json', help='Path to the dataset file')
    parser.add_argument('--stat_path', type=str, default='stats/alpaca_gpt4/stat.json', help='Path to the dataset file')
    parser.add_argument('--key', type=str, default='avg_noisy_sifd_50', help='Key to sort by')
    parser.add_argument('--num', type=int, default=2600, help='Number of top indices to select')
    parser.add_argument('--output_path', type=str, default='datasets/alpaca_gpt4/tshirt_k_50.json', help='Path to save the output dataset')  

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    main(args.dataset_path, args.stat_path, args.key, args.num, args.output_path)