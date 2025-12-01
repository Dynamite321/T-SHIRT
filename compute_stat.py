import os
import json
import torch
import pickle
import argparse
import numpy as np


def main(input_path, output_path, ratio=0.5):
    pkl_files = [f for f in os.listdir(input_path) if f.endswith('.pkl')]
    ifd_files = [f for f in pkl_files if 'sample_ifds' in f]
    noisy_ifd_files = [f for f in pkl_files if 'sample_noisy_ifds' in f]
    noisy_pmi_files = [f for f in pkl_files if 'sample_noisy_pmis' in f]
    ifd_files.sort()
    noisy_ifd_files.sort()
    noisy_pmi_files.sort()

    ifds = []
    for ifd_file in ifd_files:
        ifd_path = os.path.join(input_path, ifd_file)

        with open(ifd_path, 'rb') as f:
            sample_ifds = pickle.load(f)
        
        ifds.extend(sample_ifds)
    
    noisy_ifds = []
    for noisy_ifd_file in noisy_ifd_files:
        noisy_ifd_path = os.path.join(input_path, noisy_ifd_file)

        with open(noisy_ifd_path, 'rb') as f:
            sample_noisy_ifds = pickle.load(f)
        
        noisy_ifds.extend(sample_noisy_ifds)
    
    noisy_pmis = []
    for noisy_pmi_file in noisy_pmi_files:
        noisy_pmi_path = os.path.join(input_path, noisy_pmi_file)

        with open(noisy_pmi_path, 'rb') as f:
            sample_noisy_pmis = pickle.load(f)
        
        noisy_pmis.extend(sample_noisy_pmis)
    
    assert len(ifds) == len(noisy_ifds) == len(noisy_pmis), "Length of ifds, noisy_ifds, and noisy_pmis must be the same."

    avg_noisy_ifds = []
    std_noisy_ifds = []
    for noisy_ifd in noisy_ifds:
        avg_noisy_ifd = np.mean(noisy_ifd)
        std_noisy_ifd = np.std(noisy_ifd)
        avg_noisy_ifds.append(avg_noisy_ifd)
        std_noisy_ifds.append(std_noisy_ifd)
    
    token_pmis = []
    for noisy_pmi in noisy_pmis:
        for pmi in noisy_pmi:
            token_pmis.extend(pmi)
    
    print(len(token_pmis))
    threshold = np.percentile(np.abs(token_pmis), (1 - ratio) * 100)
    print(threshold)

    avg_noisy_sifds = []
    std_noisy_sifds = []
    for noisy_pmi in noisy_pmis:
        pmi_tensor = torch.stack([torch.as_tensor(v) for v in noisy_pmi])        # same length inside this list

        mask    = pmi_tensor.abs() > threshold
        counts  = mask.sum(dim=1).clamp(min=1)
        sums    = (pmi_tensor * mask).sum(dim=1)
        sifd    = torch.exp(sums / counts)

        avg_noisy_sifds.append(sifd.mean().item())
        std_noisy_sifds.append(sifd.std(unbiased=False).item())
        
    assert len(ifds) == len(avg_noisy_ifds) == len(std_noisy_ifds) == len(avg_noisy_sifds) == len(std_noisy_sifds), "Length of ifds, avg_noisy_ifds, std_noisy_ifds, avg_noisy_sifds, and std_noisy_sifds must be the same."
    
    stats = []
    for i in range(len(ifds)):
        stats.append({
            'ifd': ifds[i],
            'avg_noisy_ifd': avg_noisy_ifds[i],
            'std_noisy_ifd': std_noisy_ifds[i],
            f'avg_noisy_sifd_{int(ratio*100)}': avg_noisy_sifds[i],
            f'std_noisy_sifd_{int(ratio*100)}': std_noisy_sifds[i]
        })

    # Save the stats to a JSON file
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for dataset.")
    parser.add_argument("--input_path", type=str, default='token_info/alpaca_gpt4', help="Path to the input directory containing .pkl files.")
    parser.add_argument("--output_path", type=str, default='stats/alpaca_gpt4/stat.json', help="Path to save the output JSON file.")
    parser.add_argument("--ratio", type=float, default=0.5, help="Token selection ratio for computing sifds.")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    main(args.input_path, args.output_path, args.ratio)