#!/usr/bin/env python3
"""
Generate protein embeddings using ESM-C models.
Author: Hamed Heydari @ Vevo
Date: April 01, 2025
"""

import os
import pandas as pd
import numpy as np
import torch
import h5py
import argparse
import gc
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate protein embeddings using ESM-C')
    parser.add_argument('--input', type=str, 
                        default="output/human_proteins_processed.tsv",
                        help='Path to the processed protein data TSV')
    parser.add_argument('--output', type=str, 
                        default="output/protein_embeddings_{model}.h5",
                        help='Path to save the HDF5 output file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (reduce if memory issues)')
    parser.add_argument('--model', type=str, default="esmc_300m",
                        choices=["esmc_300m", "esmc_600m"],
                        help='ESM-C model to use (300M or 600M)')
    parser.add_argument('--use_flash_attn', action='store_true',
                        help='Use Flash Attention for faster processing on supported GPUs')
    parser.add_argument('--max_proteins', type=int, default=None,
                        help='Maximum number of proteins to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing HDF5 file if available')
    return parser.parse_args()

def main():
    args = parse_args()
    
    args.output = args.output.format(model=args.model)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load the processed data
    print(f"Loading protein data from {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    
    # Limit the number of proteins if specified (for testing)
    if args.max_proteins is not None:
        df = df.head(args.max_proteins)
        print(f"Limited to {args.max_proteins} proteins for testing")
    
    # Import ESM modules
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        print("Error: Could not import ESM modules. Make sure the ESM package is installed.")
        print("Refer to README.md")
        return
    
    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Clear GPU memory before loading model
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load the model
    print(f"Loading ESM-C model: {args.model}")
    try:
        model = ESMC.from_pretrained(
            args.model
        ).to(device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Function to generate embeddings for a protein sequence
    def get_embeddings(sequence):
        try:
            with torch.no_grad():  # Disable gradient calculation
                protein = ESMProtein(sequence=sequence)
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                
                # Mean pooling over the sequence length (dim=1)
                # This will give a tensor of shape [1, 960] for 300m model
                mean_pooled = logits_output.embeddings.mean(dim=1).squeeze(0).cpu().numpy()
                
                # Clear GPU memory for this computation
                if device == "cuda":
                    del protein_tensor
                    del logits_output
                    torch.cuda.empty_cache()
                
                return mean_pooled
        except Exception as e:
            print(f"Error processing sequence: {e}")
            return None
        
    # Check if we should resume from existing file
    start_idx = 0
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from existing file: {args.output}")
        with h5py.File(args.output, "r") as f:
            # Find how many embeddings have been processed
            start_idx = f.attrs.get('processed_count', 0)
            embedding_dim = f.attrs.get('embedding_dimension', 0)
            print(f"Resuming from index {start_idx}")
            
            # If no embeddings have been processed yet, need to get embedding dimension
            if start_idx == 0 or embedding_dim == 0:
                print("Getting embedding dimension from a sample protein...")
                sample_embedding = get_embeddings(df.iloc[0]['sequence'])
                if sample_embedding is None:
                    print("Error: Failed to get sample embeddings. Check if the model is loaded correctly.")
                    return
                
                embedding_dim = sample_embedding.shape[0]
                print(f"Embedding dimension: {embedding_dim}")
    else:
        # Get embedding dimensions by processing a sample protein
        print("Getting embedding dimension from a sample protein...")
        sample_embedding = get_embeddings(df.iloc[0]['sequence'])
        if sample_embedding is None:
            print("Error: Failed to get sample embeddings. Check if the model is loaded correctly.")
            return
        
        embedding_dim = sample_embedding.shape[0]
        print(f"Embedding dimension: {embedding_dim}")    
    
    # Create or open HDF5 file
    mode = "a" if args.resume and os.path.exists(args.output) else "w"
    with h5py.File(args.output, mode) as f:
        # Create datasets if not resuming or if file is new
        if mode == "w" or "embeddings" not in f:
            # Create datasets in the HDF5 file
            # Store mean-pooled embeddings as the main dataset
            embeddings_dataset = f.create_dataset(
                "embeddings", shape=(len(df), embedding_dim), dtype=np.float32
            )
            
            # Store metadata
            f.create_dataset("uniprot_ids", data=df['entry'].values.astype('S'))
            f.create_dataset("gene_names", data=df['gene_names_primary'].values.astype('S'))
            if 'ensembl_geneid' in df.columns:
                f.create_dataset("ensembl_ids", data=df['ensembl_geneid'].values.astype('S'))
            f.create_dataset("sequence_lengths", data=df['length'].values.astype(np.int32))
            f.create_dataset("sequences", data=df['sequence'].values.astype('S'))
            
            # Add additional metadata attributes
            f.attrs['num_proteins'] = len(df)
            f.attrs['embedding_model'] = args.model
            f.attrs['embedding_dimension'] = embedding_dim
            f.attrs['embedding_type'] = 'mean_pooled'
            f.attrs['flash_attention'] = "enabled" if args.use_flash_attn else "disabled"
            f.attrs['processed_count'] = 0
        else:
            # Get the existing datasets
            embeddings_dataset = f["embeddings"]
        
        # Process proteins in batches to avoid memory issues
        batch_size = args.batch_size
        remaining_proteins = len(df) - start_idx
        num_batches = (remaining_proteins + batch_size - 1) // batch_size
        
        # Process each batch
        failed_count = 0
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", ascii=True, ncols=100):
            current_start_idx = start_idx + batch_idx * batch_size
            current_end_idx = min(start_idx + (batch_idx + 1) * batch_size, len(df))
            
            batch_df = df.iloc[current_start_idx:current_end_idx]
            
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                global_idx = current_start_idx + i
                embedding = get_embeddings(row['sequence'])
                
                if embedding is not None:
                    # Store mean pooled embedding
                    embeddings_dataset[global_idx] = embedding
                else:
                    # Fill with zeros if there was an error
                    embeddings_dataset[global_idx] = np.zeros(embedding_dim)
                    failed_count += 1
                
                # Update processed count and flush to disk periodically
                if global_idx % 10 == 0:
                    f.attrs['processed_count'] = global_idx + 1
                    f.flush()
                    
                    # Force garbage collection to free memory
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
            
            # Clear memory after each batch
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Update final processed count
        f.attrs['processed_count'] = len(df)

    print(f"Embedding generation complete! Saved to {args.output}")
    if failed_count > 0:
        print(f"Warning: Failed to process {failed_count} proteins ({failed_count/len(df):.2%} of total)")

if __name__ == "__main__":
    main()