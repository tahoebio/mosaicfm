# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
"""Preprocess UniProt TSV file to extract relevant protein information for ESM-C
embedding generation.

Author: Hamed Heydari @ Vevo
Date: Apr 01, 2025
"""

import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess UniProt data for protein embedding generation",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/uniprotkb_database_Ensembl_AND_reviewed_2025_03_22.tsv.gz",
        help="Path to the UniProt TSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/human_proteins_processed.tsv",
        help="Path to save the processed TSV file",
    )
    parser.add_argument(
        "--human_only",
        action="store_true",
        help="Filter for human proteins only",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading UniProt data from {args.input}")
    # Load the UniProt TSV file
    df = pd.read_csv(args.input, sep="\t", compression="gzip")

    # Clean up column names (remove spaces and special characters)
    df.columns = [
        col.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .lower()
        for col in df.columns
    ]

    # Filter for human proteins if specified
    if args.human_only:
        print("Filtering for human proteins")
        human_proteins = df[df["organism"].str.contains("Homo sapiens", na=False)]
        df = human_proteins

    # Check for missing data
    missing_seqs = df[df["sequence"].isna()]
    if len(missing_seqs) > 0:
        print(f"Warning: {len(missing_seqs)} proteins have missing sequences")

    # Save the processed dataframe
    print(f"Saving processed data to {args.output}")
    df.to_csv(args.output, sep="\t", index=False)

    # Print summary statistics
    print(f"Total proteins processed: {len(df)}")
    print(f"Columns in output file: {', '.join(df.columns)}")
    print(
        f"Sample of gene names: {', '.join(df['gene_names_primary'].dropna().sample(5).tolist())}",
    )


if __name__ == "__main__":
    main()
