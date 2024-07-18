# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 source_directory destination_directory percentage"
    exit 1
fi

source_dir=$1 #path to the original cellxgene-MDS (dl from s3 if you don't have it: s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2024-04-29_MDS/)
destination_dir=$2 
percentage=$3

mkdir -p "$destination_dir"

chunk_dirs=($(find "$source_dir" -maxdepth 1 -type d -name 'chunk_*'))

# Calculate the number of chunk directories to copy
total_dirs=${#chunk_dirs[@]}
num_to_copy=$((total_dirs * percentage / 100))
shuffled_dirs=($(shuf -e "${chunk_dirs[@]}"))
echo $num_to_copy
echo $total_dirs
# Copy the specified percentage of chunk directories to the destination directory
for ((i=0; i<num_to_copy; i++)); do
    cp -r "${shuffled_dirs[$i]}" "$destination_dir"
done


# Run the Python code to create the index file which merges the chunks and creates index.json
python3 - <<EOF
from streaming.base.util import merge_index

out_root = '$destination_dir'
merge_index(out_root, keep_local=True)
print(f"Merging Index Complete at {out_root}.")
EOF