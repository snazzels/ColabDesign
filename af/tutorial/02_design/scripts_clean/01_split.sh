#!/bin/bash

# Function to show help
show_help() {
    echo "Usage: $0"
    echo "  -h             Display this help and exit."
}

# Parse command-line options
while getopts ":h" opt; do
    case ${opt} in
        h )
            show_help
            exit 0
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            show_help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Create the 'top' directory if it does not exist
mkdir -p top

# Function to process a single PDB file
process_pdb_file() {
    local pdb_file=$1
    local step_number=$(basename "$pdb_file" | sed 's/model_step_\([0-9]*\)\.pdb/\1/')

    # Extract models from the PDB file and save them separately in the 'top' directory
    grep -n 'MODEL\|ENDMDL' "$pdb_file" | cut -d: -f1 | \
    awk -v pdb_file="$pdb_file" -v step_number="$step_number" '{
        if (NR % 2) 
            start=$1 + 1
        else {
            end=$1 - 1
            model_count=NR / 2
            output_file=sprintf("top/step_%s_%d.pdb", step_number, model_count)
            printf "sed -n %d,%dp %s > %s\n", start, end, pdb_file, output_file
        }
    }' | bash -sf
}

# Process all PDB files in the current directory
for pdb_file in *.pdb; do
    if [ -f "$pdb_file" ]; then
        process_pdb_file "$pdb_file"
    fi
done

echo "All PDB files processed successfully and models stored in the 'top' directory."

