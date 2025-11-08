import pandas as pd
import os
import gzip
import collections
import sys

def generate_nwk_and_metadata_for_larry(raw_file_path, output_dir):
    """
    Generates one .nwk file (star tree) per barcode and a corresponding 
    metadata dictionary mapping cell IDs to cell types, using the raw 
    LARRY hematopoiesis data. Applies filtering from the CARTA paper.

    Args:
        raw_file_path (str): Path to 'in_vitro_data_per_cell.csv.gz'
        output_dir (str): Directory to save the .nwk files and metadata files.
    """
    
    print(f"Loading raw data from {raw_file_path}...")
    try:
        # Read directly from the gzipped file
        with gzip.open(raw_file_path, 'rt') as f:
            df_cells = pd.read_csv(f)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_file_path}", file=sys.stderr)
        print("Please download it from the AllonKleinLab repo first.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading or reading CSV file: {e}", file=sys.stderr)
        return

    print(f"Loaded {len(df_cells)} total cell records.")
    
    # --- Step 1: Clean and filter cell data (as per CARTA paper description) ---
    
    print("Applying initial cell filtering...")
    # Rename columns for clarity if necessary (adjust if column names differ)
    # Assuming columns like 'Cell barcode', 'Cell type annotation'
    # Check actual column names using df_cells.columns
    barcode_col = 'barcode' # Adjust if needed, e.g., 'Cell barcode'
    cell_type_col = 'cell_type' # Adjust if needed, e.g., 'Cell type annotation'
    cell_id_col = df_cells.columns[0] # Often the first column is a unique cell ID

    if barcode_col not in df_cells.columns or cell_type_col not in df_cells.columns:
         print(f"Error: Expected columns '{barcode_col}' and '{cell_type_col}' not found.", file=sys.stderr)
         print(f"Actual columns: {df_cells.columns.tolist()}", file=sys.stderr)
         return

    # Keep only relevant columns and drop rows with missing barcode or type
    df_cells = df_cells[[cell_id_col, barcode_col, cell_type_col]].dropna()

    # "merged the 'pDC' and 'Ccr7 DC' cell types into one 'DC' cell type"
    df_cells[cell_type_col] = df_cells[cell_type_col].replace({
        'pDC': 'DC',
        'Ccr7 DC': 'DC'
    })
    
    # "removed cells with the undifferentiated cell type"
    df_cells = df_cells[df_cells[cell_type_col] != 'Undifferentiated']
    
    print(f"{len(df_cells)} cells remaining after initial filtering.")

    # --- Step 2: Filter barcodes based on potency counts (as per CARTA paper) ---
    
    print("Filtering barcodes based on potency occurrence count...")
    # Group by barcode to find the set of cell types (potency) for each
    df_potency = df_cells.groupby(barcode_col)[cell_type_col].apply(frozenset)
    df_potency = df_potency.to_frame(name='potency_set')
    
    # Count how often each unique potency set appears
    potency_counts = df_potency['potency_set'].value_counts()
    
    # Identify potencies that occur less than 10 times
    potencies_to_remove = potency_counts[potency_counts < 10].index
    
    # Get the list of barcodes whose potency set should be removed
    barcodes_to_remove = df_potency[df_potency['potency_set'].isin(potencies_to_remove)].index
    
    # Filter the main cell dataframe
    df_cells_filtered = df_cells[~df_cells[barcode_col].isin(barcodes_to_remove)]
    
    num_removed = len(df_cells) - len(df_cells_filtered)
    num_barcodes_removed = len(barcodes_to_remove)
    total_barcodes_initial = df_cells[barcode_col].nunique()
    print(f"Removed {num_barcodes_removed} barcodes (with {num_removed} cells) occurring < 10 times.")
    print(f"{df_cells_filtered[barcode_col].nunique()} barcodes remain.")

    # --- Step 3: Generate .nwk and metadata for remaining barcodes ---
    
    print(f"Generating .nwk and metadata files in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Group the final filtered cells by barcode
    grouped_cells = df_cells_filtered.groupby(barcode_col)
    
    count = 0
    for barcode, group in grouped_cells:
        # Sanitize barcode to be used as filename
        safe_barcode_name = "".join(c if c.isalnum() else "_" for c in str(barcode))
        
        # List of cell IDs (leaves) for this barcode
        cell_ids = group[cell_id_col].tolist()
        
        # --- Create Newick String ---
        # Format: (leaf1,leaf2,...)root_name;
        if not cell_ids: 
            continue # Should not happen after filtering, but safety check
            
        leaf_string = ",".join(map(str, cell_ids))
        # Use barcode as the internal node name
        newick_string = f"({leaf_string}){safe_barcode_name};" 
        
        # --- Create Metadata Dictionary ---
        # Format: {cell_id: cell_type, ...}
        metadata_dict = pd.Series(group[cell_type_col].values, index=group[cell_id_col]).to_dict()
        
        # --- Save Files ---
        nwk_filename = os.path.join(output_dir, f"{safe_barcode_name}.nwk")
        meta_filename = os.path.join(output_dir, f"{safe_barcode_name}_meta.txt") # Simple text, one per line
        
        try:
            with open(nwk_filename, 'w') as f_nwk:
                f_nwk.write(newick_string)
                
            with open(meta_filename, 'w') as f_meta:
                for cell_id, cell_type in metadata_dict.items():
                    f_meta.write(f"{cell_id}\t{cell_type}\n") # Simple tab-separated format
                    
            count += 1
        except IOError as e:
            print(f"Warning: Could not write files for barcode {barcode}. Error: {e}", file=sys.stderr)
            
    print(f"Successfully generated {count} pairs of .nwk and metadata files.")


# --- HOW TO RUN ---

# 1. Download 'in_vitro_data_per_cell.csv.gz' from the AllonKleinLab repository:
#    https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation
#    Place it in the same directory as this script, or provide the full path.

# 2. Define the paths
RAW_DATA_FILE = 'in_vitro_data_per_cell.csv.gz' 
OUTPUT_DIRECTORY = 'larry_nwk_files'

# 3. Run the generation function
generate_nwk_and_metadata_for_larry(RAW_DATA_FILE, OUTPUT_DIRECTORY)

# 4. Check the OUTPUT_DIRECTORY. You will find thousands of pairs of files:
#    - BarcodeName1.nwk
#    - BarcodeName1_meta.txt
#    - BarcodeName2.nwk
#    - BarcodeName2_meta.txt
#    - ... etc.