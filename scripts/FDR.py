# Run FDR on output of type_encoding= "correlations"
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


patient = 777
# elec_name = "717_LGA18"#"717_LGB74"#"743_G45"#"717_LGB79"#"717_LGB121" #13
mode = "comp"

model = "glove50"
layer = 0#13
context = 1#32

filter_type = "160" #all, 50, 160

p_threshold = 0.05

recording_type = "tfs"
if patient == 777:
    recording_type = "podcast"

###########################################################

path_template = f"../results/{recording_type}/tk-{recording_type}-{patient}-{model}-lag2k-25-all/tk-200ms-{patient}-lay{layer}-con{context}-corr_coeffs"

filter_csv_path = None
if filter_type == "160":
    filter_csv_path = "/scratch/gpfs/tk6637/princeton/247-plotting/data/plotting/sig-elecs/podcast_160.csv"
elif filter_type == "50":
    filter_csv_path = "/scratch/gpfs/tk6637/princeton/247-plotting/data/plotting/sig-elecs/podcast_777_glove_elecs.csv"

if filter_csv_path:
    filter_df = pd.read_csv(filter_csv_path)
    filter_df["concat"] = filter_df["subject"].astype(str) + "_" + filter_df["electrode"]


corrs_all_elecs = []
pvals_all_elecs = []
cis_all_elecs = []
pvals_elecs_names = []

for filename in sorted(os.listdir(path_template)):
    pattern = r'^(.*?)_(.*?)(?=_(?:comp|prod))'
    compiled_pattern = re.compile(pattern)
    match = compiled_pattern.match(filename)
    if match:
        sid = match.group(1)
        elec_name = match.group(2)
    elif filename == "config.yml" or filename == "summary.csv" or filename.startswith("pvals_combined"):
        continue
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern")

    if filter_csv_path:
        if not filter_df['concat'].isin([f'{sid}_{elec_name}']).any(): # f'{sid}_{elec_name}' in filter_df['concat'].values
            print(f"Skipping {filename} as it is not in the filter list")
            continue

#     if filename.endswith("_corr.npy"):
#         corrs_all_elecs.append(np.load(os.path.join(path_template, filename)))
    if filename.endswith("_pval.npy"):
        pvals_all_elecs.append(np.load(os.path.join(path_template, filename)))
        pvals_elecs_names.append(f"{sid}_{elec_name}")
#     elif filename.endswith("_ci.npy"):
#         cis_all_elecs.append(np.load(os.path.join(path_template, filename)))
#     else:
#         print(f"File {filename} not corr, pval or ci")

print("#" * 50)
print("Starting FDR correction for p-values")
pvals_all_elecs_np = np.stack(pvals_all_elecs, axis=2)
not_nan_row_indices = np.where(~np.isnan(pvals_all_elecs_np).any(axis=(1,2)))[0]

p_flat = pvals_all_elecs_np[not_nan_row_indices].flatten()
rejected, p_corrected_flat, _, _ = multipletests(p_flat, method='fdr_bh')
p_corrected = p_corrected_flat.reshape(*pvals_all_elecs_np[not_nan_row_indices].shape)

full_p_corrected = np.full(pvals_all_elecs_np.shape, np.nan)
full_p_corrected[not_nan_row_indices, :, :] = p_corrected

# Save the results
print(f"#" * 50)
print(f"Saving results to {path_template} with filter type {filter_type}")
with open(os.path.join(path_template, f"pvals_combined_names{f'({filter_type})' if filter_type else ''}.pkl"), 'wb') as f:
    pickle.dump(pvals_elecs_names, f)
np.save(os.path.join(path_template, f"pvals_combined{f'({filter_type})' if filter_type else ''}.npy"), full_p_corrected)
np.save(os.path.join(path_template, f"pvals_combined_corrected{f'({filter_type})' if filter_type else ''}.npy"), full_p_corrected)
full_p_corrected

###############################################################################################################

#
# import os
# import shutil
# import glob
#
# # Define source and destination directories
# source_dir = "/scratch/gpfs/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-gemma-scope-2b-pt-res-canonical-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-alphas_-2_30_40"
# dest_dir = "/scratch/gpfs/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-gemma-scope-2b-pt-res-canonical-lag2k-25-all/tk-200ms-777-lay13-con32-corr_coeffs"
#
# # Create destination directory if it doesn't exist
# # os.makedirs(dest_dir, exist_ok=True)
#
# # Find all files matching the pattern
# pattern = os.path.join(source_dir, "*comp_pval.npy")
# files_to_move = glob.glob(pattern)
#
# print(f"Found {len(files_to_move)} files matching the pattern:")
# for file in files_to_move:
#     print(f"  {os.path.basename(file)}")
#
# # Move each file
# moved_count = 0
# for file_path in files_to_move:
#     try:
#         filename = os.path.basename(file_path)
#         dest_path = os.path.join(dest_dir, filename)
#
#         # Move the file
#         shutil.move(file_path, dest_path)
#         print(f"Moved: {filename}")
#         moved_count += 1
#
#     except Exception as e:
#         print(f"Error moving {filename}: {e}")
#
# print(f"\nSuccessfully moved {moved_count} files to {dest_dir}")

###############################################################################################################

# import csv
# import os
#
# # Write list to CSV file
# with open("/scratch/gpfs/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-gemma-scope-2b-pt-res-canonical-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-alphas_-2_30_40-sig_coeffs/summary.csv", 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#
#     # Write each item as a row with format: 777,<item>,0,4997
#     for item in counts.keys():
#         writer.writerow([item.split('_')[0], item, 0, 4997])
