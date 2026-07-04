source = "val_frame_counts_12Hz.txt"
target = "/path/to/your/frame_counts.txt"
output = "val_frame_counts_from_loader_12Hz.txt"

import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached.")

source_tokens = []
source_counts = []
with open(source, "r") as f:
    source_lines = f.readlines()
    source_lines = [tuple(line.strip().split(',')) for line in source_lines]
    source_tokens = [line[0] for line in source_lines]
    source_counts = [int(line[1]) for line in source_lines]
target_tokens = []
with open(target, "r") as f:
    target_lines = f.readlines()
    target_lines = [tuple(line.strip().split(',')) for line in target_lines]
    target_tokens = [line[0] for line in target_lines]
    
# sort source tokens using target token order
sorted_counts = []
for tgt_token in target_tokens:
    if tgt_token in source_tokens:
        idx = source_tokens.index(tgt_token)
        sorted_counts.append(source_counts[idx])
    else:
        raise ValueError(f"Token {tgt_token} not found in source tokens")

# write to output file
with open(output, "w") as f:
    for token, count in zip(target_tokens, sorted_counts):
        f.write(f"{token}, {count}\n")
