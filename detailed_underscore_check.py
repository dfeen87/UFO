import re

with open("UFO.md") as f:
    lines = f.readlines()

in_code_block = False

for line_no, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith("```"):
        in_code_block = not in_code_block
        continue
    if in_code_block:
        continue

    # Find all underscores in line
    if "_" in line:
        # Check if line has math or inline code
        # Let's see every line containing _
        print(f"L{line_no}: {line.strip()}")
