import os 
import re
import sys
delete = sys.argv[1] if len(sys.argv) > 1 else None
files = os.listdir()
score_list = []
for f in files:
    if f.startswith("ppo_connect"):
        
        # Extract reward value from filename
        match = re.search(r'best_([-\d.]+)\.zip$', f)
        if match:
            reward = float(match.group(1))
            score_list.append((reward, f))

# Find the best model (highest reward)
if score_list:
    best_reward, best_file = max(score_list)
    print(f"Keeping best model: {best_file} with reward: {best_reward}")
    
    # Remove all other files
    for reward, filename in score_list:
        if filename != best_file:
            if delete:
                os.remove(filename)
            print(f"Removed: {filename}")
