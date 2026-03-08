from datasets import load_dataset
import random

# 1. Load your dataset (change this to your dataset + split)
dataset = load_dataset("nvidia/OpenMathReasoning")['cot']

# 2. Set how many samples you want
n_samples = 10000

# 3. Sample indices from the original dataset
random.seed(0)  # for reproducibility
n = len(dataset)
sample_size = min(n_samples, n)  # just in case dataset < 500
indices = random.sample(range(n), k=sample_size)

# 4. Select those rows and attach the original index as a column
subset = dataset.select(indices)
subset = subset.add_column("orig_index", indices)

# 5. Now `subset` is a HuggingFace Dataset with all original fields + `orig_index`
print(subset[0])
