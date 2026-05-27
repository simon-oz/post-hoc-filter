from datasets import Dataset
from datasets import load_dataset

# Load the Arrow file
# dataset = load_dataset('ms_marco', 'v2.1')
# dataset.save_to_disk('/home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/msmarco')

# Convert to pandas if needed
dataset = Dataset.from_file('/home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/msmarco/test/data-00000-of-00001.arrow')
print('Available splits:', dataset.keys())
