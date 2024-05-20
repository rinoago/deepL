we use the dataset:
'ucberkeley-dlab/measuring-hate-speech'

in ./data:
- original_data.jsonl : balanced dataset of 2 classes (38186 per class)
    - label 1: hate speech
    - label 0: not hate speech
- paraphrased_data.jsonl : paraphrasing of the original data (2 paraphrases per original sentence)
- augmented_data.jsonl : original_data.jsonl + paraphrased_data.jsonl

in ./data/txt:
- data_hate.txt : hate speech data from the augmented_data.jsonl
- data_nohate.txt : not hate speech data from the augmented_data.jsonl
- data.txt : hate speech + not hate speech data from the augmented_data.jsonl 
- splits of data_nohate and data_hate with 80% train and 20% test (train size: 30548, test size: 7638):
    - train_data_nohate.txt, test_data_nohate.txt
    - train_data_hate.txt, test_data_hate.txt