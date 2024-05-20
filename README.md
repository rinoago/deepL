we use the dataset:
'ucberkeley-dlab/measuring-hate-speech'

in ./data:
- original_data.jsonl : balanced dataset of 2 classes 
    - label 1: hate speech
    - label 0: not hate speech
- paraphrased_data.jsonl : paraphrasing of the original data (2 paraphrases per original sentence)
- augmented_data.jsonl : original_data.jsonl + paraphrased_data.jsonl

- data_hate.txt : hate speech data from the augmented_data.jsonl
- data_nohate.txt : not hate speech data from the augmented_data.jsonl
- data.txt : hate speech + not hate speech data from the augmented_data.jsonl