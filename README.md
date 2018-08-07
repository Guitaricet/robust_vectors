Implementation of robust to typos and small errors word2vec.

The idea is to train vector representation of a word from letters to word2vec by RNN.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --data-dir data/brown --save-dir save/brown --model sru
```
Where data/brown should contain input.txt (plaintext file) and (optionally) valid.txt for validation checks during training
(it structure depends on --type parameter).
