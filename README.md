# Zero-shot-KPE-with-LLMs
This is code of our paper [Empirical Study of Zero-shot Keyphrase Extraction with LLMs](https://openreview.net/pdf?id=sJka8kOHfD).

## Requirements
- torch=2.2.2
- transformers==4.40.0
- nltk 3.7
- StanfordCoreNLP 3.9.1.1
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing)


## Run
```python
# Vanilla
python llama3_vanilla.py \
  --auth_token "your huggingface auth_token for Llama3"


# Candidate
python llama3_candidate.py \
  --core_nlp_path "your StanfordCoreNLP path"\
  --auth_token "your huggingface auth_token for Llama3"


# Hybrid
python llama3_hybrid.py \
  --core_nlp_path "your StanfordCoreNLP path"\
  --auth_token "your huggingface auth_token for Llama3"
```
