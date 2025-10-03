# **Classification Model**

This model was fine-tuned for **IT service ticket classification** based on the **DistilBERT base uncased** model.

* **Original model:** [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
* **Architecture:** Distilled version of BERT base – smaller, faster, and lighter than BERT while retaining high accuracy.
* **Use case:** Text classification, especially short-to-medium English texts.
* **Advantages:**

  * Faster inference than full BERT
  * Smaller memory footprint
  * Pretrained bidirectional understanding of English
  * Suitable for fine-tuning on downstream NLP tasks like sequence classification

## Files inside the zip

| File Name                 | Description                                                                    |
| ------------------------- | ------------------------------------------------------------------------------ |
| `tokenizer_config.json`   | Configuration for the tokenizer (how text is split into tokens).               |
| `model.safetensors`       | Model weights in the `safetensors` format (more secure and fast to load).      |
| `special_tokens_map.json` | Mapping of special tokens (e.g., `[CLS]`, `[SEP]`, `[PAD]`) used by the model. |
| `config.json`             | Model configuration, including architecture parameters.                        |
| `vocab.txt`               | Vocabulary file used by the tokenizer for token-to-id mapping.                 |
| `training_args.bin`       | Training arguments used during fine-tuning.                                    |
| `tokenizer.json`          | Full tokenizer data in JSON format, compatible with Hugging Face Transformers. |

