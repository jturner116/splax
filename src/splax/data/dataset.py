import numpy as np


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        queries = [item["query"] for item in batch]
        positives = [item["pos"][0] for item in batch]
        negatives = [item["neg"][0] for item in batch]

        # Tokenize queries
        query_encodings = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # Return NumPy arrays
        )

        # Tokenize positive and negative documents
        all_docs = positives + negatives
        doc_encodings = self.tokenizer(
            all_docs,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Split doc encodings back into positives and negatives
        pos_doc_encodings = {
            key: val[: len(batch)] for key, val in doc_encodings.items()
        }
        neg_doc_encodings = {
            key: val[len(batch) :] for key, val in doc_encodings.items()
        }

        # Combine positive and negative documents
        # Shape: [batch_size, 2, seq_length]
        doc_input_ids = np.stack(
            [
                pos_doc_encodings["input_ids"],
                neg_doc_encodings["input_ids"],
            ],
            axis=1,
        )
        doc_attention_mask = np.stack(
            [
                pos_doc_encodings["attention_mask"],
                neg_doc_encodings["attention_mask"],
            ],
            axis=1,
        )

        batch = {
            "query_input_ids": np.asarray(query_encodings["input_ids"]),
            "query_attention_mask": np.asarray(query_encodings["attention_mask"]),
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
        }

        return batch
