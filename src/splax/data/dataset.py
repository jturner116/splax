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


class MultipleNegativesCollateFn:
    def __init__(self, tokenizer, num_negatives: int = 2):
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives

    def __call__(self, batch):
        queries = [item["query"] for item in batch]
        positives = [item["pos"][0] for item in batch]

        negatives = []
        for item in batch:
            item_negs = item["neg"][: self.num_negatives]
            # If not enough negatives, pad with copies of the first negative
            if len(item_negs) < self.num_negatives:
                padding_needed = self.num_negatives - len(item_negs)
                item_negs = item_negs + [item_negs[0]] * padding_needed
            negatives.append(item_negs)

        # Tokenize queries
        query_encodings = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # Return NumPy arrays
        )

        # Tokenize positive and negative documents
        pos_encodings = self.tokenizer(
            positives,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        neg_encodings = self.tokenizer(
            [neg for neg_list in negatives for neg in neg_list],
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        pos_doc_encodings = {
            key: val[:, np.newaxis, :] for key, val in pos_encodings.items()
        }

        neg_doc_encodings = {
            key: val.reshape(len(batch), self.num_negatives, -1)
            for key, val in neg_encodings.items()
        }

        # Combine positive and negative documents
        # Shape: [batch_size, 2, seq_length]
        doc_input_ids = np.concatenate(
            [
                pos_doc_encodings["input_ids"],
                neg_doc_encodings["input_ids"],
            ],
            axis=1,
        )
        doc_attention_mask = np.concatenate(
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
