import datasets
import re
from sklearn.metrics import f1_score


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a_th"] + " " + doc["ctx_b_th"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label_th"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings_th"]],
        }
        return out_doc

    return dataset.map(_process_doc)

# def macro_f1_score(items):
#     unzipped_list = list(zip(*items))
#     golds = unzipped_list[0]
#     preds = unzipped_list[1]
#     fscore = f1_score(golds, preds, average='macro')
#     return fscore