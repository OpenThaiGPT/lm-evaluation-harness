"""
Social media messages in Thai language with sentiment label (positive, neutral, negative, question). 
Released to public domain under Creative Commons Zero v1.0 Universal license.

Last update: 2019-03-31

huggingface: https://huggingface.co/datasets/wisesight_sentiment
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@software{bact_2019_3457447,
  author       = {Suriyawongkul, Arthit and
                  Chuangsuwanich, Ekapol and
                  Chormai, Pattarawat and
                  Polpanumas, Charin},
  title        = {PyThaiNLP/wisesight-sentiment: First release},
  month        = sep,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.3457447},
  url          = {https://doi.org/10.5281/zenodo.3457447}
}
"""


class WisesightSentiment(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "wisesight_sentiment"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "What is the sentiment of this sentence: "
            + doc["texts"]
            + "\nAnswer:",
            "choices": ["Positive", "Neutral", "Negative"],
            "gold": doc["category"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
