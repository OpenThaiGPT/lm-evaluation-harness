"""
SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://w4ngatang.github.io/static/papers/superglue.pdf

SuperGLUE is a benchmark styled after GLUE with a new set of more difficult language
understanding tasks.

Homepage: https://super.gluebenchmark.com/

TODO: WSC requires free-form generation.
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, acc_all, metric_max_over_ground_truths, yesno
from lm_eval.utils import general_detokenize


_CITATION = """
@inproceedings{NEURIPS2019_4496bf24,
    author = {Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
    url = {https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf},
    volume = {32},
    year = {2019}
}
"""


class Copa(Task):
    VERSION = 0
    DATASET_PATH = "Patt/copa_th"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # Drop the period
        connector = {
            "สาเหตุ": "สาเหตุ",
            "ผล": "ผล",
        }[doc["question_th"]]
        return doc["premise_th"].strip()[:-1] + f" {connector}"
    
    def doc_to_target(self, doc):
        correct_choice = doc["choice1_th"] if doc["label"] == 0 else doc["choice2_th"]
        # Connect the sentences
        return " " + self.convert_choice(correct_choice)
    
    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1_th"])
        choice2 = " " + self.convert_choice(doc["choice2_th"])

        ll_choice1, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _ = rf.loglikelihood(ctx, choice2)

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]



class MultiRC(Task):
    VERSION = 1
    DATASET_PATH = "Patt/MultiRC_TH_drop"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return f"{doc['paragraph_TH']}\nQuestion: {doc['question_TH']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + self.format_answer(answer=doc["answer_TH"], label=doc["label"])

    @staticmethod
    def format_answer(answer, label):
        label_str = "ใช่" if label else "ไม่ใช่"
        return f"{answer}\nใช่หรือไม่? {label_str}"

    def construct_requests(self, doc, ctx):
        true_choice = self.format_answer(answer=doc["answer_TH"], label=True)
        false_choice = self.format_answer(answer=doc["answer_TH"], label=False)

        ll_true_choice, _ = rf.loglikelihood(ctx, f" {true_choice}")
        ll_false_choice, _ = rf.loglikelihood(ctx, f" {false_choice}")

        return ll_true_choice, ll_false_choice

    def process_results(self, doc, results):
        ll_true_choice, ll_false_choice = results
        pred = ll_true_choice > ll_false_choice
        return {"acc": (pred, doc)}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": acc_all}


class ReCoRD(Task):
    VERSION = 0
    DATASET_PATH = "Patt/ReCoRD_TH_drop"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        # In ReCoRD, each doc manifests multiple "examples" in the context of few shot example packing.
        # Each doc consists of multiple answer candidates, each of which is scored yes/no.
        if self._training_docs is None:
            self._training_docs = []
            for doc in self.dataset["train"]:
                self._training_docs.append(self._process_doc(doc))
        return self._training_docs

    def validation_docs(self):
        # See: training_docs
        for doc in self.dataset["validation"]:
            yield self._process_doc(doc)

    @classmethod
    def _process_doc(cls, doc):
        return {
            "passage_TH": doc["passage_TH"],
            "query_TH": doc["query_TH"],
            "entities_TH": sorted(list(set(doc["entities_TH"]))),
            "answers_TH": sorted(list(set(doc["answers_TH"]))),
        }

    def doc_to_text(self, doc):
        initial_text, *highlights = doc["passage_TH"].strip().split("\n @highlight\n")
        text = initial_text + "\n\n"
        for highlight in highlights:
            text += f"  - {highlight}.\n"
        return text

    @classmethod
    def format_answer(cls, query, entity):
        return f"  - {query}".replace("@placeholder", entity)

    def doc_to_target(self, doc):
        # We only output the first correct entity in a doc
        return self.format_answer(query=doc["query_TH"], entity=doc["answers_TH"][0])

    def construct_requests(self, doc, ctx):
        requests = [
            rf.loglikelihood(ctx, self.format_answer(query=doc["query_TH"], entity=entity))
            for entity in doc["entities_TH"]
        ]
        return requests

    def process_results(self, doc, results):
        # ReCoRD's evaluation is actually deceptively simple:
        # - Pick the maximum likelihood prediction entity
        # - Evaluate the accuracy and token F1 PER EXAMPLE
        # - Average over all examples
        max_idx = np.argmax(np.array([result[0] for result in results]))

        prediction = doc["entities_TH"][max_idx]
        gold_label_set = doc["answers_TH"]
        f1 = metric_max_over_ground_truths(
            squad_metrics.compute_f1, prediction, gold_label_set
        )
        em = metric_max_over_ground_truths(
            squad_metrics.compute_exact, prediction, gold_label_set
        )

        return {
            "f1": f1,
            "em": em,
        }

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }
