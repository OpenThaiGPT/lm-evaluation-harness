"""
GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
https://openreview.net/pdf?id=rJ4km2R5t7

The General Language Understanding Evaluation (GLUE) benchmark is a collection of
resources for training, evaluating, and analyzing natural language understanding
systems. GLUE consists of:
- A benchmark of nine sentence- or sentence-pair language understanding tasks built
on established existing datasets and selected to cover a diverse range of dataset
sizes, text genres, and degrees of difficulty, and
- A diagnostic dataset designed to evaluate and analyze model performance with
respect to a wide range of linguistic phenomena found in natural language.

Homepage: https://gluebenchmark.com/
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize


# TODO(jon-tow): Add citations for the individual datasets/tasks that make up GLUE.
_CITATION = """
@inproceedings{wang-etal-2018-glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex  and
      Singh, Amanpreet  and
      Michael, Julian  and
      Hill, Felix  and
      Levy, Omer  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5446",
    doi = "10.18653/v1/W18-5446",
    pages = "353--355",
    abstract = "Human ability to understand language is \textit{general, flexible, and robust}. In contrast, most NLU models above the word level are designed for a specific task and struggle with out-of-domain data. If we aspire to develop models with understanding beyond the detection of superficial correspondences between inputs and outputs, then it is critical to develop a unified model that can execute a range of linguistic tasks across different domains. To facilitate research in this direction, we present the General Language Understanding Evaluation (GLUE, gluebenchmark.com): a benchmark of nine diverse NLU tasks, an auxiliary dataset for probing models for understanding of specific linguistic phenomena, and an online platform for evaluating and comparing models. For some benchmark tasks, training data is plentiful, but for others it is limited or does not match the genre of the test set. GLUE thus favors models that can represent linguistic knowledge in a way that facilitates sample-efficient learning and effective knowledge-transfer across tasks. While none of the datasets in GLUE were created from scratch for the benchmark, four of them feature privately-held test data, which is used to ensure that the benchmark is used fairly. We evaluate baselines that use ELMo (Peters et al., 2018), a powerful transfer learning technique, as well as state-of-the-art sentence representation models. The best models still achieve fairly low absolute scores. Analysis with our diagnostic dataset yields similarly weak performance over all phenomena tested, with some exceptions.",
}
"""

# Inference Tasks

class RTE(Task):
    VERSION = 0
    DATASET_PATH = "Patt/RTE_TH_drop"
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
        return "{}\nQuestion: {} True or False?\nAnswer:".format(
            doc["premise_TH"],
            doc["hypothesis_TH"],
        )

    def doc_to_target(self, doc):
        # 0 = entailment
        # 1 = not_entailment
        return " {}".format({0: "True", 1: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_false > ll_true
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
