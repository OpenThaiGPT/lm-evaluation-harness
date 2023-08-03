"""
SCROLLS: Standardized CompaRison Over Long Language Sequences
https://arxiv.org/abs/2201.03533

SCROLLS is a suite of datasets that require synthesizing information over long texts.
The benchmark includes seven natural language tasks across multiple domains,
including summarization, question answering, and natural language inference.

Homepage: https://www.scrolls-benchmark.com/

Since SCROLLS tasks are generally longer than the maximum sequence length of many models,
it is possible to create "subset" tasks that contain only those samples whose tokenized length
is less than some pre-defined limit. For example, to create a subset of "Qasper" that would
be suitable for a model using the GPTNeoX tokenizer and a 4K maximium sequence length:

```
class QasperGPTNeoX4K(Qasper):
    PRUNE_TOKENIZERS = ["EleutherAI/pythia-410m-deduped"]
    PRUNE_MAX_TOKENS = 4096
    PRUNE_NUM_PROC = _num_cpu_cores() # optional, to speed up pruning of large datasets like NarrativeQA
```

`PRUNE_TOKENIZERS` can contain more than one tokenizer; this will include only samples that are
less than `PRUNE_MAX_TOKENS` for ALL of the tokenizers. This can be useful to comparing models
that use different tokenizers but the same maximum sequence length.

Once the subset task class has been defined in this file, it can be used by adding the class
to `lm_eval/tasks/__init__.py`.

NOTE: GovReport may need `max_gen_toks` set larger for causal models.
"""
from abc import abstractmethod
from datasets import load_metric
from transformers import AutoTokenizer
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from functools import reduce
import transformers.data.metrics.squad_metrics as squad_metrics
import numpy as np
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

_CITATION = """
@inproceedings{shaham-etal-2022-scrolls,
    title = "{SCROLLS}: Standardized {C}ompa{R}ison Over Long Language Sequences",
    author = "Shaham, Uri  and
      Segal, Elad  and
      Ivgi, Maor  and
      Efrat, Avia  and
      Yoran, Ori  and
      Haviv, Adi  and
      Gupta, Ankit  and
      Xiong, Wenhan  and
      Geva, Mor  and
      Berant, Jonathan  and
      Levy, Omer",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.823",
    pages = "12007--12021"
}
"""

# SCROLLS is formualted as a sequence-to-sequence task.
# To allow for evaluation of causal models, we'll
# reformualte these with appropriate prompts


def _download_metric():
    import os
    import shutil
    from huggingface_hub import hf_hub_download
    scrolls_metric_path = hf_hub_download(repo_id="tau/scrolls", repo_type="dataset", filename="metrics/scrolls.py")
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path


def _process_doc_prepended_question(doc):
    # "When a query is given in addition to the raw text (as
    # in QMSum, Qasper, NarrativeQA, QuALITY, and ContractNLI),
    # we prepend it to the text, using two newlines as a natural separator"

    return {
        "body": doc["body"],
        "summary": doc["summary"],
        "question": "บทสรุปของข้อความก่อนหน้าคืออะไร?"
    }


# def _drop_duplicates_in_input(untokenized_dataset):
#     # from scrolls/evaluator/dataset_evaluator.py

#     indices_to_keep = []
#     id_to_idx = {}
#     outputs = []
#     for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
#         if id_ in id_to_idx:
#             outputs[id_to_idx[id_]].append(output)
#             continue
#         indices_to_keep.append(i)
#         id_to_idx[id_] = len(outputs)
#         outputs.append([output])
#     untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
#     untokenized_dataset = untokenized_dataset.remove_columns("output")
#     untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
#     return untokenized_dataset


def _num_cpu_cores():
    # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        import os
        return len(os.sched_getaffinity(0))

class _SCROLLSTaskThai(Task):
    VERSION = 0
    DATASET_PATH = "thaisum"
    DATASET_NAME = None
    PRUNE_TOKENIZERS = None
    PRUNE_MAX_TOKENS = None
    PRUNE_NUM_PROC = None

    def __init__(self, no_metric=False):
        super().__init__()
        self.metric = load_metric(_download_metric(), config_name="gov_report") if not no_metric else None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        for doc in self.dataset["train"]:
            yield from self._process_doc(doc)

    def validation_docs(self):
        for doc in self.dataset["validation"]:
            yield from self._process_doc(doc)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["body"]

    # def download(self, *args, **kwargs):
    #     super().download(*args, **kwargs)
    #     del self.dataset["test"]
    #     for split in self.dataset:
    #         self.dataset[split] = _drop_duplicates_in_input(self.dataset[split])
    #     if self.PRUNE_TOKENIZERS is not None and self.PRUNE_TOKENIZERS is not None:
    #         self.prune()

    def _get_prune_text(self, sample):
        return self.doc_to_text(self._process_doc(sample)[0])


    def doc_to_target(self, doc):
       return doc["summary"]
       #return " " + ", ".join(doc["summary"])

    def doc_to_text(self, doc):
        return f"{doc['body']}\n\nคำถาม: {doc['question']}\nคำตอบ:"

    def higher_is_better(self):
        print("_scrolls_metrics ==== >",self._scrolls_metrics())
        return {x: True for x in self._scrolls_metrics().keys()}

    @abstractmethod
    def _scrolls_metrics(self):
        pass

    def _make_compute_metrics(self, value):
        def compute_metrics(samples):

            predictions, references = zip(*samples)  # unzip, if you will
            # print("predictions = ",list(predictions))
            print("predictions = ",predictions)
            print("references = ",references)
            # print("references = ",type(references))
            # print("references = ",list(references))
            # predictions = list(predictions)
            # references = list(references)
            # print("references = ",type(references))
            computed = self.metric.compute(predictions=[predictions], references=[references])
            return computed[value]
        return compute_metrics

    def aggregation(self):
        return {
            key: self._make_compute_metrics(value) for key, value in self._scrolls_metrics().items()
        }


class _SCROLLSSummaryTaskThai(_SCROLLSTaskThai):

    def _process_doc(self, doc):
        return [doc]

    def _scrolls_metrics(self):
        return {"rouge1": "rouge/rouge1", "rouge2": "rouge/rouge2", "rougeL": "rouge/rougeL"}

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["summary"]),
            "rouge2": (results[0], doc["summary"]),
            "rougeL": (results[0], doc["summary"])
        }

    def construct_requests(self, doc, ctx):
        return [rf.greedy_until(ctx, {'until': ["\n"]})]

    # def doc_to_text(self, doc):
    #     return f"{doc['body']}\n\nคำถาม: บทสรุปของข้อความก่อนหน้าคืออะไร?\nคำตอบ:"



class ThaiSum(_SCROLLSSummaryTaskThai):
    """QMSum: A New Benchmark for Query-based Multi-domain
    Meeting Summarization

    https://arxiv.org/abs/2104.05938
    """

    DATASET_NAME = "thaisum"

    def _process_doc(self, doc):
        return [_process_doc_prepended_question(doc)]

    # def doc_to_text(self, doc):
    #     return f"{doc['body']}\n\nคำถาม: {doc['question']}\nAnswer:"


def construct_tasks():
    return {
        "thaisum": ThaiSum,
    }
