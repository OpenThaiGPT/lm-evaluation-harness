"""
MMLU (Massive Multitask Language Understanding) 
is a new benchmark designed to measure knowledge acquired during pretraining 
by evaluating models exclusively in zero-shot and few-shot settings. This makes 
the benchmark more challenging and more similar to how we evaluate humans. 
The benchmark covers 57 subjects across STEM, the humanities, the social sciences, 
and more. It ranges in difficulty from an elementary level to an advanced 
professional level, and it tests both world knowledge and problem solving ability. 
Subjects range from traditional areas, such as mathematics and history, 
to more specialized areas like law and ethics. The granularity and breadth of 
the subjects makes the benchmark ideal for identifying a model's blind spots.

huggingface: https://huggingface.co/datasets/cais/mmlu
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
"""


class MMLUTask(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                map(self._process_doc, self.dataset["auxiliary_train"])
            )
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "Question: " + doc["question"] + "\nAnswer:",
            "choices": doc["choices"],
            "gold": doc["answer"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class MMLUAbstractAlgebra(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "abstract_algebra"


class MMLUAnatomy(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "anatomy"


class MMLUAstronomy(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "astronomy"


class MMLUBusinessEthics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "business_ethics"


class MMLUClinicalKnowledge(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "clinical_knowledge"


class MMLUCollegeBiology(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "college_biology"


class MMLUCollegeChemistry(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "college_chemistry"


class MMLUCollegeComputerScience(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "college_computer_science"


class MMLUCollegeMathematics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "college_mathematics"


class MMLUCollegeMedicine(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "college_medicine"


class MMLUCollegePhysics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "college_physics"


class MMLUComputerSecurity(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "computer_security"


class MMLUConceptualPhysics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "conceptual_physics"


class MMLUEconometrics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "econometrics"


class MMLUElectricalEngineering(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "electrical_engineering"


class MMLUElementaryMathematics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "elementary_mathematics"


class MMLUFormalLogic(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "formal_logic"


class MMLUGlobalFacts(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "global_facts"


class MMLUHighSchoolBiology(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_biology"


class MMLUHighSchoolChemistry(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_chemistry"


class MMLUHighSchoolComputerScience(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_computer_science"


class MMLUHighSchoolEuropeanHistory(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_european_history"


class MMLUHighSchoolGeography(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_geography"


class MMLUHighSchoolGovernmentAndPolitics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_government_and_politics"


class MMLUHighSchoolMacroeconomics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_macroeconomics"


class MMLUHighSchoolMathematics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_mathematics"


class MMLUHighSchoolMicroeconomics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_microeconomics"


class MMLUHighSchoolPhysics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_physics"


class MMLUHighSchoolPsychology(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_psychology"


class MMLUHighSchoolStatistics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_statistics"


class MMLUHighSchoolUsHistory(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_us_history"


class MMLUHighSchoolWorldHistory(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "high_school_world_history"


class MMLUHumanAging(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "human_aging"


class MMLUHumanSexuality(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "human_sexuality"


class MMLUInternationalLaw(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "international_law"


class MMLUJurisprudence(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "jurisprudence"


class MMLULogicalFallacies(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "logical_fallacies"


class MMLUMachineLearning(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "machine_learning"


class MMLUManagement(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "management"


class MMLUMarketing(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "marketing"


class MMLUMedicalGenetics(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "medical_genetics"


class MMLUMiscellaneous(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "miscellaneous"


class MMLUMoralDisputes(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "moral_disputes"


class MMLUMoralScenarios(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "moral_scenarios"


class MMLUNutrition(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "nutrition"


class MMLUPhilosophy(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "philosophy"


class MMLUPrehistory(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "prehistory"


class MMLUProfessionalAccounting(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "professional_accounting"


class MMLUProfessionalLaw(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "professional_law"


class MMLUProfessionalMedicine(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "professional_medicine"


class MMLUProfessionalPsychology(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "professional_psychology"


class MMLUPublicRelations(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "public_relations"


class MMLUSecurityStudies(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "security_studies"


class MMLUSociology(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "sociology"


class MMLUUsForeignPolicy(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "us_foreign_policy"


class MMLUVirology(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "virology"


class MMLUWorldReligions(MMLUTask):
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = "world_religions"
