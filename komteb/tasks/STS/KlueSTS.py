import datasets
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS

class KlueSTS(AbsTaskSTS):
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # hf_part is the name of load_dataset
        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], self.description["hf_part"], revision=self.description.get("revision", None) 
        )
        # convert the columns
        self.dataset = self.dataset.map(lambda doc: {"sentence1":doc["sentence1"], "sentence2":doc["sentence2"], 
                                                     "score":doc['labels']['real-label']}, remove_columns=['guid', 'source','labels'])
        self.data_loaded = True
        
    @property
    def description(self):
        return {
            "name": "KlueSTS",
            "hf_hub_name": "klue",
            "hf_part": "sts", # name for load dataset arguments
            "description": "The text describes Semantic Textual Similarity (STS) as a measure of semantic equivalence between sentences, \
                            its importance in NLP tasks, the formulation of STS as a regression task, \
                            and the evaluation of model performance using Pearson's correlation coefficient and F1 score.",
            "reference": "https://klue-benchmark.com/tasks/67/overview/description",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation"],
            "eval_langs": ["ko"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5
        }