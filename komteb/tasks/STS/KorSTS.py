import datasets
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS

class KorSTS(AbsTaskSTS):
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
                                                     "score":doc['score']}, remove_columns=['genre', 'filename','year', 'id'])
        self.data_loaded = True
        
    @property
    def description(self):
        return {
            "name": "KorSTS",
            "hf_hub_name": "kor_nlu",
            "hf_part": "sts", # name for load dataset arguments
            "description": "KorSTS dataset stems from the STS-B dataset.",
            "reference": "https://github.com/kakaobrain/KorNLUDatasets",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation"],
            "eval_langs": ["ko"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5
        }