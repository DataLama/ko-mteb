import datasets
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification

class KlueYnat(AbsTaskClassification):
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
        label_feature = self.dataset['train'].features['label']
        # convert the columns
        self.dataset = self.dataset.map(lambda doc: {"text":doc["title"], "label":doc["label"], 
                                                     "label_text":label_feature.int2str(doc["label"])}, remove_columns=['guid', 'title', 'url', 'date'])
        self.data_loaded = True

    @property
    def description(self):
        return {
            "name": "KlueYnat",
            "hf_hub_name": "klue",
            "hf_part": "ynat", # name for load dataset arguments
            "description": "The text discusses topic classification (TC) as the prediction of topics in text snippets, \
                the inclusion of TC in the KLUE benchmark, the introduction of YNAT as a dataset for Korean TC, \
                and the evaluation metric of Macro-F1 score used for assessing system performance.",
            "reference": "https://klue-benchmark.com/tasks/66/overview/description",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["validation"],
            "eval_langs": ['ko'],
            "main_score": "accuracy"
        }