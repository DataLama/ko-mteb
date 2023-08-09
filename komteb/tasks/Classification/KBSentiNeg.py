import datasets
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification

class KBSentiNeg(AbsTaskClassification):
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
        self.dataset = self.dataset.map(lambda doc: {"text":doc["sentence"], "label":doc["label"], 
                                                     "label_text":label_feature.int2str(doc["label"])}, remove_columns=['sentence'])
        self.data_loaded = True

    @property
    def description(self):
        return {
            "name": "KBSentiNeg",
            "hf_hub_name": "skt/kobest_v1",
            "hf_part": "sentineg", # name for load dataset arguments
            "description": "Predict the polarity of a negated sentence.",
            "reference": "https://arxiv.org/pdf/2204.04541.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ['ko'],
            "main_score": "accuracy",
        }