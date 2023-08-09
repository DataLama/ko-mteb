import datasets
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification

class KBBoolQ(AbsTaskPairClassification):
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
        new_ds = dict()
        for split in ["validation", "test"]:
            dict_of_list = {'sent1':self.dataset[split]['paragraph'], 'sent2':self.dataset[split]['question'], 'labels':self.dataset[split]['label']}
            new_ds[split] = datasets.Dataset.from_list([dict_of_list])

        self.dataset = datasets.DatasetDict(new_ds)
        self.data_loaded = True

    @property
    def description(self):
        return {
            "name": "KBBoolQ",
            "hf_hub_name": "skt/kobest_v1",
            "hf_part": "boolq", # name for load dataset arguments
            "description": "Identify whether a given question is true or false considering a paragraph.",
            "reference": "https://arxiv.org/pdf/2204.04541.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ['ko'],
            "main_score": "ap",
        }