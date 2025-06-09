import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import logging
import re

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.label_map = {0: "O", 1: "ACCOUNT", 2: "METER"}
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                return_offsets_mapping=True,
                                padding=True,
                                truncation=True)

        model_inputs = {k: v for k, v in inputs.items() if k != 'offset_mapping'}

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        return self._extract_entities(text, inputs['offset_mapping'][0], predictions[0])

    def _extract_entities(self, text, offsets, predictions):
        entities = []
        current_entity = None

        for (start, end), pred in zip(offsets, predictions):
            label = self.label_map[pred.item()]

            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            if not current_entity:
                current_entity = {
                    'type': label,
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                }
            else:
                current_entity['end'] = end
                current_entity['text'] += text[start:end]

        if current_entity:
            entities.append(current_entity)

        return self._postprocess_entities(entities)

    def _postprocess_entities(self, entities):
        results = []
        account_candidates = []
        meter_candidates = []

        for ent in entities:
            clean_text = re.sub(r'\D', '', ent["text"])
            if not clean_text:
                continue

            if ent["type"] == "ACCOUNT":
                account_candidates.append(clean_text)
            elif ent["type"] == "METER":
                meter_candidates.append(clean_text[:5])

        if account_candidates:
            results.append({
                "type": "ACCOUNT",
                "text": max(account_candidates, key=len)
            })

        if meter_candidates:
            results.append({
                "type": "METER",
                "text": meter_candidates[-1]
            })

        return results