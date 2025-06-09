from typing import List, Dict

def validate_entities(entities: List[Dict]) -> bool:
    required = {'ACCOUNT', 'METER'}
    present = {e['type'] for e in entities}
    return required.issubset(present)

def validate_account_number(number: str) -> bool:
    return len(number) == 9 and number.isdigit()