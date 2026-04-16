import json
with open('frontend/messages/fr.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(list(d.get('AnalysisPanel', {}).keys()))
