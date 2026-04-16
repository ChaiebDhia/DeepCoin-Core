import json
with open('frontend/messages/fr.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(json.dumps(d.get('AgentPipeline'), indent=2))
