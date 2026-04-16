import json

with open('frontend/messages/fr.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data['AnalysisPanel'].update({
    'consistentMatch': 'Correspondance Cohérente',
    'whatDoesThisMean': 'Qu\'est-ce que cela signifie ?',
    'top5Predictions': 'Top-5 prédictions',
    'cnType': 'Type CN',
    'materialCheck': 'Vérification du Matériau',
    'deepInvestigationMode': '🔍 Mode Investigation Approfondie',
    'investigationExplanation': 'Le classificateur visuel a renvoyé un signal faible. Le système a donc activé son pipeline le plus puissant : l\'Agent d\'Investigation a analysé les attributs visuels de la pièce et les a croisés avec les 9 541 types de la base de connaissances du Corpus Nummorum.',
    'tip': '💡 Astuce :',
    'obverseViews': 'Le modèle de classification a été entraîné sur les avers. Si vous possédez ce côté de la pièce, son téléchargement pourrait améliorer considérablement le score de confiance.',
    'feedbackTitle': 'Signaler un résultat incorrect',
    'feedbackDescription': 'Aidez à améliorer DeepCoin AI. Dites-nous ce qui n\'a pas fonctionné avec cette analyse.',
    'feedbackExpectedLabel': 'Type Attendue ou Description Courte',
    'feedbackExpectedPlaceholder': 'ex: Drachme d\'Alexandre le Grand',
    'feedbackReasonLabel': 'Raison',
    'feedbackReasonPlaceholder': 'ex: Le modèle l\'a confondu avec une pièce de bronze',
    'feedbackCancel': 'Annuler',
    'feedbackSubmit': 'Soumettre le Retour',
    'feedbackSuccess': 'Merci pour votre retour !',
    'reportIssue': 'Signaler Problème'
})

data['AgentPipeline'] = data.get('AgentPipeline', {})
data['AgentPipeline'].update({
    'processingCnnAgents': 'Traitement — CNN + Agents…'
})

data['CoinUploader'] = data.get('CoinUploader', {})
data['CoinUploader'].update({
    'browseFiles': 'parcourir',
    'tip1': 'Placez la pièce sur un fond uni.',
    'tip2': 'Prenez la photo de bien au-dessus, en évitant les reflets.',
    'tip3': 'L\'avers (avec le portrait principal) donne typiquement les meilleurs résultats.',
    'screenshotWarningTitle': 'Capture d\'écran Possible Détectée',
    'screenshotWarningDesc': 'Cette image semble être une capture d\'écran informatique plutôt qu\'une photographie numérisée. Les artefacts de compression et les arrière-plans numériques peuvent réduire considérablement la précision du CNN (souvent autour de 10-30%). Pour de meilleurs résultats, téléchargez la photographie d\'origine si possible.'
})

with open('frontend/messages/fr.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

with open('frontend/messages/en.json', 'r', encoding='utf-8') as f:
    data_en = json.load(f)

data_en['AnalysisPanel'].update({
    'consistentMatch': 'Consistent Match',
    'whatDoesThisMean': 'What does this mean?',
    'top5Predictions': 'Top-5 predictions',
    'cnType': 'CN Type',
    'materialCheck': 'Material Check',
    'deepInvestigationMode': '🔍 Deep Investigation Mode',
    'investigationExplanation': 'The visual classifier returned a low signal, so the system activated its most powerful pipeline: the Investigation Agent has analysed the coin\'s visual attributes and cross-referenced all 9,541 types in the Corpus Nummorum knowledge base.',
    'tip': '💡 Tip:',
    'obverseViews': 'The classification model was trained on obverse views (portrait or main inscription side). If you have that side of the coin, re-uploading it may significantly improve the confidence score.',
    'feedbackTitle': 'Report Incorrect Result',
    'feedbackDescription': 'Help improve DeepCoin AI. Let us know what went wrong with this analysis.',
    'feedbackExpectedLabel': 'Expected Type or Short Description',
    'feedbackExpectedPlaceholder': 'e.g. Drachm of Alexander the Great',
    'feedbackReasonLabel': 'Reason / What was missed?',
    'feedbackReasonPlaceholder': 'e.g. Model confused it with bronze / The portrait is clearly Augustus',
    'feedbackCancel': 'Cancel',
    'feedbackSubmit': 'Submit Feedback',
    'feedbackSuccess': 'Feedback submitted successfully!',
    'reportIssue': 'Report Issue'
})

data_en['AgentPipeline'] = data_en.get('AgentPipeline', {})
data_en['AgentPipeline'].update({
    'processingCnnAgents': 'Processing — CNN + Agents…'
})

data_en['CoinUploader'] = data_en.get('CoinUploader', {})
data_en['CoinUploader'].update({
    'browseFiles': 'browse files',
    'tip1': 'Place the coin on a plain background.',
    'tip2': 'Shoot from directly above, avoiding bright glare.',
    'tip3': 'The obverse (main portrait side) typically yields the best results.',
    'screenshotWarningTitle': 'Possible Screenshot Detected',
    'screenshotWarningDesc': 'This image appears to be a computer screenshot rather than a scanned photograph. Compression artifacts and digital backgrounds can severely reduce the CNN\'s accuracy (often dropping to 10-30%). For best results, upload the original photograph if available.'
})

with open('frontend/messages/en.json', 'w', encoding='utf-8') as f:
    json.dump(data_en, f, ensure_ascii=False, indent=2)