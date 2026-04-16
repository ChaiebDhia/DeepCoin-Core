import os
import re
import json

FRONTEND_DIR = '../frontend'
MESSAGES_DIR = os.path.join(FRONTEND_DIR, 'messages')

EN_MESSAGES = {}
FR_MESSAGES = {}

TEXT_REGEX = re.compile(r'>([^<>{]+?)<')

def translate_to_french(text):
    # This acts as a placeholder for LLM or API translation
    # In a real pipeline, you would use google_trans, OpenAI, etc.
    return f"[FR] {text}"

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    matches = TEXT_REGEX.findall(content)
    
    unique_keys = {}
    for match in matches:
        text = match.strip()
        if len(text) > 2 and not text.isdigit() and re.search('[a-zA-Z]', text):
            # Generate a safe translation key
            key = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())[:30]
            
            EN_MESSAGES[key] = text
            FR_MESSAGES[key] = translate_to_french(text)
            
            # Replace in file with `{t('key')}` (simplified)
            # A real AST parser is recommended for complex substitutions
            new_content = new_content.replace(f">{match}<", f">{{t('{key}')}}<")
            
    if new_content != content:
        # Add import if translations were made
        if "useTranslations" not in new_content and '"use client"' in new_content:
            new_content = new_content.replace(
                '"use client";', 
                '"use client";\nimport { useTranslations } from "next-intl";\n'
            )
            # Inject const t = useTranslations() inside the default export component
            new_content = re.sub(
                r'(export default function [^{]+\{)',
                r'\1\n  const t = useTranslations("Common");',
                new_content
            )
            print(f"Updated {filepath} with next-intl")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

def main():
    os.makedirs(MESSAGES_DIR, exist_ok=True)
    
    print("Scanning frontend components...")
    
    # Target components folder
    components_dir = os.path.join(FRONTEND_DIR, 'components')
    for root, dirs, files in os.walk(components_dir):
        for file in files:
            if file.endswith('.tsx'):
                process_file(os.path.join(root, file))
                
    # Save the JSON dictionaries
    with open(os.path.join(MESSAGES_DIR, 'en.json'), 'w', encoding='utf-8') as f:
        json.dump({"Common": EN_MESSAGES}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(MESSAGES_DIR, 'fr.json'), 'w', encoding='utf-8') as f:
        json.dump({"Common": FR_MESSAGES}, f, indent=2, ensure_ascii=False)
        
    print("Translation files en.json and fr.json generated successfully.")
    print("Please review the changes. You will likely need to adjust complex nested JSX expressions manually.")

if __name__ == "__main__":
    main()
