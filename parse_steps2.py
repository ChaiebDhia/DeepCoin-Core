with open('frontend/components/ui/TutorialModal.tsx', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'const STEPS: Step[] = [' in line:
        print(f"Line {i+1}:")
        for j in range(max(0, i-5), min(len(lines), i+6)):
            print(f"{j+1}: {lines[j].rstrip()}")
        print('-'*40)
