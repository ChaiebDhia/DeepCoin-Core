with open('frontend/components/ui/TutorialModal.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's completely write a new clean TutorialModal.tsx based on the existing one,
# but keeping only ONE declaration of STEPS: Step[] = [ ... ]

# Let's find the start of the first STEPS
idx1 = text.find('  const STEPS: Step[] = [')
print(f'idx1: {idx1}')

idx2 = text.find('  const STEPS: Step[] = [', idx1 + 1)
print(f'idx2: {idx2}')

idx3 = text.find('  const STEPS: Step[] = [', idx2 + 1)
print(f'idx3: {idx3}')

