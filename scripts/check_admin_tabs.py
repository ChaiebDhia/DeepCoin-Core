import re
content = open('frontend/app/admin/page.tsx', encoding='utf-8-sig').read()
lines = content.splitlines()
print('Total lines:', len(lines))
for i, line in enumerate(lines[:120]):
    if 'tab' in line.lower() or 'Tab' in line or 'overview' in line.lower() or 'active' in line.lower():
        print(f'L{i+1}: {line.strip()[:120]}')
# last TabsContent
idx = content.rfind('</TabsContent>')
print('\nLast TabsContent context:')
print(repr(content[idx:idx+30]))
start = max(0, idx-100)
print('Before:', repr(content[start:idx]))

