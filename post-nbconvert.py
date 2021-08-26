import re

math_mode = False
indexes = []

with open('README.md') as f:
    content = f.read()

new_content = ''
i = 0
while i < len(content):
    if content[i] == '$':
        if not math_mode:
            math_mode = True
            if i+1 < len(content) and content[i+1] == '$':
                i += 1
            i_start = i+1
        else:
            math_mode = False
            formula = content[i_start:i]
            formula = formula.replace('&', '%26')
            formula = formula.replace('+', '%2B')
            formula = formula.replace('\n', ' ')
            formula = formula.strip(' ')
            url = rf'https://render.githubusercontent.com/render/math?math={formula}'
            new_content += rf'<img src="{url}">'
            if i+1 < len(content) and content[i+1] == '$':
                i += 1
    elif not math_mode:
        new_content += content[i]
    i += 1

with open('README.md', 'w') as f:
    f.write(new_content)
