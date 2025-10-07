import io

p = 'src/app/main.py'
txt = io.open(p, 'r', encoding='utf-8', errors='replace').read().splitlines()

# 找到包含 h0, w0 = frame.shape[:2] 的行
targets = [i for i, ln in enumerate(txt) if 'h0, w0 = frame.shape[:2]' in ln]
changed = False
for idx in targets:
    prev = idx - 1
    if prev >= 0 and 'try:' in txt[prev]:
        # 去掉同一行注释后拼接的 try:
        txt[prev] = txt[prev].replace('try:', '')
        # 在后续 50 行内删除与之对应的 except Exception: 与下一行的 pass
        for j in range(idx, min(len(txt), idx + 60)):
            if txt[j].lstrip().startswith('except Exception'):
                # 删除 except 与下一行（可能是 pass）
                txt[j] = ''
                if j + 1 < len(txt) and txt[j+1].lstrip().startswith('pass'):
                    txt[j+1] = ''
                break
        changed = True

if changed:
    io.open(p, 'w', encoding='utf-8', newline='').write('\n'.join([l for l in txt]))
    print('fixed try/except around h0,w0 block')
else:
    print('no changes')

