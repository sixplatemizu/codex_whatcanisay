import io, os

p = os.path.join('src','app','main.py')
with io.open(p, 'r', encoding='utf-8', errors='replace') as f:
    s = f.read()
old = "if 'gaze_dot_switch' in locals() and gaze_dot_switch.value:"
if old in s:
    s = s.replace(old, '')
    with io.open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)
    print('fixed')
else:
    print('not found')
