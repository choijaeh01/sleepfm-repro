from pathlib import Path

p = Path(r'C:\Projects\aiot\sleepfm-clinical\sleepfm\models\dataset.py')
text = p.read_text(encoding='utf-8')
old = 'path.split("/")[-1].split(".")[0]'
new = 'os.path.basename(path).split(".")[0]'
count = text.count(old)
text = text.replace(old, new)
backup = p.with_suffix('.py.bak_windows_pathfix')
backup.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
p.write_text(text, encoding='utf-8')
print({'replacements': count, 'backup': str(backup)})
