from pathlib import Path
p = Path(r'C:\Projects\aiot\sleepfm-clinical\sleepfm\preprocessing\preprocessing.py')
text = p.read_text(encoding='utf-8')
old = "edf_file.split('/')[-1].replace(replace_str, '.hdf5')"
new = "os.path.basename(edf_file).replace(replace_str, '.hdf5')"
count = text.count(old)
backup = p.with_suffix('.py.bak_windows_pathfix')
backup.write_text(text, encoding='utf-8')
text = text.replace(old, new)
p.write_text(text, encoding='utf-8')
print({'replacements': count, 'backup': str(backup)})
