"""Static whole-website inverse usage inventory; no website/library imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
EXCLUDED = ['.git', '.venv', '.codex', '.agents', '.aws', '__pycache__',
            'Testing', 'test-results', 'test-inputs', 'datasets', 'instance',
            'runs', 'uploads', '.test-tools']


def main():
    command = ['rg', '--files', '--hidden', '--no-ignore']
    for name in EXCLUDED:
        command += ['-g', '!' + name + '/**']
    command += ['-g', '!**/__pycache__/**', '-g', '!*.pyc']
    suffixes = {'.py', '.js', '.mjs', '.cjs', '.html', '.htm', '.sh',
                '.toml', '.json', '.yaml', '.yml', '.css', '.svg'}
    files = sorted(p for p in subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
                   if Path(p).suffix in suffixes)
    pattern = re.compile(r'one_hot_to_strings|one_hot_to_indices|indices_to_strings|convert_input_type')
    imports, hits, dynamic, errors, inventory = [], [], [], [], {}
    for name in files:
        data = (ROOT / name).read_bytes()
        text = data.decode('utf-8')
        inventory[name] = hashlib.sha256(data).hexdigest()
        group = 'test/tooling' if name.startswith(('tests/', 'scripts/')) else 'website/source/config'
        for line, value in enumerate(text.splitlines(), 1):
            if pattern.search(value):
                hits.append(dict(path=name, line=line, text=value, group=group))
        if Path(name).suffix != '.py':
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            errors.append(dict(path=name, error=str(exc)))
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and 'grelu' in ast.unparse(node):
                imports.append(dict(path=name, line=node.lineno, statement=ast.unparse(node), group=group))
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                if fn in ['getattr', 'eval', 'exec', '__import__'] or 'import_module' in fn:
                    dynamic.append(dict(path=name, line=node.lineno, expression=ast.unparse(node), group=group))
    library = ROOT / '.venv/lib/python3.14/site-packages/grelu'
    external = []
    for path in sorted(library.rglob('*.py')):
        for line, value in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(value):
                external.append(dict(path=str(path.relative_to(library)), line=line, text=value))
    result = dict(method='Whole-website source scan plus AST import/dynamic-call inventory. Manual indirect-path review: USAGE_AUDIT.md.',
                  root=str(ROOT), command=command, excluded_directories=EXCLUDED,
                  suffixes=sorted(suffixes), file_count=len(files), source_sha256=inventory,
                  inverse_and_dispatcher_hits=hits, grelu_imports=imports,
                  dynamic_calls_for_manual_review=dynamic, python_parse_errors=errors,
                  library_inverse_and_dispatcher_callsites=external,
                  website_hit_count=sum(h['group'] == 'website/source/config' for h in hits))
    (OUT / 'usage-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(file_count=len(files), website_hit_count=result['website_hit_count'], parse_errors=errors)))
    return 1 if result['website_hit_count'] or errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
