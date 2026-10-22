"""Check source snapshots, explanation anchors, and every local book link."""
import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
payload = (HERE/'book-data.js').read_text(encoding='utf-8')
data = json.loads(payload.removeprefix('window.BOOK = ').strip().removesuffix(';'))
keys = {item['key'] for item in data['lessons']}
assert len(keys) == len(data['lessons']) == 64
assert data['definitionCount'] == 60
assert len(data['blocks']) == 9
actual_definitions = set()
for path in ROOT.rglob('*.py'):
    if HERE in path.parents:
        continue
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            actual_definitions.add(node.name)
            actual_definitions.update(node.name+'.'+n.name for n in node.body if isinstance(n,ast.FunctionDef))
        elif isinstance(node,ast.FunctionDef):
            actual_definitions.add(node.name)
assert actual_definitions == {item['key'] for item in data['lessons'] if item['kind'] != '실행 구간'}
for item in data['lessons']+data['blocks']:
    original = (ROOT/item['file']).read_text(encoding='utf-8').splitlines()
    source = item['source'].splitlines()
    assert source == original[item['start']-1:item['start']-1+len(source)],item['key']
    assert item['steps'],item['key']
    for step in item['steps']:
        assert item['start'] <= step['line'] < item['start']+len(source),(item['key'],step)
        assert original[step['line']-1].strip(),(item['key'],step)
        assert all(step.get(key) for key in ('title','before','after','text'))
    assert set(item['calls']) <= keys,item['key']
    if 'rects' in item:
        for x,y,w,h in item['rects']:
            assert 0 <= x < x+w <= 380
            assert 0 <= y < y+h <= 488

class Page(HTMLParser):
    def __init__(self,path):
        super().__init__()
        self.ids=set(); self.links=[]; self.path=path
        self.feed(path.read_text(encoding='utf-8'))
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids,(self.path,attrs['id'])
            self.ids.add(attrs['id'])
        for key in ('href','src'):
            if key in attrs:self.links.append(attrs[key])
        if tag=='option' and 'value' in attrs:self.links.append(attrs['value'])

pages={p.name:Page(p) for p in HERE.glob('*.html')}
assert len(pages)==7
link_count=0
for page in pages.values():
    for link in page.links:
        target=urlsplit(link)
        assert not target.scheme,(page.path,link,'Book must work offline')
        path=(page.path.parent/unquote(target.path)).resolve() if target.path else page.path
        assert path.is_relative_to(HERE),link
        assert path.is_file(),(page.path,link)
        if target.fragment:
            fragment=unquote(target.fragment)
            if path.name=='Transformer_Code_Walkthrough_ELI5.html':
                assert fragment.removeprefix('fn=') in keys,(page.path,link)
            else:assert fragment in pages[path.name].ids,(page.path,link)
        link_count+=1
for file in data['sourceFiles']:
    assert 'file='+file in pages['Transformer_AI_Code_Book.html'].ids
assert (HERE/'assets/encoder-reference.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
script=(HERE/'book.js').read_text(encoding='utf-8')
required=set(re.findall(r"\$\('([^']+)'\)",script))
for name in ('Transformer_Code_Walkthrough_ELI5.html','Transformer_Encoder_Blocks_ELI5.html'):
    common=required-{'highlight','shape','block-position'}
    assert common <= pages[name].ids,(name,common-pages[name].ids)
assert {'highlight','shape','block-position'} <= pages['Transformer_Encoder_Blocks_ELI5.html'].ids
print(f'PASS: 7 pages, 60 definitions + 4 execution sections, 9 diagram blocks, {link_count} local links/assets')
