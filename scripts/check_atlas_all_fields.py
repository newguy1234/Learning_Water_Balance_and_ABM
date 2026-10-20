import sys
from unittest.mock import patch
sys.path.insert(0,r'C:\Users\psypc2\Documents\PaperAtlas-copy')
from paper_atlas import providers as p
from paper_atlas.cli import RESEARCH_PRESETS
q='Japan farmer fertiliser behaviour'
assert p._scope_query(q,'all_fields')==q
assert 'computer science' in p._scope_query(q,'cs_applied')
for scope in ['all_fields','cs_methods']:
 with patch.object(p,'_get_json',return_value={'results':[]}) as call:
  p.search_openalex(q,5,1990,scope)
  params=call.call_args.kwargs['params']
  assert ('primary_topic.field.id:17' in params.get('filter',''))==(scope=='cs_methods')
with patch.dict(p._PROVIDERS,{'openalex':lambda *args:[]}):
 assert p.search_literature(q,providers=('openalex',),scope='all_fields')==([],[])
assert 'japan_farmer_fieldwork' in RESEARCH_PRESETS
print('PASS: all-fields query, provider filtering, dispatcher and research preset')
