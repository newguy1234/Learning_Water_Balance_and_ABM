from pathlib import Path
import shutil

root=Path(r'C:\Users\psypc2\Documents\PaperAtlas-copy')
backup=Path('.presentation-build/atlas-backup');backup.mkdir(parents=True,exist_ok=True)
for name in ['providers.py','cli.py','service.py','app.py']:
 p=root/'paper_atlas'/name
 if not (backup/name).exists():shutil.copy2(p,backup/name)
 s=p.read_text(encoding='utf-8')
 if name=='providers.py':
  s=s.replace('{"cs_methods", "cs_applied"}', '{"cs_methods", "cs_applied", "all_fields"}')
  s=s.replace('Scope must be cs_methods or cs_applied','Scope must be cs_methods, cs_applied or all_fields')
  s=s.replace('search_openalex_semantic(seed_text, limit, None, "cs_applied")','search_openalex_semantic(seed_text, limit, None, "all_fields")')
 if name=='cli.py':
  s=s.replace('choices=("cs_methods", "cs_applied")','choices=("cs_methods", "cs_applied", "all_fields")')
  s=s.replace('Run a multi-query, CS-scoped literature search','Run a multi-query literature search with explicit discipline scope')
  if '"japan_farmer_fieldwork":' not in s:
   s=s.replace('RESEARCH_PRESETS = {','RESEARCH_PRESETS = {\n    "japan_farmer_fieldwork": [\n        "Japan rice farmers irrigation water management interviews field observations",\n        "Japan rice farmers fertilizer reduction behavioural field experiment",\n        "Japanese farmers labour constraints irrigation survey",\n        "Japan rice farmers risk perceptions information adoption fertilizer",\n        "Promoting Methane Reductions Through Behavioural Nudges Informational Boosts",\n        "Can objective information update subjective beliefs sustainable farming Japanese rice farmers",\n    ],')
 if name=='service.py':
  s=s.replace('"Computer science methods" if scope == "cs_methods" else "CS methods applied across fields"','{"cs_methods": "Computer science methods", "cs_applied": "CS methods applied across fields", "all_fields": "All disciplines"}[scope]')
 if name=='app.py' and '"All disciplines' not in s:
  s=s.replace('scope_options = {','scope_options = {\n                        "All disciplines · agriculture, environment and behaviour": "all_fields",')
 p.write_text(s,encoding='utf-8')
print('Updated Atlas provider, CLI, collection labels and UI; originals backed up.')
