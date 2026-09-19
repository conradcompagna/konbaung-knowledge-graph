from pathlib import Path
import json, zipfile, hashlib, sys, re
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parents[1]; root=O.parent
inv=json.loads((O/'inventory.json').read_text(encoding='utf-8'))
wanted={
 'chronology.csv':('Konbaung_Chronological_Trends_Evidence.zip','results/triple_analysis_ledger.csv'),
 'page_chronology.csv':('Konbaung_Complete_Narrative_Analysis.zip','results/page_chronology_full.csv'),
 'person_edges.csv':('Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip','09_ENTITY_RESOLUTION_ANALYSIS/04_OUTPUTS/named_person_network_edges.csv'),
 'person_centrality.csv':('Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip','09_ENTITY_RESOLUTION_ANALYSIS/04_OUTPUTS/named_person_network_centrality.csv'),
 'actor_groups.csv':('Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip','03_BASE_STATISTICAL_ANALYSIS/02_METHODS/entity_actor_group_map.csv'),
 'domain_map.csv':('Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip','03_BASE_STATISTICAL_ANALYSIS/02_METHODS/relation_domain_map.csv'),
}
provenance=[]
for name,(arc,suffix) in wanted.items():
 with zipfile.ZipFile(root/arc) as z:
  n=next(n for n in z.namelist() if n.endswith(suffix));raw=z.read(n)
  (O/'inputs'/name).write_bytes(raw)
  provenance.append(dict(file=name,source=arc+'!/'+n,sha256=hashlib.sha256(raw).hexdigest()))
for name in ['triples.jsonl','sentences.jsonl','pages.jsonl']:
 a=(O/'inputs'/name).read_bytes();b=(O/'recovered_source/data'/name).read_bytes()
 assert a==b,name
 provenance.append(dict(file=name,source='CRC and SHA256 verified recovered canonical archive, identical to 08_NEW_STRUCTURAL_AUDITS/01_CANONICAL_DATA',sha256=hashlib.sha256(a).hexdigest()))
T=[json.loads(s) for s in (O/'inputs/triples.jsonl').read_text(encoding='utf-8').splitlines()]
S=[json.loads(s) for s in (O/'inputs/sentences.jsonl').read_text(encoding='utf-8').splitlines()]
d=pd.DataFrame([dict(triple_id=t['triple_id'],sentence_id=t['sentence_id'],volume=t['volume_id'],page=t['owner_page'],ordinal=t['ordinal'],
 s=t['subject']['tag'],p=t['predicate']['tag'],o=t['object']['tag'],srow=t['subject']['embedding_row'],prow=t['predicate']['embedding_row'],orow=t['object']['embedding_row'],
 S=t['axial_categories']['subject']['id'],P=t['axial_categories']['relation']['id'],O=t['axial_categories']['object']['id']) for t in T])
c=pd.read_csv(O/'inputs/chronology.csv');assert c.triple_id.is_unique
d=d.merge(c[['triple_id','reign_segment','reign_inherited','register','phase']],on='triple_id',validate='one_to_one')
d['page_key']=d.volume+':'+d.page.astype(str)
d.to_csv(O/'inputs/analysis_triples.csv.gz',index=False)
checks={'triples':len(d),'sentences':len(S),'triple_bearing_sentences':d.sentence_id.nunique(),'unique_entities':len(set(d.s)|set(d.o)),
 'unique_predicates':d.p.nunique(),'entity_categories':len(set(d.S)|set(d.O)),'relation_categories':d.P.nunique(),'triple_owner_pages':d.page_key.nunique(),
 'duplicate_triple_ids':int(d.triple_id.duplicated().sum()),'missing_sentence_joins':len(set(d.sentence_id)-{s['sentence_id'] for s in S}),
 'unavailable_annotation_sentences':sum(s['annotation_status']!='canonical_v3' for s in S),'category_self_occurrences':int((d.S==d.O).sum()),'exact_tag_self_occurrences':int((d.s==d.o).sum()),
 'register_values':d.register.value_counts(dropna=False).to_dict(),'reign_values':d.reign_segment.value_counts().to_dict()}
assert checks['triples']==27129 and checks['sentences']==11282 and checks['entity_categories']==52 and checks['relation_categories']==81
(O/'input_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
(O/'data_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
