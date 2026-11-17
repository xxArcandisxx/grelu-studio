"""Create the missing synthetic inputs and a complete website acceptance manifest."""
import csv
import io
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab_compute import training_example

DEST=ROOT/'test-inputs'/'website-suite'
DEST.mkdir(parents=True,exist_ok=True)

def save(name,value):
    path=DEST/name
    path.write_text(json.dumps(value,indent=2)+'\n' if isinstance(value,(dict,list)) else value)
    return str(path.relative_to(ROOT))

account=save('account.json',{'name':'Website QA Researcher','email':'website.qa@example.test',
    'password':'College-QA-initial-2026','reset_password':'College-QA-replacement-2026',
    'purpose':'Synthetic credentials for isolated local acceptance tests only; no email is sent.'})
patterns=save('motif-patterns.txt','Motif_A:TATAAA\nMotif_B:GATA\n')
training=save('training-200bp.csv',training_example(200))
trained_rows=list(csv.reader(io.StringIO(training_example(2114))))[:9]
stream=io.StringIO();csv.writer(stream,lineterminator='\n').writerows(trained_rows)
fine_human=save('fine-tuning-2114bp.csv',stream.getvalue())
invalid_dna=save('invalid-dna.txt','>invalid\n'+'A'*99+'X'+'C'*100+'\n')
invalid_csv=save('invalid-variant-columns.csv','id,sequence,position,reference,alternate\na,ACGT,1,A,C\n')
invalid_pwm=save('invalid-matrix.jaspar','>incomplete\nA [ 1 2 3 ]\n')
duplicate_train=save('invalid-training-duplicates.csv','sequence,label\n'+'\n'.join('A'*200+',0.5' for _ in range(8))+'\n')
settings=save('settings.json',{
 'design_demo':{'iterations':'2','candidates':'4','target':'0','off_target':'1','edit_start':'81','edit_end':'120'},
 'design_human':{'iterations':'1','candidates':'2','target':'0','edit_start':'1050','edit_end':'1065'},
 'scan_demo':{'start':'101','end':'108','target':'0'},'scan_human':{'start':'1058','end':'1059','target':'0'},
 'attribution_demo':{'start':'51','end':'150','target':'0','randomized_check':True},
 'attribution_human':{'start':'1008','end':'1107','target':'0','randomized_check':True},
 'training':{'length':'200','epochs':'2','learning_rate':'0.001'},
 'fine_tuning':{'epochs':'1','learning_rate':'0.0001','target':'0'},
 'simulation_demo':{'motif':'TATAAA','positions':'81','orientation':'both','partner':'GATA','gaps':'0,5','target':'0','controls':True},
 'simulation_human':{'motif':'TATAAA','positions':'1058','orientation':'+','partner':'','gaps':'0','target':'0','controls':True},
 'pwm':{'threshold':'0.8','background':'0.25,0.25,0.25,0.25','smoothing':'0.1'},
 'comparison':{'policy':'center','target_a':'0','target_b':'0'},
 'history':{'query':'regulatory','workflow':'variants'},
 'mobile':{'width':390,'height':844},
 'saturation':{'locus':next(iter(json.loads((ROOT/'curated_loci.json').read_text()))),'expected_substitutions':24},
 'model_zoo':{'repo':'Genentech/GM12878_dnase-model','revision':'9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e',
              'filename':'model.ckpt','mode':'cached','invalid_revision':'0'*40},
})
cases=[]
for model,folder in [('demo','quick-demo-200bp'),('human','trained-model-2114bp')]:
    base='test-inputs/'+folder+'/'
    cases += [
      {'id':f'variants-{model}','kind':'variant','route':'/variants','model':model,
       'files':{'reference-file':base+'01a_Predicted_Regulatory_Effects_reference.fasta','alternate-file':base+'01b_Predicted_Regulatory_Effects_alternate.fasta'}},
    ]
    for route,filename in [('/allele-specific','02_Allele_Specific_Regulation_two_haplotypes.fasta'),
       ('/prioritize','03_Variant_Prioritization_reference_and_two_SNVs.fasta'),
       ('/multi-allelic','04_Multi_Allelic_reference_and_three_alternates.fasta'),
       ('/predict','06_Sequence_Prediction_three_records.fasta'),('/qc','07_Sequence_QC_exact_short_and_unknown.fasta')]:
        cases.append({'id':route[1:]+'-'+model,'kind':'analysis','route':route,'model':model,'files':{'file':base+filename}})
    cases.append({'id':'models-'+model,'kind':'analysis','route':'/models','model':model,'files':{}})
    prefix='demo-200bp' if model=='demo' else 'trained-2114bp'
    for mode,fixture in [('design',prefix+'-seed.txt'),('batch-variants',prefix+'-variants.csv'),
                         ('mutagenesis',prefix+'-seed.txt'),('attribution',prefix+'-seed.txt')]:
        cases.append({'id':mode+'-'+model,'kind':'advanced','route':'/'+mode,'model':model,
                      'files':{'file':'test-inputs/new-workflows/'+fixture}})
    cases.append({'id':'simulation-'+model,'kind':'simulation','route':'/motif-simulation','model':model,
        'files':{'sequence-file':'test-inputs/new-workflows/'+prefix+'-seed.txt'}})
cases += [
 {'id':'prediction-shuffles','kind':'analysis','route':'/predict','model':'demo','shuffle':True,
  'files':{'file':'test-inputs/quick-demo-200bp/06_Sequence_Prediction_three_records.fasta'}},
 {'id':'curated-saturation','kind':'saturation','route':'/saturation','files':{}},
 {'id':'benchmark-regression','kind':'research','route':'/research/benchmark','evaluation':'regression',
  'files':{'file':'test-inputs/09a_Benchmark_Evaluation_regression.csv'}},
 {'id':'benchmark-binary','kind':'research','route':'/research/benchmark','evaluation':'binary',
  'files':{'file':'test-inputs/09b_Benchmark_Evaluation_binary_classification.csv'}},
 {'id':'track-analysis','kind':'research','route':'/research/tracks','files':{'file':'test-inputs/10_Track_Analysis_three_numeric_tracks.csv'}},
 {'id':'bed-validation','kind':'research','route':'/research/bed','files':{'file':'test-inputs/11_BED_Validation_two_valid_two_invalid.bed'}},
 {'id':'motifs','kind':'motifs','route':'/motifs','patterns':patterns,'files':{'file':'test-inputs/new-workflows/motif-regions.txt'}},
 {'id':'pwm-jaspar','kind':'pwm','route':'/pwm-scan','files':{'sequence-file':'test-inputs/new-workflows/pwm-regions.txt','matrix-file':'test-inputs/new-workflows/pwm-motifs.jaspar'}},
 {'id':'pwm-meme','kind':'pwm','route':'/pwm-scan','files':{'sequence-file':'test-inputs/new-workflows/pwm-regions.txt','matrix-file':'test-inputs/new-workflows/pwm-motifs.meme'}},
 {'id':'model-comparison','kind':'comparison','route':'/model-comparison','files':{'sequence-file':'test-inputs/new-workflows/trained-2114bp-comparison-records.txt'}},
 {'id':'training','kind':'training','route':'/training','files':{'file':training}},
 {'id':'fine-tuning-local','kind':'fine-tuning-local','route':'/fine-tuning','files':{'file':training}},
 {'id':'fine-tuning-human','kind':'fine-tuning-human','route':'/fine-tuning','files':{'file':fine_human}},
 {'id':'accounts','kind':'accounts','route':'/account/register','scenario':account,'files':{}},
 {'id':'model-zoo-cache','kind':'model-zoo','route':'/model-zoo','scenario':settings,'files':{}},
 {'id':'history-reports','kind':'history','route':'/experiments','scenario':settings,'files':{}},
 {'id':'mobile-navigation','kind':'mobile','route':'/','scenario':settings,'files':{}},
 {'id':'negative-inputs','kind':'negative','route':'/predict','files':{'dna':invalid_dna,'csv':invalid_csv,'pwm':invalid_pwm,'training':duplicate_train}},
]
save('manifest.json',{'version':1,'description':'File-only acceptance inputs. No Run example actions or invented numerical responses.',
 'settings':settings,'accounts':account,'cases':cases,'external_dependencies':['Existing cached pinned GM12878 checkpoint; live model downloads are excluded from this offline suite.']})
for case in cases:
    for path in case['files'].values():
        assert (ROOT/path).is_file(),path
print(f'Validated {len(cases)} file-backed scenarios.')
