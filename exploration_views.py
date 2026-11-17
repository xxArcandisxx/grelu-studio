"""PWM scanning, motif placement simulations, and explicit model comparison."""
import math
import re
import time

from flask import Blueprint, current_app, make_response, render_template, request

from advanced_views import example_sequence, integer
from core.pwm import EXAMPLE_MATRIX, parse_motifs, scan_pwm
from experiment_store import experiment, save_experiment
from lab_views import check_csrf
from studio_services import length_error

exploration = Blueprint('exploration',__name__)
TITLES = {'pwm-scan':'PWM motif scanning','motif-simulation':'Motif simulations','model-comparison':'Model comparison'}


def integer_list(raw,label,low,high,limit=12):
    tokens = [token.strip() for token in raw.split(',')]
    if not tokens or len(tokens)>limit or any(not token for token in tokens):
        raise ValueError(f'{label} needs 1–{limit} comma-separated whole numbers.')
    result = [integer(token,label,low,high) for token in tokens]
    if len(result)!=len(set(result)):
        raise ValueError(f'{label} must not contain duplicates.')
    return result


def input_text(name,file_name):
    upload = request.files.get(file_name)
    return upload.read().decode('utf-8-sig') if upload and upload.filename else request.form.get(name,'')


@exploration.route('/pwm-scan',methods=['GET','POST'],defaults={'mode':'pwm-scan'})
@exploration.route('/motif-simulation',methods=['GET','POST'],defaults={'mode':'motif-simulation'})
@exploration.route('/model-comparison',methods=['GET','POST'],defaults={'mode':'model-comparison'})
def run(mode):
    from studio_web import MODELS, PROVENANCE, inference_lock, model_choices, parse_sequence, resolve_model
    from workbench import parse_batch
    choices = model_choices()
    alternate = 'human' if 'human' in choices['local'] else next((key for key in choices['local'] if key!='demo'),'demo')
    defaults = {'text':'','matrices':'','threshold':'0.8','background':'0.25,0.25,0.25,0.25','smoothing':'0.1',
                'model':'demo','model_a':'demo','model_b':alternate,'target':'0','target_a':'0','target_b':'0',
                'motif':'TATAAA','positions':'81,101,121','orientation':'both','partner':'','gaps':'0,5,10','policy':'strict'}
    values = {key:request.form.get(key,value) for key,value in defaults.items()}
    values['controls'] = 'controls' in request.form if request.method=='POST' else True
    result,error,status,presentation = None,None,200,{}
    if request.method=='POST':
        check_csrf()
        started = time.monotonic()
        try:
            values['text'] = input_text('text','sequence-file')
            if mode=='pwm-scan':
                values['matrices'] = input_text('matrices','matrix-file')
                if 'example' in request.form:
                    values['text'] = '>region_A\nGCGCTATAAAGCGCTATTAAGCGC\n>region_B\nCCCTTTATACCCGATATACCC'
                    values['matrices'] = EXAMPLE_MATRIX
                records = parse_batch(values['text'],parse_sequence)
                if sum(len(row['sequence']) for row in records)>50000:
                    raise ValueError('Scan at most 50,000 bases per run.')
                motifs = parse_motifs(values['matrices'])
                frequencies = [float(value.strip()) for value in values['background'].split(',')]
                result = scan_pwm(records,motifs,float(values['threshold']),frequencies,float(values['smoothing']))
                presentation = {'metrics':[['Sequences',len(records)],['Matrices',len(motifs)],['Hits',len(result['hits'])]]}
            elif mode=='motif-simulation':
                model = resolve_model(values['model'])
                if model is None:
                    raise ValueError('Choose an installed model.')
                length = MODELS[model]['seq_len']
                if 'example' in request.form:
                    values['text']='>synthetic_background\n'+example_sequence(length)
                    center = length//2+1
                    values['positions'] = ','.join(str(max(1,center+offset)) for offset in (-20,0,20) if center+offset+len(values['motif'])-1<=length)
                records = parse_batch(values['text'],parse_sequence)
                if len(records)!=1:
                    raise ValueError('Provide one reference sequence or FASTA record.')
                if len(records[0]['sequence'])!=length:
                    raise ValueError(length_error(length,records))
                motif,partner = values['motif'].strip().upper(),values['partner'].strip().upper()
                if not re.fullmatch('[ACGT]{3,50}',motif) or (partner and not re.fullmatch('[ACGT]{3,50}',partner)):
                    raise ValueError('Use 3–50 A/C/G/T bases for each motif.')
                positions = integer_list(values['positions'],'Positions',1,length)
                gaps = integer_list(values['gaps'],'Partner gaps',0,length,4) if partner else [0]
                if values['orientation'] not in ('+','-','both'):
                    raise ValueError('Choose a valid primary-motif orientation.')
                strands = ('+','-') if values['orientation']=='both' else (values['orientation'],)
                if len(positions)*len(strands)*len(gaps)>12:
                    raise ValueError('Use at most 12 combinations of position, strand, and partner spacing.')
                target = integer(values['target'],'Output track index',0,100000)
                from workflows.explore import simulate_motif
                with inference_lock:
                    result = simulate_motif(MODELS[model],records[0]['sequence'],motif,positions,target,strands,partner,gaps,values['controls'])
                result.update(inputs=records,actual_model=PROVENANCE[model]['model_id'],demo=model=='demo')
                result['model']['provenance']=PROVENANCE[model]
                conditions = result['results']['conditions']
                presentation = {'metrics':[['Reference score',result['results']['reference_score']],['Conditions',len(conditions)],['Controls','Scrambled' if values['controls'] else 'None']],
                    'tables':[{'title':'Motif placement effects','headings':['Position','Strand','Partner gap','Score','Delta','Scrambled score','Motif − control'],
                               'rows':[[row[key] for key in ('position','strand','partner_gap','score','delta','control_score','motif_minus_control')] for row in conditions]}],
                    'plots':[{'title':'Condition scores (configuration order)','series':[{'label':'Motif placement','values':[row['score'] for row in conditions]},
                        {'label':'Unmodified input','values':[result['results']['reference_score']]*len(conditions)}]}]}
            else:
                keys = [resolve_model(values['model_a']),resolve_model(values['model_b'])]
                if any(key is None for key in keys):
                    raise ValueError('Choose two installed models.')
                if 'example' in request.form:
                    length=max(MODELS[key]['seq_len'] for key in keys)
                    values['text']='\n'.join(f'>sequence_{index+1}\n'+''.join(__import__('random').Random(31+index).choices('ACGT',k=length)) for index in range(3))
                records = parse_batch(values['text'],parse_sequence)
                if len(records)>6:
                    raise ValueError('Compare at most six sequences per CPU run.')
                if values['policy'] not in ('strict','center'):
                    raise ValueError('Choose strict lengths or explicit center-crop/N-pad handling.')
                targets = [integer(values[key],'Track index',0,100000) for key in ('target_a','target_b')]
                from workflows.explore import compare_models
                with inference_lock:
                    result = compare_models([MODELS[key] for key in keys],records,targets,values['policy'])
                result.update(inputs=records,demo='demo' in keys,model_provenance=[PROVENANCE[key] for key in keys],
                              actual_model=' vs '.join(PROVENANCE[key]['model_id'] for key in keys),
                              model_configurations=[{name:value for name,value in MODELS[key].items() if name!='path'} for key in keys])
                if keys[0]==keys[1]:
                    result['warnings'].append('The same model is selected twice. This is a reproducibility check; identical inputs should give identical scores.')
                rows=result['results']['comparisons']
                correlation=result['results']['pearson']
                presentation={'metrics':[['Sequences',len(records)],['Pearson r',correlation if correlation is not None else 'Undefined']],
                    'tables':[{'title':'Selected-track comparison','headings':['Record','Model A','Model B','Raw B − A','Input A','Input B'],
                        'rows':[[row[key] for key in ('id','score_a','score_b','raw_difference','transformation_a','transformation_b')] for row in rows]}],
                    'plots':[{'title':'Selected-track scores by input record','series':[{'label':'Model A','values':[row['score_a'] for row in rows]},
                        {'label':'Model B','values':[row['score_b'] for row in rows]}]}]}
            result['workflow']=mode
            result['runtime_s']=round(time.monotonic()-started,3)
            replay_keys = {'pwm-scan':('text','matrices','threshold','background','smoothing'),
                'motif-simulation':('text','model','motif','positions','orientation','partner','gaps','target','controls'),
                'model-comparison':('text','model_a','model_b','target_a','target_b','policy')}[mode]
            result['saved_run_id']=save_experiment(TITLES[mode],mode,result,request.path,
                {key:values[key] for key in replay_keys},presentation)
            from report_builder import build_report
            presentation=build_report({'payload':result,'presentation':presentation})
        except (ValueError,UnicodeError) as exc:
            result = None
            error,status=str(exc),400
        except Exception:
            result = None
            current_app.logger.exception('Exploration workflow failed: %s',mode)
            error,status='The model could not complete this analysis. Check its installation and the server log.',500
    response=make_response(render_template('exploration.html',mode=mode,title=TITLES[mode],values=values,
        result=result,error=error,choices=choices,selected=values['model'],report=presentation,
        saved=experiment(result['saved_run_id']) if result else None),status)
    response.headers['Cache-Control']='no-store'
    return response
