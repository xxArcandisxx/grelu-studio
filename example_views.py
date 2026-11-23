"""Model-sized example inputs and downloads; never runs inference or training."""
import csv
import io
import json
from pathlib import Path
import random
import zipfile

from flask import Blueprint, abort, jsonify, make_response, request, send_file

examples = Blueprint('examples', __name__)
ROOT = Path(__file__).resolve().parent
WORKFLOWS = {'variants','predict','qc','models','allele-specific','prioritize','multi-allelic',
    'design','batch-variants','mutagenesis','attribution','training','fine-tuning','motifs',
    'pwm-scan','motif-simulation','model-comparison','benchmark','tracks','dataset','saturation','model-zoo'}


def _integer(raw, label, low, high):
    try:
        value = int(raw)
    except (ValueError, TypeError):
        raise ValueError(f'{label} must be a whole number between {low} and {high}.') from None
    if str(value) != str(raw).strip() or not low <= value <= high:
        raise ValueError(f'{label} must be a whole number between {low} and {high}.')
    return value


def _sequence(length, seed=19):
    return ''.join(random.Random(seed).choices('ACGT', k=length))


def _fasta(rows):
    return ''.join(f'>{name}\n' + '\n'.join(sequence[i:i+70] for i in range(0,len(sequence),70))+'\n' for name,sequence in rows)


def _fixture(relative, fallback):
    path = ROOT / 'test-inputs' / relative
    return path.read_text() if path.is_file() else fallback()


def _file(name, content, media_type='text/plain'):
    return {'name':name,'content':content,'media_type':media_type}


def build_example(workflow, query):
    from core.model_lengths import required_length
    from studio_web import MODELS, ZOO_MODELS, model_choices, resolve_model
    if workflow not in WORKFLOWS:
        abort(404)
    fields, files = {}, []
    message = 'Example inputs loaded. Review the settings, then select Run when ready.'
    if workflow == 'dataset':
        kind = query.get('kind', 'fasta')
        if kind not in {'fasta', 'csv'}:
            raise ValueError('Choose FASTA / DNA or labeled CSV.')
        if kind == 'csv':
            from lab_compute import training_example
            text = _fixture('new-workflows/training-gc-regression.csv', lambda:training_example(200))
            name = 'training-gc-regression.csv'
        else:
            from research_views import EXAMPLES
            text = _fixture('new-workflows/dataset-overview.txt', lambda:EXAMPLES['dataset'])
            name = 'dataset-overview.fasta'
        fields = {'text':text, 'kind':kind}
        files = [_file(name, text, 'text/csv' if kind == 'csv' else 'text/plain')]
        message = 'Dataset example loaded. Select Run analysis to calculate its overview.'
    elif workflow in {'benchmark','tracks'}:
        from research_views import EXAMPLES
        kind = query.get('kind','regression')
        if workflow == 'benchmark':
            if kind not in {'regression','binary'}:
                raise ValueError('Choose regression or binary classification.')
            name = '09b_Benchmark_Evaluation_binary_classification.csv' if kind=='binary' else '09a_Benchmark_Evaluation_regression.csv'
            text = _fixture(name,lambda:EXAMPLES['benchmark'])
            fields['kind'] = kind
        else:
            name = '10_Track_Analysis_three_numeric_tracks.csv'
            text = _fixture(name,lambda:EXAMPLES[workflow])
        fields['text'] = text
        files = [_file(name,text,'text/csv')]
    elif workflow=='motifs':
        text = _fixture('new-workflows/motif-regions.txt',lambda:_fasta([('region_A',_sequence(60)+'TATAAAGATATATAAA'+_sequence(60))]))
        patterns = _fixture('website-suite/motif-patterns.txt',lambda:'Motif_A:TATAAA\nMotif_B:GATA\n')
        fields = {'text':text,'patterns':patterns,'k':'6'}
        files = [_file('motif-regions.fasta',text),_file('motif-patterns.txt',patterns)]
    elif workflow=='pwm-scan':
        from core.pwm import EXAMPLE_MATRIX
        text = _fixture('new-workflows/pwm-regions.txt',lambda:_fasta([('region_A','GCGCTATAAAGCGCTATTAAGCGC')]))
        matrices = _fixture('new-workflows/pwm-motifs.jaspar',lambda:EXAMPLE_MATRIX)
        fields = {'text':text,'matrices':matrices,'threshold':'0.8','background':'0.25,0.25,0.25,0.25','smoothing':'0.1'}
        files = [_file('pwm-regions.fasta',text),_file('pwm-motifs.jaspar',matrices)]
    elif workflow=='saturation':
        from curated_scan import LOCI
        key = next(iter(LOCI))
        fields = {'locus':key}
        content = json.dumps({'locus':key,'label':LOCI[key]['label'],'assembly':'hg38'},indent=2)+'\n'
        files = [_file('reference-locus.json',content,'application/json')]
        message = 'Example locus selected. Select Run live scan when the configured checkpoint is available.'
    elif workflow=='model-zoo':
        repo = query.get('repo','Genentech/GM12878_dnase-model')
        if repo not in ZOO_MODELS:
            raise ValueError('Choose a model from the catalog.')
        revision = '9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e' if repo=='Genentech/GM12878_dnase-model' else ''
        fields = {'revision':revision}
        content = json.dumps({'repo':repo,'revision':revision or 'Resolve published revision when installing'},indent=2)+'\n'
        files = [_file('model-zoo-selection.json',content,'application/json')]
        message = 'Example configuration loaded. No model has been downloaded or installed.'
    else:
        model = resolve_model(query.get('model','demo'))
        if model is None:
            raise ValueError('Choose an installed model before loading an example.')
        length = required_length(MODELS[model])
        seed = _sequence(length)
        folder = 'quick-demo-200bp' if length==200 else 'trained-model-2114bp' if length==2114 else None
        if workflow=='training':
            length = _integer(query.get('length','200'),'Training length',50,2000)
        elif workflow=='fine-tuning' and length>4096:
            raise ValueError('Choose a model with at most 4,096 input bases for this CPU fine-tuning example.')
        if workflow in {'training','fine-tuning'}:
            from lab_compute import training_example
            content = training_example(length)
            if workflow=='fine-tuning':
                content = '\n'.join(content.splitlines()[:9])+'\n'
            fields = {'text':content,'epochs':'2' if workflow=='training' else '1',
                      'learning_rate':'0.001' if workflow=='training' else '0.0001'}
            if workflow=='training':fields['length']=str(length)
            else:fields['target']='0'
            files = [_file(f'{workflow}-{length}bp.csv',content,'text/csv')]
            message = 'Synthetic GC-content regression data loaded. Review it, then select Start training or Start fine-tuning; loading does not start a job.'
        elif workflow=='model-comparison':
            first = resolve_model(query.get('model_a','demo'))
            second = resolve_model(query.get('model_b','demo'))
            if first is None or second is None:
                raise ValueError('Choose two installed models.')
            lengths = [required_length(MODELS[key]) for key in (first,second)]
            if max(lengths)>20000:
                raise ValueError('Choose models with windows up to 20,000 bases for CPU comparison.')
            length = max(lengths)
            content = _fasta([(f'sequence_{i+1}',_sequence(length,31+i)) for i in range(3)])
            fields = {'text':content,'target_a':'0','target_b':'0','policy':'strict' if lengths[0]==lengths[1] else 'center'}
            files = [_file(f'model-comparison-{length}bp.fasta',content)]
            if lengths[0]!=lengths[1]:
                message = 'Example loaded with explicit center-crop/N-pad handling because the selected models need different window lengths. Review before running.'
        elif workflow=='models':
            fields = {'model':model}
            selection = {'model':model,'label':model_choices()['local'][model],'input_length':length}
            files = [_file('model-selection.json',json.dumps(selection,indent=2)+'\n','application/json')]
            message = 'Installed model selected. Select Run analysis to inspect it; no model has been loaded by this button.'
        elif workflow=='variants':
            position = length//2
            alternate = seed[:position]+('A' if seed[position]!='A' else 'C')+seed[position+1:]
            ref = _fixture(folder+'/01a_Predicted_Regulatory_Effects_reference.fasta',lambda:_fasta([('reference',seed)])) if folder else _fasta([('reference',seed)])
            alt = _fixture(folder+'/01b_Predicted_Regulatory_Effects_alternate.fasta',lambda:_fasta([('alternate',alternate)])) if folder else _fasta([('alternate',alternate)])
            fields = {'reference':ref,'alternate':alt}
            files = [_file(f'reference-{length}bp.fasta',ref),_file(f'alternate-{length}bp.fasta',alt)]
        elif workflow in {'predict','qc','allele-specific','prioritize','multi-allelic'}:
            names = {'predict':'06_Sequence_Prediction_three_records.fasta','qc':'07_Sequence_QC_exact_short_and_unknown.fasta',
                     'allele-specific':'02_Allele_Specific_Regulation_two_haplotypes.fasta',
                     'prioritize':'03_Variant_Prioritization_reference_and_two_SNVs.fasta',
                     'multi-allelic':'04_Multi_Allelic_reference_and_three_alternates.fasta'}
            def fallback():
                def mutate(offset,alternate_index=0):
                    offset%=length
                    bases=[base for base in 'ACGT' if base!=seed[offset]]
                    return seed[:offset]+bases[alternate_index]+seed[offset+1:]
                if workflow=='predict':
                    return _fasta([(f'sequence_{i+1}',_sequence(length,31+i)) for i in range(min(3,max(1,500000//length)))])
                if workflow=='qc':
                    return _fasta([('short',seed[:max(1,length//2)]),('full',seed),('unknown','N'*min(4,length)+seed[min(4,length):])])
                rows=[('reference',seed),('alternate_1',mutate(length//2))]
                if workflow=='allele-specific' and length>1:
                    value=rows[1][1];offset=(length//2+1)%length
                    rows[1]=('alternate_1',value[:offset]+('A' if value[offset]!='A' else 'C')+value[offset+1:])
                elif workflow!='allele-specific':
                    rows.append(('alternate_2',mutate(length//2+1,1 if length==1 else 0)))
                    if workflow=='multi-allelic':
                        combined=rows[1][1];offset=(length//2+1)%length
                        rows.append(('combined',combined[:offset]+('A' if combined[offset]!='A' else 'C')+combined[offset+1:]))
                return _fasta(rows)
            content = _fixture(folder+'/'+names[workflow],fallback) if folder else fallback()
            fields = {'fasta':content}
            if workflow=='qc':message='QC example loaded with a deliberately short record and unknown bases. Select Run analysis to inspect them.'
            files = [_file(f'{workflow}-{length}bp.fasta',content)]
        elif workflow=='batch-variants':
            from advanced_views import variant_example
            content = variant_example(length)
            fields = {'text':content,'shuffles':'0'}
            files = [_file(f'batch-variants-{length}bp.csv',content,'text/csv')]
        else:
            content = _fixture('new-workflows/'+('demo-200bp-seed.txt' if length==200 else 'trained-2114bp-seed.txt'),lambda:_fasta([('synthetic_seed',seed)])) if length in (200,2114) else _fasta([('synthetic_seed',seed)])
            center = length//2+1
            fields = {'text':content,'target':'0'}
            if workflow=='design':
                fields.update(iterations='2' if model=='demo' else '1',candidates='4' if model=='demo' else '2',off_target='',edit_start='',edit_end='')
            elif workflow=='mutagenesis':
                fields.update(start=str(center),end=str(min(length,center+(7 if model=='demo' else 1))))
            elif workflow=='attribution':
                start=max(1,center-50)
                fields.update(start=str(start),end=str(min(length,start+99)),randomized_check=False)
            elif workflow=='motif-simulation':
                if length<3:raise ValueError('This motif example needs a model window of at least three bases.')
                motif='TATAAA'[:min(length,6)]
                fields.update(motif=motif,positions=str(min(max(1,center),max(1,length-len(motif)+1))),orientation='both',partner='',gaps='0',controls=True)
            files = [_file(f'{workflow}-{length}bp.fasta',content)]
    return {'workflow':workflow,'fields':fields,'files':files,'message':message}


@examples.get('/api/examples/<workflow>')
def example_inputs(workflow):
    try:
        data = build_example(workflow,request.args)
    except ValueError as exc:
        return jsonify(error=str(exc)),400
    response = jsonify(data)
    response.headers['Cache-Control']='no-store'
    return response


@examples.get('/api/examples/<workflow>/download')
def download_inputs(workflow):
    try:
        data = build_example(workflow,request.args)
    except ValueError as exc:
        return jsonify(error=str(exc)),400
    files = data['files']
    buffer = io.BytesIO()
    if len(files)==1:
        item=files[0]
        buffer.write(item['content'].encode())
        name,media_type=item['name'],item['media_type']
    else:
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            for item in files:archive.writestr(item['name'],item['content'])
        name,media_type=workflow+'-examples.zip','application/zip'
    buffer.seek(0)
    response = send_file(buffer,as_attachment=True,download_name=name,mimetype=media_type,max_age=0)
    response.headers['Cache-Control']='no-store'
    return response
