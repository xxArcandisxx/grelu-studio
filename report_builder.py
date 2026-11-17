"""Build bounded tables and self-contained SVG charts from saved numerical results."""
import math


def chart(title, series):
    colors = ['#286c52', '#b66b32', '#3b76a0', '#8770ad']
    valid = [(label, [float(value) for value in values]) for label, values in series
             if values and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)]
    if not valid:
        return None
    minimum = min(min(values) for _, values in valid)
    maximum = max(max(values) for _, values in valid)
    span = maximum - minimum or 1
    plotted = []
    for index, (label, values) in enumerate(valid[:4]):
        points = [(70 + i / max(1, len(values)-1) * 500,
                   100 if minimum == maximum else 180 - (value-minimum)/span*155) for i,value in enumerate(values)]
        plotted.append({'label':label,'color':colors[index],
                        'points':' '.join(f'{x:.3f},{y:.3f}' for x,y in points),
                        'first_x':points[0][0], 'first_y':points[0][1], 'count':len(values)})
    return {'title':title,'series':plotted,'minimum':f'{minimum:.5g}','maximum':f'{maximum:.5g}'}


def build_report(saved):
    payload = saved['payload']
    result = payload.get('results', payload)
    view = saved.get('presentation') or {}
    tables = list(view.get('tables', []))
    plots = []
    metrics = view.get('metrics', [])
    def table(title, headings, rows):
        tables.append({'title':title,'headings':headings,'rows':list(rows)})
    if not tables:
        if 'predictions' in result:
            predictions = result['predictions']
            table('Prediction scores', ['Record','Track','Score','Reverse-complement score'],
                  ([row['id'],track,score,row.get('reverse_complement_scores',{}).get(track)]
                   for row in predictions for track,score in row['scores'].items()))
            if predictions and predictions[0].get('tracks'):
                track = next(iter(predictions[0]['tracks']))
                plot = chart('Predicted output bins · ' + track,
                             [(row['id'],row['tracks'][track]) for row in predictions[:4] if track in row.get('tracks',{})])
                if plot:
                    plots.append(plot)
        if 'designs' in result:
            for design in result['designs']:
                table('Optimization history', ['Iteration','Objective','Mutation'],
                      ([entry['iter'],entry['score'],str(entry.get('mutation') or 'Starting sequence')] for entry in design['history']))
                plot = chart('Accepted optimization steps', [('Objective',[entry['score'] for entry in design['history']])])
                if plot:
                    plots.append(plot)
        if 'ranked_variants' in result:
            table('Ranked variant effects', ['Rank','Variant','Track','Effect'],
                  ([row['rank'],row['variant_id'],row['top_task'],row['top_effect']] for row in result['ranked_variants']))
        if 'positions' in result:
            positions = result['positions']
            if positions and 'deltas' in positions[0]:
                table('Substitution effects', ['Position','Reference','Alternate','Delta'],
                      ([row['position'],row['reference'],base,delta] for row in positions for base,delta in row['deltas'].items()))
            elif positions and 'contribution' in positions[0]:
                table('Largest input contributions', ['Position','Reference','Contribution'],
                      ([row['position'],row['reference'],row['contribution']] for row in result.get('top_positions',positions)))
                plot = chart('Signed contributions in the display region', [('Contribution',[row['contribution'] for row in result.get('display_positions',positions[:200])])])
                if plot:
                    plots.append(plot)
        if 'metrics' in result and isinstance(result['metrics'],dict):
            table('Analysis metrics',['Metric','Value'],result['metrics'].items())
        if 'hits' in result and result['hits']:
            keys = ['record','motif','start','end','strand']
            if 'score' in result['hits'][0]:
                keys += ['score','relative_score']
            table('Motif locations', [key.replace('_',' ').title() for key in keys],
                  ([row.get(key) for key in keys] for row in result['hits']))
        if 'discovery' in result:
            table('Exploratory enriched words',['Word','Count','Background','Enrichment'],
                  ([row['kmer'],row['count'],row['background_count'],row['enrichment']] for row in result['discovery']))
        if 'history' in result and result['history'] and 'epoch' in result['history'][0]:
            table('Learning history',['Epoch','Training MSE','Validation MSE'],
                  ([row['epoch'],row['training_loss'],row['validation_loss']] for row in result['history']))
            plots.append(chart('Learning curves',[(name,[row[key] for row in result['history']])
                for name,key in [('Training','training_loss'),('Validation','validation_loss')]]))
    if tables and result.get('predictions') and not plots:
        predictions = result['predictions']
        if predictions[0].get('tracks'):
            track = next(iter(predictions[0]['tracks']))
            plotted = chart('Predicted output bins · ' + track,
                [(row['id'],row['tracks'][track]) for row in predictions[:4] if track in row.get('tracks',{})])
            if plotted:
                plots.append(plotted)
    provenance = payload.get('model_provenance') or payload.get('provenance')
    if not provenance and isinstance(payload.get('model'),dict):
        model_details = payload['model']
        provenance = model_details.get('provenance') or (model_details if 'model_id' in model_details else None)
    entries = provenance if isinstance(provenance,list) else [provenance] if isinstance(provenance,dict) else []
    if entries:
        table('Model provenance',['Model','Checkpoint SHA-256','Torch version'],
            ([entry.get('model_id','Unknown'),entry.get('checkpoint_sha256','No checkpoint file'),entry.get('torch_version','—')] for entry in entries))
    # Presentation-provided series are data, never executable SVG or HTML.
    for specification in view.get('plots',[]):
        plot = chart(specification['title'], [(entry['label'],entry['values']) for entry in specification['series']])
        if plot:
            plots.append(plot)
    for block in tables:
        block['total'] = len(block['rows'])
        block['rows'] = block['rows'][:200]
    evidence = [entry.get('message','') for entry in payload.get('evidence',[])]
    warnings = list(dict.fromkeys(payload.get('warnings',[])))
    if payload.get('demo') and not any('random' in warning.lower() and 'weight' in warning.lower() for warning in warnings):
        warnings.insert(0,'Demo CNN uses random weights; these outputs demonstrate the analysis workflow.')
    settings = payload.get('settings', {})
    if not settings and 'history' in payload:
        settings = {key:payload[key] for key in ('epochs','learning_rate','split_seed','target_track','data_source','optimizer_steps','best_epoch') if key in payload}
    model = payload.get('actual_model')
    if not model and isinstance(payload.get('model_provenance'),dict):
        model = payload['model_provenance'].get('model_id')
    if not model and isinstance(payload.get('model'),dict):
        model = payload['model'].get('provenance',{}).get('model_id')
    return {'tables':tables,'plots':[plot for plot in plots if plot], 'metrics':metrics,
            'evidence':evidence,'warnings':warnings,'settings':settings,'model':model,
            'method':result.get('method') or payload.get('method'), 'inputs':payload.get('inputs')}
