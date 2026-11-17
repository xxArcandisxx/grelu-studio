"""Real-browser, file-only whole-website acceptance suite.

Run with a Python environment containing Playwright. The app uses .venv/bin/python.
No mocked inference, example buttons, or remote model downloads are used.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]


class WebsiteSuite:
    def __init__(self, browser, manifest, output, state, external_url=None):
        self.browser=browser
        self.external_url=external_url
        self.manifest=manifest
        self.settings=json.loads((ROOT/manifest['settings']).read_text())
        self.account=json.loads((ROOT/manifest['accounts']).read_text())
        self.output=output
        self.state=state
        self.process=None
        self.server_log=(output/'server.log').open('a')
        self.context=browser.new_context(viewport={'width':1440,'height':1000},locale='en-IN',timezone_id='Asia/Kolkata',accept_downloads=True)
        self.page=self.context.new_page()
        self.page.set_default_timeout(45000)
        self.page.set_default_navigation_timeout(180000)
        self.errors=[]
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.results=[]
        self.saved={}
        self.records={}
        self.models={}
        self.download_counter=0
        if external_url:self.url=external_url.rstrip('/')
        else:self.start_server()
        self.context.route('**/*',lambda route:route.continue_() if route.request.url.startswith(self.url+'/') or route.request.url.startswith(('file:','data:','about:')) else route.abort())

    def start_server(self, port=0):
        if self.external_url:
            raise RuntimeError('The test runner must never restart a user-owned server.')
        ready=self.state/'ready.json'
        if ready.exists(): ready.unlink()
        self.process=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'tests/serve_website_suite.py'),'--state',str(self.state),'--port',str(port)],
            cwd=ROOT,stdout=self.server_log,stderr=subprocess.STDOUT,env=os.environ.copy())
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Isolated server exited. See server.log.')
            if ready.exists():
                self.url=json.loads(ready.read_text())['url']
                return
            time.sleep(.1)
        raise TimeoutError('Isolated server did not start within 120 seconds.')

    def stop_server(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait(timeout=10)
        self.process=None

    def screenshot(self,path):
        self.page.evaluate('window.scrollTo(0,0)')
        self.page.screenshot(path=str(path),full_page=True)

    def close(self):
        self.context.close()
        self.stop_server()
        self.server_log.close()

    def goto(self,path):
        response=self.page.goto(self.url+path,wait_until='domcontentloaded')
        assert response and response.status==200,f'{path}: HTTP {response.status if response else None}'
        return response

    def fill(self,key,value):
        locator=self.page.locator('#'+key)
        if locator.count()==0:raise AssertionError('Missing control #'+key)
        tag=locator.evaluate('(element)=>element.tagName.toLowerCase()')
        if tag=='select':
            self.page.locator(f'#{key} option[value="{value}"]').wait_for(state='attached')
            locator.select_option(str(value))
        elif isinstance(value,bool):locator.set_checked(value)
        else:locator.fill(str(value))

    def upload(self,selector,path):
        file=ROOT/path
        assert file.is_file(),str(file)
        self.page.locator('#'+selector).set_input_files(str(file))
        editor={'reference-file':'reference','alternate-file':'alternate','sequence-file':'text','matrix-file':'matrices'}.get(selector)
        if selector=='file' and self.page.locator('#model').count() and self.page.locator('textarea').count()==1:
            editor='fasta' if self.page.locator('#fasta').count() else 'text'
        if editor:
            expected=file.read_text().lstrip('\ufeff')
            self.page.wait_for_function('([id,value])=>document.getElementById(id).value===value',arg=[editor,expected])

    def native_submit(self,name):
        with self.page.expect_navigation(wait_until='domcontentloaded',timeout=180000):
            self.page.get_by_role('button',name=name,exact=True).click()

    def download(self,selector,case_id,suffix=None):
        with self.page.expect_download(timeout=45000) as event:
            self.page.locator(selector).click()
        item=event.value
        self.download_counter+=1
        name=f'{case_id}-{self.download_counter:03d}-'+(suffix or item.suggested_filename)
        destination=self.output/'downloads'/name
        item.save_as(str(destination))
        assert destination.stat().st_size>0,'Empty download: '+name
        return destination

    def finite(self,value):
        if isinstance(value,float):assert math.isfinite(value),'Non-finite numerical result'
        elif isinstance(value,dict):
            for child in value.values():self.finite(child)
        elif isinstance(value,list):
            for child in value:self.finite(child)

    def capture(self,case_id,variant=False):
        if variant:
            data=json.loads(self.download('#download',case_id).read_text())
        else:
            self.page.locator('#run-record').wait_for(state='attached',timeout=180000)
            data=json.loads(self.page.locator('#run-record').text_content())
            downloaded=json.loads(self.download('#download-record',case_id).read_text())
            assert downloaded==data,'Downloaded JSON differs from rendered record'
        self.finite(data)
        run_id=data.get('saved_run_id')
        assert run_id,'Completed run was not saved'
        self.saved[case_id]=run_id
        self.records[case_id]=data
        response=self.context.request.get(self.url+'/experiments/'+run_id+'/record.json')
        assert response.status==200
        assert response.json()['id']==run_id
        report=self.context.request.get(self.url+'/experiments/'+run_id+'/report.html')
        assert report.status==200
        assert '/static/' not in report.text(),'Report depends on application assets'
        assert "default-src 'none'" in report.headers.get('content-security-policy','')
        (self.output/'downloads'/(case_id+'-report.html')).write_bytes(report.body())
        for selector in ('#download-csv','#exploration-csv'):
            if self.page.locator(selector).count():
                path=self.download(selector,case_id)
                rows=list(csv.reader(io.StringIO(path.read_text())))
                assert len(rows)>1,'CSV has no data rows'
                assert all(len(row)==len(rows[0]) for row in rows),'CSV field counts differ'
                if data.get('workflow')=='motif-simulation':
                    strand_column=rows[0].index('strand')
                    assert all(row[strand_column] in ('+','-') for row in rows[1:])
        if self.page.locator('#download-fasta').count():
            path=self.download('#download-fasta',case_id)
            sequence=''.join(line.strip() for line in path.read_text().splitlines() if not line.startswith('>'))
            assert sequence==data['results']['designs'][0]['designed_sequence']
        self.screenshot(self.output/'screenshots'/(case_id+'.png'))
        return data

    def run_case(self,case):
        started=time.monotonic()
        before=len(self.errors)
        item={'id':case['id'],'route':case['route'],'fixtures':list(case.get('files',{}).values())}
        if case.get('scenario'):item['fixtures'].append(case['scenario'])
        try:
            getattr(self,'case_'+case['kind'].replace('-','_'))(case)
            assert len(self.errors)==before,'Browser JavaScript error: '+'; '.join(self.errors[before:])
            item['status']='passed'
            print('PASS '+case['id'],flush=True)
        except Exception as exc:
            item.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            try:self.page.screenshot(path=str(self.output/'screenshots'/(case['id']+'-FAILED.png')),full_page=True)
            except Exception:pass
            print('FAIL '+case['id']+': '+str(exc)[:220],flush=True)
        item['seconds']=round(time.monotonic()-started,2)
        self.results.append(item)
        self.write_results()

    def write_results(self):
        summary={'started':self.started,'finished':datetime.now(timezone.utc).isoformat(),'fixture_manifest':'test-inputs/website-suite/manifest.json',
            'browser':'Chromium headless','real_http':True,'server_mode':'existing user server' if self.external_url else 'isolated server','run_example_buttons_used':False,'mocked_inference':False,
            'passed':sum(item['status']=='passed' for item in self.results),'failed':sum(item['status']=='failed' for item in self.results),
            'checks':self.results,'browser_errors':self.errors,'downloads':self.download_counter,
            'scope_limits':['Remote model downloads, other Model Zoo checkpoints, GPU execution, and production email delivery are not exercised. Cached GM12878 registration is tested with real weights.']}
        (self.output/'results.json').write_text(json.dumps(summary,indent=2))
        lines=['# Whole-website file-based test results','',f"Passed: **{summary['passed']}** · Failed: **{summary['failed']}**",'',
            'Real Chromium browser, real HTTP server, real uploaded fixture files, and real model computations. No Run example buttons or mocked inference.',
            '', '| Scenario | Result | Seconds |','|---|---|---|']
        for item in self.results:lines.append(f"| {item['id']} | {item['status']} | {item['seconds']} |")
        lines+=['','## Scope limits','',*['- '+note for note in summary['scope_limits']]]
        for item in self.results:
            if item['status']=='failed':lines+=['',f"## Failure: {item['id']}",'', '```',item['traceback'],'```']
        (self.output/'report.md').write_text('\n'.join(lines)+'\n')

    def case_variant(self,case):
        self.goto(case['route']);self.fill('model',case['model'])
        for control,path in case['files'].items():self.upload(control,path)
        self.page.get_by_role('button',name='Run comparison',exact=True).click()
        self.page.locator('#output').wait_for(state='visible',timeout=180000)
        data=self.capture(case['id'],variant=True)
        expected=200 if case['model']=='demo' else 2114
        assert len(data['inputs']['reference'])==expected
        assert sum(a!=b for a,b in zip(data['inputs']['reference'],data['inputs']['alternate']))==1
        assert self.page.locator('#plot polyline, #plot circle').count()>=2

    def case_analysis(self,case):
        self.goto(case['route']);self.fill('model',case['model'])
        for control,path in case['files'].items():self.upload(control,path)
        if case.get('shuffle'):self.fill('shuffle',True)
        self.native_submit('Run analysis')
        data=self.capture(case['id'])
        route=case['route']
        if route=='/predict':
            assert len(data['results']['predictions'])==3
            if case.get('shuffle'):assert all('shuffled_baseline' in row for row in data['results']['predictions'])
        elif route=='/qc':
            assert len(data['records'])==3
            assert any(not row['matches_model_length'] for row in data['records'])
            assert sum(row['unknown'] for row in data['records'])==4
        elif route=='/models':assert data['results']['model']['n_tasks']>=1
        elif route=='/allele-specific':assert all(row['differences']==2 for row in data['results']['comparisons'])
        elif route=='/prioritize':assert all(row['differences']==1 for row in data['results']['comparisons'])
        elif route=='/multi-allelic':assert {row['differences'] for row in data['results']['comparisons']}=={1,2}

    def case_advanced(self,case):
        self.goto(case['route']);self.fill('model',case['model'])
        for control,path in case['files'].items():self.upload(control,path)
        suffix='demo' if case['model']=='demo' else 'human'
        if case['route']=='/design':
            for control,value in self.settings['design_'+suffix].items():self.fill(control,value)
            name='Optimize sequence →'
        elif case['route']=='/mutagenesis':
            for control,value in self.settings['scan_'+suffix].items():self.fill(control,value)
            name='Run sequence scan →'
        elif case['route']=='/attribution':
            for control,value in self.settings['attribution_'+suffix].items():self.fill(control,value)
            name='Compute attribution →'
        else:
            self.fill('shuffles','5' if case['model']=='demo' else '0');name='Score variants →'
        self.native_submit(name)
        data=self.capture(case['id'])
        if case['route']=='/design':
            result=data['results']['designs'][0]
            assert result['score_after']>=result['score_before']
            assert self.page.locator('.history-plot polyline').count()==1
        elif case['route']=='/batch-variants':assert len(data['results']['ranked_variants'])==3
        elif case['route']=='/mutagenesis':assert data['results']['substitutions']==(24 if suffix=='demo' else 6)
        else:
            assert len(data['results']['matrix'])==4
            assert self.page.locator('.attribution-plot rect').count()>0

    def case_saturation(self,case):
        self.goto(case['route']);self.fill('locus',self.settings['saturation']['locus'])
        assert self.page.locator('#scan-button').is_enabled()
        self.page.locator('#scan-button').click()
        self.page.locator('#scan-results #run-record').wait_for(state='attached',timeout=180000)
        data=self.capture(case['id'])
        assert sum(len(row['deltas']) for row in data['positions'])==self.settings['saturation']['expected_substitutions']

    def case_research(self,case):
        self.goto(case['route'])
        if case.get('evaluation'):self.fill('kind',case['evaluation'])
        if case.get('input_format'):self.fill('kind',case['input_format'])
        for control,path in case['files'].items():self.upload(control,path)
        self.native_submit('Run analysis')
        data=self.capture(case['id'])
        if case['id']=='benchmark-regression':assert data['metrics']['MAE']==1 and data['metrics']['RMSE']==1
        elif case['id']=='benchmark-binary':assert data['metrics']['AUROC']==1
        elif case['id']=='track-analysis':assert data['pearson'][0][1]>0.99 and data['pearson'][0][2]<-.99
        elif data.get('analysis')=='dataset':
            assert data['summary']['records']==case['expected_records']
            assert sum(data['base_counts'].values())==data['summary']['total_bases']

    def case_motifs(self,case):
        self.goto(case['route']);self.fill('patterns',(ROOT/case['patterns']).read_text())
        self.fill('k','6')
        for control,path in case['files'].items():self.upload(control,path)
        self.native_submit('Scan & discover motifs →')
        data=self.capture(case['id'])
        assert data['hits'] and data['discovery']

    def case_pwm(self,case):
        self.goto(case['route'])
        for control,path in case['files'].items():self.upload(control,path)
        for control,value in self.settings['pwm'].items():self.fill(control,value)
        self.native_submit('Scan matrices →')
        data=self.capture(case['id'])
        assert len(data['hits'])>=2
        assert all(0<=row['relative_score']<=1 for row in data['hits'])
        if case['id']=='pwm-meme':
            first=self.records['pwm-jaspar']['hits']
            assert [(row['start'],row['strand']) for row in first]==[(row['start'],row['strand']) for row in data['hits']]

    def case_simulation(self,case):
        self.goto(case['route']);self.fill('model',case['model'])
        for control,path in case['files'].items():self.upload(control,path)
        self.page.locator('details').filter(has=self.page.locator('summary',has_text='Partner motif')).locator('summary').click()
        settings=self.settings['simulation_demo' if case['model']=='demo' else 'simulation_human']
        for control,value in settings.items():self.fill(control,value)
        self.native_submit('Run simulations →')
        data=self.capture(case['id'])
        assert data['results']['conditions']
        assert all(row['control_score'] is not None for row in data['results']['conditions'])
        length=len(data['inputs'][0]['sequence'])
        assert all(len(row['sequence'])==length for row in data['results']['conditions'])

    def case_comparison(self,case):
        self.goto(case['route']);self.fill('model_a','demo');self.fill('model_b','human')
        for control,path in case['files'].items():self.upload(control,path)
        for control,value in self.settings['comparison'].items():self.fill(control,value)
        self.native_submit('Compare models →')
        data=self.capture(case['id'])
        assert len(data['results']['comparisons'])==3
        assert all(row['model_length']==200 for row in data['results']['effective_inputs'][0])
        assert all(row['model_length']==2114 for row in data['results']['effective_inputs'][1])

    def training_job(self,case,model=None):
        self.goto(case['route'])
        if model:self.fill('model',model)
        config=self.settings['training' if case['kind']=='training' else 'fine_tuning']
        for control,value in config.items():self.fill(control,value)
        for control,path in case['files'].items():self.upload(control,path)
        self.native_submit('Start training →' if case['kind']=='training' else 'Start fine-tuning →')
        assert '/jobs/' in self.page.url
        self.page.wait_for_function("['complete','failed'].includes(document.getElementById('job-state').textContent)",timeout=240000)
        job_id=self.page.url.split('/')[-1]
        data=self.context.request.get(self.url+'/api/jobs/'+job_id).json()
        assert data['state']=='complete',data['message']
        assert len(data['history'])==int(config['epochs'])
        assert data['result']['optimizer_steps']>0
        assert self.page.locator('#learning-plot polyline').count()==2
        self.download('a[href$="/checkpoint"]',case['id'])
        self.saved[case['id']]=data['result']['saved_run_id']
        self.models[case['id']]=data['result']['model_key']
        self.screenshot(self.output/'screenshots'/(case['id']+'.png'))
        report=self.context.request.get(self.url+'/experiments/'+data['result']['saved_run_id']+'/report.html')
        assert report.status==200
        (self.output/'downloads'/(case['id']+'-report.html')).write_bytes(report.body())
        # Reuse each actual trained checkpoint through the normal upload page.
        self.goto('/predict');self.fill('model',data['result']['model_key'])
        directory='trained-model-2114bp' if model=='human' else 'quick-demo-200bp'
        self.upload('file',f'test-inputs/{directory}/06_Sequence_Prediction_three_records.fasta')
        self.native_submit('Run analysis')
        prediction=self.capture(case['id']+'-prediction')
        assert len(prediction['results']['predictions'])==3

    def case_training(self,case):self.training_job(case)
    def case_fine_tuning_local(self,case):self.training_job(case,self.models['training'])
    def case_fine_tuning_human(self,case):self.training_job(case,'human')

    def login(self,password):
        self.goto('/account/login');self.fill('email',self.account['email']);self.fill('password',password)
        self.native_submit('Sign in →')

    def case_accounts(self,case):
        self.goto('/account/register')
        for key in ('name','email','password'):self.fill(key,self.account[key])
        self.native_submit('Create account →')
        activation=self.page.locator('a[href^="/account/activate?token="]').get_attribute('href')
        self.login(self.account['password'])
        assert 'Activate your account' in self.page.locator('.error').inner_text()
        self.goto(activation);self.native_submit('Activate my account →')
        assert 'activated' in self.page.locator('#account-title').inner_text()
        self.login(self.account['password'])
        assert self.page.url.endswith('/account')
        known=self.saved.get('variants-demo') or next(iter(self.saved.values()))
        assert self.context.request.get(self.url+'/experiments/'+known).status==200
        self.goto('/account/forgot');self.fill('email',self.account['email']);self.native_submit('Create recovery link →')
        reset=self.page.locator('a[href^="/account/reset?token="]').get_attribute('href')
        self.goto(reset);self.fill('password',self.account['reset_password']);self.fill('confirm_password',self.account['reset_password'])
        self.native_submit('Update password →')
        self.login(self.account['password'])
        assert 'incorrect' in self.page.locator('.error').inner_text()
        self.login(self.account['reset_password'])
        assert self.page.url.endswith('/account')
        self.screenshot(self.output/'screenshots'/'accounts.png')
        self.native_submit('Sign out')
        assert self.context.request.get(self.url+'/experiments/'+known).status==404
        self.login(self.account['reset_password'])
        assert self.context.request.get(self.url+'/experiments/'+known).status==200

    def case_model_zoo(self,case):
        self.goto('/model-zoo')
        scenario=self.settings['model_zoo']
        form=self.page.locator('form').filter(has=self.page.locator('input[name="repo"][value="'+scenario['repo']+'"]'))
        form.locator('input[name="revision"]').fill(scenario['revision'])
        with self.page.expect_navigation(wait_until='domcontentloaded'):
            form.get_by_role('button',name='Use cached checkpoint',exact=True).click()
        self.page.wait_for_function("['complete','failed'].includes(document.getElementById('job-state').textContent)",timeout=180000)
        job_id=self.page.url.split('/')[-1]
        job=self.context.request.get(self.url+'/api/jobs/'+job_id).json()
        assert job['state']=='complete',job['message']
        assert job['result']['revision']==scenario['revision'] and job['result']['source']=='local_cache'
        key=job['result']['model_key']
        self.goto('/models');self.fill('model',key);self.native_submit('Run analysis');self.capture('model-zoo-inspection')
        self.goto('/model-zoo')
        card=self.page.locator('article.model-card').filter(has=self.page.get_by_role('heading',name=scenario['repo'],exact=True))
        with self.page.expect_navigation(wait_until='domcontentloaded'):
            card.get_by_role('button',name='Remove from workspace',exact=True).click()
        assert self.context.request.get(self.url+'/api/model-info?model='+key).status==400

    def case_history(self,case):
        if self.external_url:
            return self.case_history_light(case)
        self.goto('/experiments')
        self.fill('q',self.settings['history']['query']);self.fill('workflow',self.settings['history']['workflow'])
        with self.page.expect_navigation(wait_until='domcontentloaded'):
            self.page.get_by_role('button',name='Filter',exact=True).click()
        assert self.page.locator('.experiment-card').count()>0
        run_id=self.saved['variants-demo']
        self.goto('/experiments/'+run_id)
        report=self.download('a[href$="/report.html"]',case['id'])
        self.download('a[href$="/record.json"]',case['id'])
        standalone=self.browser.new_context(offline=True,java_script_enabled=False)
        tab=standalone.new_page();tab.goto(report.resolve().as_uri(),wait_until='load')
        assert tab.locator('h1').inner_text()=='Predicted regulatory effects'
        assert tab.locator('svg').count()>0
        tab.pdf(path=str(self.output/'downloads'/'standalone-report.pdf'),print_background=True)
        assert (self.output/'downloads'/'standalone-report.pdf').stat().st_size>1000
        standalone.close()
        before=self.context.request.get(self.url+'/experiments/'+run_id+'/record.json').json()
        self.native_submit('Run a new experiment →')
        assert self.page.url.split('/')[-1]!=run_id
        after=self.context.request.get(self.url+'/experiments/'+run_id+'/record.json').json()
        assert before==after
        other=self.browser.new_context()
        assert other.request.get(self.url+'/experiments/'+run_id).status==404
        other.close()
        self.goto('/experiments/'+self.saved['pwm-meme'])
        self.native_submit('Delete this saved record')
        assert self.context.request.get(self.url+'/experiments/'+self.saved['pwm-meme']).status==404
        port=urlparse(self.url).port
        self.stop_server();self.start_server(port)
        self.goto('/experiments/'+run_id)
        assert self.page.locator('h1').inner_text()=='Predicted regulatory effects'
        self.screenshot(self.output/'screenshots'/'history-after-restart.png')
        self.goto('/models');self.fill('model',self.models['training']);self.native_submit('Run analysis')
        assert self.page.locator('#run-record').count()==1

    def case_history_light(self,case):
        run_id=self.saved['pwm-meme']
        self.goto('/experiments')
        self.fill('q','PWM');self.fill('workflow','pwm-scan')
        with self.page.expect_navigation(wait_until='domcontentloaded'):
            self.page.get_by_role('button',name='Filter',exact=True).click()
        assert self.page.locator('.experiment-card').count()>0
        self.goto('/experiments/'+run_id)
        report=self.download('a[href$="/report.html"]',case['id'])
        self.download('a[href$="/record.json"]',case['id'])
        offline=self.browser.new_context(offline=True,java_script_enabled=False)
        try:
            tab=offline.new_page();tab.goto(report.resolve().as_uri(),wait_until='load')
            assert 'PWM motif scanning' in tab.locator('h1').inner_text()
            tab.pdf(path=str(self.output/'downloads'/'standalone-pwm-report.pdf'),print_background=True)
            original=ROOT/'test-results/whole-website/downloads/variants-demo-report.html'
            if original.exists():
                tab.goto(original.resolve().as_uri(),wait_until='load')
                assert tab.locator('svg').count()>0
                tab.pdf(path=str(self.output/'downloads'/'standalone-signal-report.pdf'),print_background=True)
        finally:offline.close()
        before=self.context.request.get(self.url+'/experiments/'+run_id+'/record.json').json()
        self.native_submit('Run a new experiment →')
        repeated=self.page.url.split('/')[-1]
        assert repeated!=run_id
        assert self.context.request.get(self.url+'/experiments/'+run_id+'/record.json').json()==before
        other=self.browser.new_context()
        try:assert other.request.get(self.url+'/experiments/'+run_id).status==404
        finally:other.close()
        self.native_submit('Delete this saved record')
        assert self.context.request.get(self.url+'/experiments/'+repeated).status==404
        self.goto('/experiments/'+run_id)
        self.screenshot(self.output/'screenshots'/'history-reports.png')

    def case_mobile(self,case):
        self.page.set_viewport_size(self.settings['mobile'])
        self.goto('/')
        self.page.locator('.menu-toggle').click()
        self.page.locator('body.navigation-open').wait_for()
        group=self.page.locator('.nav-group').filter(has=self.page.locator('summary',has_text='Interpretation & design'))
        group.locator('summary').click()
        group.get_by_role('link',name='PWM motif scanning').click()
        self.page.wait_for_url(self.url+'/pwm-scan')
        assert not self.page.locator('body').evaluate("body=>body.classList.contains('navigation-open')")
        for path in ('/pwm-scan','/predict','/model-zoo','/experiments'):
            self.goto(path)
            assert self.page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),f'Horizontal mobile overflow: {path}'
            self.screenshot(self.output/'screenshots'/('mobile-'+path.strip('/')+'.png'))
        self.page.set_viewport_size({'width':1440,'height':1000})

    def case_negative(self,case):
        self.goto('/predict');self.fill('model','demo');self.upload('file',case['files']['dna'])
        assert not self.page.locator('#fasta').evaluate('(element)=>element.checkValidity()')
        self.goto('/predict');self.fill('model','human')
        self.upload('file',self.manifest['cases'][4]['files']['file'])
        assert not self.page.locator('#fasta').evaluate('(element)=>element.checkValidity()')
        self.goto('/batch-variants');self.upload('file',case['files']['csv'])
        self.native_submit('Score variants →');assert self.page.locator('[role="alert"]').count()>0
        self.goto('/pwm-scan');self.upload('sequence-file','test-inputs/new-workflows/pwm-regions.txt');self.upload('matrix-file',case['files']['pwm'])
        self.native_submit('Scan matrices →');assert self.page.locator('.form-error').count()>0
        self.goto('/training');self.upload('file',case['files']['training'])
        self.native_submit('Start training →');assert self.page.locator('.form-error').count()>0
        self.screenshot(self.output/'screenshots'/'negative-inputs.png')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--url',help='Use an existing server; never restart it.')
    parser.add_argument('--account-fixture',help='Synthetic credentials JSON for an existing-server check.')
    parser.add_argument('--browser-cache')
    parser.add_argument('--library-path')
    parser.add_argument('--only',help='Comma-separated scenario IDs; dependencies still need the complete run.')
    args=parser.parse_args()
    if args.browser_cache:os.environ['PLAYWRIGHT_BROWSERS_PATH']=args.browser_cache
    if args.library_path:os.environ['LD_LIBRARY_PATH']=args.library_path+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    manifest=json.loads((ROOT/'test-inputs/website-suite/manifest.json').read_text())
    output=Path(args.output).resolve() if args.output else ROOT/'test-results'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    (output/'screenshots').mkdir(parents=True,exist_ok=True)
    (output/'downloads').mkdir(exist_ok=True)
    from playwright.sync_api import sync_playwright
    suite=None
    try:
        with tempfile.TemporaryDirectory(prefix='grelu-website-suite-') as state:
            with sync_playwright() as playwright:
                browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
                try:
                    suite=WebsiteSuite(browser,manifest,output,Path(state),external_url=args.url)
                    if args.account_fixture:
                        suite.account=json.loads(Path(args.account_fixture).read_text())
                        for case in manifest['cases']:
                            if case['kind']=='accounts':case['scenario']=str(Path(args.account_fixture).resolve().relative_to(ROOT))
                    suite.started=datetime.now(timezone.utc).isoformat()
                    selected=set(args.only.split(',')) if args.only else None
                    for case in manifest['cases']:
                        if not selected or case['id'] in selected:suite.run_case(case)
                    suite.write_results()
                    print(json.dumps({'output':str(output),'passed':sum(row['status']=='passed' for row in suite.results),
                        'failed':sum(row['status']=='failed' for row in suite.results)}),flush=True)
                finally:
                    if suite:suite.close()
                    browser.close()
    finally:
        if suite and any(row['status']=='failed' for row in suite.results):sys.exit(1)


if __name__=='__main__':
    main()
