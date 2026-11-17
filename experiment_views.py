"""Saved-run history, reproducible reruns, and standalone report exports."""
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Blueprint, abort, current_app, jsonify, make_response, redirect, render_template, request

from experiment_store import REPEAT_PATHS, delete_experiment, experiment, list_experiments
from report_builder import build_report

experiments = Blueprint('experiments',__name__)


@experiments.app_template_filter('display_time')
def display_time(value):
    timestamp = datetime.fromisoformat(value)
    return timestamp.astimezone(ZoneInfo(current_app.config.get('DISPLAY_TIMEZONE','Asia/Kolkata'))).strftime('%d %b %Y · %I:%M %p %Z')



def owned(experiment_id):
    saved = experiment(experiment_id)
    if not saved:
        abort(404)
    return saved


@experiments.get('/experiments')
def history():
    try:
        page = max(1,min(100000,int(request.args.get('page','1'))))
    except ValueError:
        page = 1
    query, workflow = request.args.get('q','')[:150],request.args.get('workflow','')[:80]
    rows,total,workflows = list_experiments(query,workflow,page)
    response = make_response(render_template('experiments.html',title='Experiment history',mode='experiments',
        runs=rows,total=total,workflows=workflows,page=page,query=query,workflow=workflow))
    response.headers['Cache-Control']='no-store'
    return response


@experiments.get('/experiments/<experiment_id>')
def detail(experiment_id):
    from studio_web import model_choices
    saved = owned(experiment_id)
    response = make_response(render_template('experiment.html',title=saved['title'],mode='experiments',
        saved=saved,report=build_report(saved),choices=model_choices(),can_repeat=saved.get('path') in REPEAT_PATHS))
    response.headers['Cache-Control']='no-store'
    return response


@experiments.get('/experiments/<experiment_id>/record.json')
def record(experiment_id):
    saved = owned(experiment_id)
    response = jsonify({key:saved[key] for key in ('id','title','workflow','created','payload','form')})
    response.headers['Content-Disposition']=f'attachment; filename="grelu-{saved["workflow"]}-{experiment_id[:8]}.json"'
    response.headers['Cache-Control']='no-store'
    return response


@experiments.get('/experiments/<experiment_id>/report.html')
def report(experiment_id):
    saved = owned(experiment_id)
    response = make_response(render_template('report.html',saved=saved,report=build_report(saved)))
    response.headers.update({'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store',
        'Content-Disposition':f'attachment; filename="grelu-report-{experiment_id[:8]}.html"',
        'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'"})
    return response


@experiments.post('/experiments/<experiment_id>/delete')
def delete(experiment_id):
    from lab_views import check_csrf
    check_csrf()
    owned(experiment_id)
    delete_experiment(experiment_id)
    return redirect('/experiments')
