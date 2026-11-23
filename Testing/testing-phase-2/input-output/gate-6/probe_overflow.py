"""Minimal isolated legacy-JSON overflow reproduction, no model computation."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import verify_storage as support


def main(out):
    app,_=support.configure(out);client=app.test_client();client.get('/experiments')
    import experiment_store,lab_jobs
    with client:
        client.get('/experiments')
        identifier=experiment_store.save_experiment('legacy exponent fixture','qc',{'value':1},'/qc',{})
        with sqlite3.connect(out/'state/experiments.db') as db:db.execute('UPDATE experiments SET payload=?,metadata=? WHERE id=?',('{"value":1e309}','{}',identifier))
        response=client.get('/experiments/'+identifier+'/record.json')
        jobid='c'*32
        state=dict(id=jobid,owner=lab_jobs.owner(),state='complete',created=time.time(),updated=time.time(),history=[],result={'value':'exponent-marker'},worker_pid=os.getpid())
        path=lab_jobs.job_path(jobid,create=True)/'status.json'
        path.write_text(json.dumps(state).replace('"exponent-marker"','1e309'))
        job=client.get('/api/jobs/'+jobid)
        support.save(out/'observations.json',dict(experiment_status=response.status_code,experiment_body=response.text,job_status=job.status_code,job_body=job.text,
            expected='Visible errors; JSON 1e309 overflows float64 and cannot be emitted as Infinity.',model_computations=0))
        print(response.status_code,job.status_code,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
