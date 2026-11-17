"""Serve an isolated real-HTTP app for the browser suite; never touch live user data."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--state',required=True)
    parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    state=Path(args.state)
    state.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'test-inputs/website-suite/manifest.json').read_text())
    settings=json.loads((ROOT/manifest['settings']).read_text())
    model=settings['model_zoo']
    from huggingface_hub import hf_hub_download
    checkpoint=hf_hub_download(model['repo'],model['filename'],revision=model['revision'],local_files_only=True)
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        GRELU_TRAIN_BATCH_SIZE='1',HF_HUB_OFFLINE='1',GRELU_HUMAN_CHECKPOINT=checkpoint,GRELU_HUMAN_INPUT_LENGTH='2114',
        GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase',GRELU_SECRET_KEY='isolated-website-acceptance-key')
    from studio_web import app
    app.config.update(TESTING=False,ACCOUNT_DATABASE=str(state/'accounts.db'),
        HISTORY_DATABASE=str(state/'history.db'),LAB_ROOT=str(state/'lab'),
        ACCOUNT_LOCAL_PREVIEW=True,SESSION_COOKIE_SECURE=False)
    from werkzeug.serving import make_server
    server=make_server('127.0.0.1',args.port,app,threaded=True)
    url=f'http://127.0.0.1:{server.server_port}'
    (state/'ready.json').write_text(json.dumps({'url':url,'pid':os.getpid()}))
    print(url,flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
