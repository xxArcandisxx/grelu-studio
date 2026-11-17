"""Install optional browser QA tools locally, without changing app/system packages."""
import os
from pathlib import Path
import subprocess
import venv

ROOT=Path(__file__).resolve().parents[1]
tools=ROOT/'.test-tools'
tools.mkdir(exist_ok=True)
python=tools/'venv/bin/python'
if not python.exists():venv.EnvBuilder(with_pip=True).create(tools/'venv')
subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--no-cache-dir','playwright==1.65.2'],check=True)
environment=os.environ.copy();environment['PLAYWRIGHT_BROWSERS_PATH']=str(tools/'browsers')
subprocess.run([str(python),'-m','playwright','install','chromium','--only-shell'],env=environment,check=True)
packages=tools/'runtime-downloads';packages.mkdir(exist_ok=True)
needed=['libnspr4','libnss3','libasound2t64']
missing=[name for name in needed if not list(packages.glob(name+'_*.deb'))]
if missing:subprocess.run(['apt-get','download',*missing],cwd=packages,check=True)
for package in packages.glob('*.deb'):
 subprocess.run(['dpkg-deb','-x',str(package),str(tools/'libraries')],check=True)
print('Project-local browser QA tools are ready.',flush=True)
