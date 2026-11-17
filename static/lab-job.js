(() => {
  const id = document.getElementById('job-panel').dataset.job;
  const labels = {
    initial_validation_mse:'Initial validation MSE',best_validation_mse:'Best validation MSE',
    mean_label_baseline_mse:'Mean-label baseline MSE',best_epoch:'Best checkpoint epoch',
    training_records:'Training records',validation_records:'Validation records',
    epochs:'Completed epochs',batch_size:'Batch size',optimizer_steps:'Optimizer updates',input_length:'Input length',tracks:'Output tracks',
    repository:'Repository',revision:'Pinned revision',target_track:'Target track',data_source:'Dataset',
  };
  function drawHistory(history) {
    const plot = document.getElementById('learning-plot');
    plot.replaceChildren();
    if (!history.length) return;
    const values = history.flatMap(row => [row.training_loss,row.validation_loss]);
    const minimum = Math.min(...values),maximum = Math.max(...values),span = maximum-minimum || 1;
    const add = (tag,attributes,text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg',tag);
      for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,value);
      if(text !== undefined) node.textContent=text;
      plot.append(node);
    };
    for(const value of new Set([minimum,maximum])) {
      const y=maximum === minimum ? 100 : 180-(value-minimum)/span*150;
      add('line',{x1:75,x2:580,y1:y,y2:y,stroke:'#e0e5de'});
      add('text',{x:2,y:y+4,fill:'#6b7b73','font-size':11},value.toPrecision(4));
    }
    for(const [key,color] of [['training_loss','#286c52'],['validation_loss','#b66b32']]) {
      const points=history.map((row,index)=>[75+index/Math.max(1,history.length-1)*505,maximum===minimum?100:180-(row[key]-minimum)/span*150]);
      add('polyline',{points:points.map(pair=>pair.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2});
      points.forEach(([x,y])=>add('circle',{cx:x,cy:y,r:3,fill:color}));
    }
    add('text',{x:325,y:212,'text-anchor':'middle',fill:'#6b7b73','font-size':11},'Epoch');
  }
  async function refresh() {
    try {
      const response=await fetch('/api/jobs/'+id,{cache:'no-store'});
      if(!response.ok) throw new Error('Could not read run status. Reload this page to reconnect.');
      const job=await response.json();
      document.getElementById('job-state').textContent=job.state;
      document.getElementById('job-message').textContent=job.message;
      const section=document.getElementById('learning-section');
      section.hidden=job.kind==='Model installation';
      const history=document.getElementById('job-history');history.replaceChildren();
      for(const epoch of job.history) {
        const tr=document.createElement('tr');
        for(const value of [epoch.epoch,epoch.training_loss.toPrecision(5),epoch.validation_loss.toPrecision(5)]) {
          const td=document.createElement('td');td.textContent=value;tr.append(td);
        }
        history.append(tr);
      }
      drawHistory(job.history);
      if(job.state==='complete') {
        const metrics=document.getElementById('job-metrics');metrics.replaceChildren();
        for(const [key,label] of Object.entries(labels)) {
          if(job.result[key] == null) continue;
          const p=document.createElement('p');p.className='model-help';
          p.textContent=label+': '+(typeof job.result[key]==='number'?Number(job.result[key].toPrecision(6)):job.result[key]);
          metrics.append(p);
        }
        const actions=document.getElementById('job-actions');actions.replaceChildren();
        const link=(href,text,quiet=false)=>{const a=document.createElement('a');a.href=href;a.className=quiet?'button quiet':'button';a.textContent=text;actions.append(a);};
        link('/model-zoo','Open model collection →');
        if(job.result.saved_run_id) link('/experiments/'+job.result.saved_run_id,'Saved results & report →',true);
        if(job.result.checkpoint) link('/jobs/'+id+'/checkpoint','Download checkpoint',true);
        const json=document.createElement('button');json.className='quiet';json.textContent='Download run JSON';
        json.addEventListener('click',()=>{
          const url=URL.createObjectURL(new Blob([JSON.stringify(job,null,2)],{type:'application/json'}));
          const a=document.createElement('a');a.href=url;a.download='grelu-run-'+id.slice(0,8)+'.json';a.click();
          setTimeout(()=>URL.revokeObjectURL(url),1000);
        });actions.append(json);
      } else if(job.state!=='failed') setTimeout(refresh,1500);
    } catch(error) { document.getElementById('job-error').textContent=error.message; }
  }
  refresh();
})();
