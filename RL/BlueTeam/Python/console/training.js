'use strict';

// Presentation and API orchestration only. Training, metric aggregation and model
// validation live behind the Training Manager in the native environment.
window.TrainingWorkbench = (() => {
 const byId = id => document.getElementById(id);
 const colors = ['#6ce8c8', '#70b7ff', '#f4cf78', '#e8a5eb'];
 const activeStates = new Set(['starting', 'running', 'evaluating', 'stopping']);
 const state = {api:null,onLoadModel:null,visible:false,options:null,runs:[],selected:null,overlays:new Map(),overlayRequests:new Map(),activeId:null,busy:false,timer:null,polling:false,pollError:null,selectionSequence:0,lastHistory:0,receivedAt:0,chartFrame:null};
 const finite = value => typeof value === 'number' && Number.isFinite(value);
 const number = (value, precision=2) => finite(value) ? value.toLocaleString(undefined, {maximumFractionDigits:precision}) : '—';
 const seconds = value => finite(value) ? `${number(value)} s` : 'Unavailable';
 const percent = value => finite(value) ? `${number(value * 100, 1)}%` : 'Unavailable';
 const duration = value => {
  if(!finite(value)) return '—';
  const total = Math.max(0, Math.floor(value)), hours = Math.floor(total / 3600), minutes = Math.floor(total % 3600 / 60);
  return hours ? `${hours}h ${minutes}m ${total % 60}s` : `${minutes}m ${total % 60}s`;
 };
 function node(tag, text, className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
 function text(id, value){byId(id).textContent=value;}
 function error(message){byId('training-error').hidden=!message;text('training-error',message||'');}
 function unwrap(value){return value?.run || value;}
 function runName(run){return run?.runName || run?.config?.runName || run?.id || 'Untitled run';}
 function algorithmLabel(id){return state.options?.algorithms?.find(row=>row.id===id)?.label || id || '—';}
 function checkpointChoices(run){
  const source=run?.checkpoints;
  if(Array.isArray(source))return source.map(row=>typeof row==='string'?{id:row,label:row}:{...row,id:row.id||row.name||row.kind}).filter(row=>['best','latest','final'].includes(row.id));
  return Object.entries(source||{}).filter(([id,value])=>['best','latest','final'].includes(id)&&Boolean(value)).map(([id,value])=>({id,label:id,...(typeof value==='object'?value:{})}));
 }
 function renderCheckpointNote(){
  const checkpoint=checkpointChoices(state.selected).find(row=>row.id===byId('training-checkpoint').value);
  if(!checkpoint){text('training-model-status','A model becomes available once a checkpoint has been saved.');return;}
  const details=[`${checkpoint.label||checkpoint.id} checkpoint`];
  if(finite(checkpoint.episode))details.push(`episode ${checkpoint.episode}`);
  if(finite(checkpoint.score))details.push(`fixed-panel mean warning ${seconds(checkpoint.score)}`);
  text('training-model-status',`${details.join(' · ')}.${checkpoint.episode===0?' Episode 0 is the untrained policy.':''} Checkpoint selection uses the fixed evaluation panel, separately from the best training episode shown above. Loading checks map, sensor and model compatibility.`);
 }
 function hasActiveRun(){return Boolean(state.activeId);}
 function updateControls(){
  const unavailable=!state.options?.environment?.available;
  byId('training-fields').disabled=state.busy||hasActiveRun();
  byId('training-start').disabled=state.busy||hasActiveRun()||unavailable||!state.options?.algorithms?.length;
  byId('training-stop').disabled=state.busy||!hasActiveRun()||state.runs.find(row=>row.id===state.activeId)?.status==='stopping'||(state.selected?.id===state.activeId&&state.selected?.status==='stopping');
  byId('training-stop').textContent=state.selected?.status==='stopping'?'Stopping after safe cleanup…':'Stop Training';
  const checkpoints=checkpointChoices(state.selected);
  byId('training-checkpoint').disabled=state.busy||!checkpoints.length||hasActiveRun();
  byId('training-load-model').disabled=state.busy||!checkpoints.length||hasActiveRun();
  byId('training-refresh').disabled=state.busy||hasActiveRun();
  byId('training-history-refresh').disabled=state.busy;
 }
 function renderOptions(){
  const options=state.options,environment=options.environment||{};
  byId('training-environment').dataset.ready=String(Boolean(environment.available));
  text('training-environment-text',`${environment.detail || (environment.available?'Native training environment is ready.':'Unreal is unavailable. Launch the native application with -IstanaBlueLive, then check the connection.')}${finite(environment.maxSensors)?` This scene permits at most ${environment.maxSensors} sensors, within the selected budget.`:''}`);
  const chosen=byId('training-algorithm').value;
  byId('training-algorithm').replaceChildren(...(options.algorithms||[]).map(row=>{const el=node('option',row.label);el.value=row.id;return el;}));
  if((options.algorithms||[]).some(row=>row.id===chosen))byId('training-algorithm').value=chosen;
  text('training-warning-definition',environment.warningDefinition || 'Warning-time definition is supplied by the native environment.');
  text('training-success-definition',environment.successDefinition || 'Success rate is the fraction of threats confirmed with at least the native defence lead time remaining before protected-zone entry.');
  text('training-checkpoint-criterion',options.checkpointCriterion || 'Checkpoint selection criterion is supplied by the Training Manager.');
  text('training-unavailable-algorithms',typeof options.unavailableAlgorithms==='string'?options.unavailableAlgorithms:'');
  text('training-warning-note',`${environment.warningDefinition || 'Warning time follows the existing native definition.'} The agent optimizes mean per-drone warning. Native episode reward is a separate diagnostic score.`);
  const oldChoices=new Map([...byId('training-catalogue').querySelectorAll('input')].map(input=>[input.value,input.checked]));
  byId('training-catalogue').replaceChildren();
  for(const sensor of options.catalogue||[]){
   const label=node('label',undefined,'training-sensor'),input=document.createElement('input'),body=node('span');
   input.type='checkbox';input.value=sensor.id;input.checked=oldChoices.has(sensor.id)?oldChoices.get(sensor.id):sensor.enabled!==false;
   input.setAttribute('aria-label',`Enable ${sensor.label||sensor.id}`);
   const details=[`${number(sensor.cost)} budget units`];
   const spec=sensor.manufacturer_specifications||sensor.manufacturerSpecifications||{},geometry=sensor.calculated_geometry||sensor.calculatedGeometry||{};
   const horizontal=sensor.horizontalFovDeg??sensor.horizontal_fov_deg??spec.horizontal_fov_deg,vertical=sensor.verticalFovDeg??geometry.vertical_fov_deg;
   if(finite(horizontal))details.push(`${number(horizontal,1)}° H${finite(vertical)?` / ${number(vertical,1)}° V`:''} FOV`);
   if(sensor.directional)details.push('directional · yaw / pitch');
   const ranges=Object.values(sensor.ranges||{}).filter(finite);if(ranges.length)details.push(`up to ${number(Math.max(...ranges),0)} m simulation range`);
   body.append(node('strong',sensor.label||sensor.id),node('small',details.join(' · ')));label.append(input,body);byId('training-catalogue').append(label);
  }
  if(!options.catalogue?.length)byId('training-catalogue').append(node('p','Sensor profiles load from the native environment when it is available.','helper'));
  text('training-sensor-count',`${options.catalogue?.length||0} profiles`);
  renderAlgorithmNote();updateControls();
 }
 function renderAlgorithmNote(){
  const algorithm=state.options?.algorithms?.find(row=>row.id===byId('training-algorithm').value);
  text('training-algorithm-note',algorithm?.description || (algorithm?.id==='ppo'?'Proximal Policy Optimization with clipped policy updates and a value function.':'Existing native warning-time policy gradient trainer.'));
 }
 async function refreshOptions(initial=false){
  try{
   const options=await state.api('/api/training/options');state.options=options;
   if(initial){for(const [id,key] of [['training-episodes','episodes'],['training-budget','budget'],['training-checkpoint-frequency','checkpointFrequency']])if(finite(options.defaults?.[key]))byId(id).value=options.defaults[key];}
   renderOptions();
  }catch(e){error(`Cannot check the native training environment: ${e.message}`);}
 }
 async function refreshHistory(){
  const response=await state.api('/api/training/runs');state.runs=response.runs||[];state.lastHistory=Date.now();
  state.activeId=state.runs.find(row=>activeStates.has(row.status))?.id||null;
  if(state.selected&&activeStates.has(state.selected.status)&&!state.runs.some(row=>row.id===state.selected.id))state.activeId=state.selected.id;
  renderHistory();updateControls();
 }
 async function selectRun(id){
  const sequence=++state.selectionSequence;error('');
  try{
   const run=unwrap(await state.api(`/api/training/run/${encodeURIComponent(id)}`));
   if(sequence!==state.selectionSequence)return;
   state.selected=run;state.receivedAt=Date.now();state.overlays.delete(run.id);state.overlayRequests.delete(run.id);
   if(activeStates.has(run.status))state.activeId=run.id;
   else if(state.activeId===run.id)state.activeId=null;
   renderRun();renderHistory();
  }catch(e){if(sequence===state.selectionSequence)error(`Cannot open experiment: ${e.message}`);}
 }
 async function refreshSelected(){
  const id=state.selected?.id,sequence=state.selectionSequence;if(!id)return;
  const run=unwrap(await state.api(`/api/training/run/${encodeURIComponent(id)}`));
  if(sequence!==state.selectionSequence||state.selected?.id!==id)return;
  state.selected=run;state.receivedAt=Date.now();
  if(activeStates.has(run.status))state.activeId=run.id;else if(state.activeId===run.id)state.activeId=null;
  renderRun();renderHistory();
 }
 function latestMetric(run){return run?.latestMetric || run?.metrics?.at(-1) || {};}
 function addStatistic(parent,label,value,unavailable=false){const wrap=node('div'),term=node('dt',label),definition=node('dd',value,unavailable?'unavailable':undefined);wrap.append(term,definition);parent.append(wrap);}
 function renderRun(){
  const run=state.selected,latest=latestMetric(run),summary=run?.summary||{},config=run?.config||{},windowSize=run?.movingAverageWindow||20;
  // A page reload may discover an already-running server job. Show its actual
  // locked configuration, rather than leaving unrelated form defaults visible.
  if(run&&activeStates.has(run.status)){
   byId('training-name').value=runName(run);
   for(const [id,key] of [['training-episodes','episodes'],['training-budget','budget'],['training-seed','seed'],['training-checkpoint-frequency','checkpointFrequency']])if(finite(config[key]))byId(id).value=config[key];
   byId('training-algorithm').value=run.algorithm||config.algorithm;renderAlgorithmNote();
   if(Array.isArray(config.enabledSensorIds))for(const input of byId('training-catalogue').querySelectorAll('input'))input.checked=config.enabledSensorIds.includes(input.value);
  }
  text('training-run-name',run?runName(run):'Your next experiment starts here');
  text('training-run-meta',run?`${algorithmLabel(run.algorithm||config.algorithm)} · budget ${number(config.budget)} · seed ${config.seed??'saved with configuration'}${run.phase?` · ${run.phase.replaceAll('_',' ')}`:''}`:'Configure a run to learn from real simulator episodes.');
  text('training-state',(run?.status||'ready').toUpperCase());byId('training-state').dataset.status=run?.status||'ready';
  const episode=run?.episode||0,total=run?.totalEpisodes||config.episodes||0;
  text('training-episode-progress',`${episode.toLocaleString()} / ${total.toLocaleString()} episodes`);
  byId('training-progress').max=Math.max(1,total);byId('training-progress').value=episode;
  renderElapsed();
  text('training-current-reward',number(latest.reward));
  text('training-current-outcome',!finite(latest.successRate)?'Awaiting episode':latest.successRate===1?'All threats timely':latest.successRate===0?'No timely confirmations':`${percent(latest.successRate)} timely`);
  text('training-current-budget',finite(latest.budgetUsed)?`${number(latest.budgetUsed)} / ${number(config.budget)}`:'—');
  text('training-current-threats',finite(latest.threatsDetected)&&finite(latest.threatsMissed)?`${number(latest.threatsDetected,0)} / ${number(latest.threatsMissed,0)}`:'—');
  text('training-warning-latest',seconds(summary.latestWarningTime??latest.warningTime));
  text('training-warning-average',seconds(summary.averageWarningTime));text('training-warning-best',seconds(summary.bestWarningTime));
  text('training-warning-moving',seconds(summary.movingAverageWarningTime??latest.movingAverageWarningTime));text('training-warning-window-label',`Moving average · ${windowSize} episodes`);
  byId('training-moving-window').replaceChildren(new Option(`${windowSize} episodes`,String(windowSize)));byId('training-moving-window').disabled=true;
  text('training-summary-scope',episode?`${episode.toLocaleString()} completed episodes${run?.status==='completed'?' · final':' · recorded so far'}`:'No recorded episodes');
  const target=byId('training-summary');target.replaceChildren();
  const stats=[['Average reward',summary.averageReward,number],['Best episode reward',summary.bestReward,number],['Average warning',summary.averageWarningTime,seconds],['Best episode warning',summary.bestWarningTime,seconds],['Timely success rate',summary.successRate,percent],['Average sensors used',summary.averageSensorsUsed,number],['Average budget used',summary.averageBudgetUsed,number],['Training duration',run?.elapsedSeconds,duration]];
  for(const [label,value,format] of stats)if(finite(value))addStatistic(target,label,format(value));
  if(!target.children.length)addStatistic(target,'Recorded results','Available after the first completed episode',true);
  const metrics=byId('training-latest-metrics');metrics.replaceChildren();
  for(const [label,key,format] of [['Total reward','reward',number],['Mean warning time','warningTime',seconds],['First detection time','firstDetectionTime',seconds],['Confirmation time','confirmationTime',seconds],['Timely success rate','successRate',percent],['Threats detected','threatsDetected',number],['Threats missed','threatsMissed',number],['Sensors placed','sensorsPlaced',number],['Budget used','budgetUsed',number],['Episode duration','episodeDuration',seconds],['Policy loss','policyLoss',number],['Value loss','valueLoss',number],['Entropy','entropy',number]]){
   const value=latest[key],algorithm=state.options?.algorithms?.find(row=>row.id===(run?.algorithm||config.algorithm)),supported=algorithm?.metrics;
   const unsupported=Array.isArray(supported)&&!supported.includes(key)&&['policyLoss','valueLoss','entropy'].includes(key);
   const missingEvent=episode&&Object.hasOwn(latest,key)&&value===null&&['firstDetectionTime','confirmationTime'].includes(key);
   addStatistic(metrics,label,finite(value)?format(value):missingEvent?(key==='firstDetectionTime'?'No detection observed':'No confirmation observed'):unsupported?'Not provided by this algorithm':episode&&['policyLoss','valueLoss','entropy'].includes(key)?'No optimizer update this episode':'Not recorded',!finite(value));
  }
  for(const [label,key] of [['Mean confirmation-based warning','confirmationWarningTime'],['Team warning time','teamWarningTime']])if(finite(latest[key]))addStatistic(metrics,label,seconds(latest[key]));
  if(run?.metricScope)addStatistic(metrics,'Optimizer metric scope',run.metricScope,true);
  const choices=checkpointChoices(run),oldCheckpoint=byId('training-checkpoint').value;byId('training-checkpoint').replaceChildren();
  for(const row of choices){const option=node('option',`${row.label||row.id}${finite(row.episode)?` · episode ${row.episode}`:''}`);option.value=row.id;byId('training-checkpoint').append(option);}
  if(choices.some(row=>row.id===oldCheckpoint))byId('training-checkpoint').value=oldCheckpoint;
  else if(choices.some(row=>row.id==='best'))byId('training-checkpoint').value='best';
  if(!choices.length)byId('training-checkpoint').append(new Option('No saved checkpoint',''));
  renderCheckpointNote();
  const storage=run?.artifactDirectory||run?.directory||run?.outputDirectory||run?.storagePath;
  text('training-storage',storage?`Saved in ${storage}`:run?`Saved in training_runs/${run.id}/`:'');
  const active=(run?.id===state.activeId?run:null)||state.runs.find(row=>row.id===state.activeId);
  text('training-action-status',active?`${runName(active)} · ${active.status}. Completed episodes and checkpoints are saved as training progresses.`:run?.status==='failed'?`Run failed: ${run.error||'See the recorded run error.'}`:run?.status==='stopped'?'Stopped cleanly. Recorded results and saved checkpoints remain available.':run?.status==='completed'?'Training complete. Open a checkpoint in Live Unreal to evaluate it.':'No active run.');
  if(run?.error)error(run.error);
  renderCharts();updateControls();
 }
 function renderElapsed(){
  const run=state.selected;let elapsed=run?.elapsedSeconds;
  if(finite(elapsed)&&activeStates.has(run.status))elapsed+=(Date.now()-state.receivedAt)/1000;
  text('training-elapsed',`Elapsed ${duration(elapsed)}`);
 }
 function renderHistory(){
  const body=byId('training-history-body');body.replaceChildren();
  for(const run of state.runs){
   const row=node('tr'),config=run.config||{},summary=run.summary||{};row.setAttribute('aria-current',String(run.id===state.selected?.id));
   const nameCell=node('td'),open=node('button',runName(run));open.type='button';open.title=runName(run);open.onclick=()=>selectRun(run.id);nameCell.append(open,node('small',(run.status||'saved').replaceAll('_',' ')));row.append(nameCell);
   const date=new Date(run.startedAt||run.createdAt||'');
   for(const value of [algorithmLabel(run.algorithm||config.algorithm),`${run.episode??run.episodes??0} / ${run.totalEpisodes??config.episodes??'—'}`,number(run.budget??config.budget),Number.isNaN(date.getTime())?'—':date.toLocaleString(undefined,{dateStyle:'short',timeStyle:'short'}),number(summary.averageReward??run.averageReward),seconds(summary.averageWarningTime??run.averageWarningTime),percent(summary.successRate??run.successRate)])row.append(node('td',value));
   const overlayCell=node('td'),label=node('label',undefined,'check'),input=document.createElement('input');input.type='checkbox';input.checked=state.overlays.has(run.id)||state.overlayRequests.has(run.id);input.disabled=run.id===state.selected?.id||(!input.checked&&state.overlays.size+state.overlayRequests.size>=3);input.setAttribute('aria-label',`Overlay ${runName(run)}`);input.onchange=()=>toggleOverlay(run.id,input.checked);label.append(input,node('span',run.id===state.selected?.id?'Selected':'Compare'));overlayCell.append(label);row.append(overlayCell);body.append(row);
  }
  if(!state.runs.length){const row=node('tr'),cell=node('td','No saved experiments yet. Your first run will appear here.','training-empty');cell.colSpan=9;row.append(cell);body.append(row);}
 }
 async function toggleOverlay(id,enabled){
  if(!enabled){state.overlayRequests.delete(id);state.overlays.delete(id);renderHistory();renderCharts();return;}
  const request={};
  try{
   if(state.overlays.size+state.overlayRequests.size>=3)throw new Error('Choose up to three overlay runs.');
   state.overlayRequests.set(id,request);renderHistory();
   const run=unwrap(await state.api(`/api/training/run/${encodeURIComponent(id)}`));
   if(state.overlayRequests.get(id)!==request)return;
   state.overlayRequests.delete(id);
   if(state.selected?.id!==id&&state.overlays.size<3)state.overlays.set(id,run);
   renderHistory();renderCharts();
  }catch(e){if(state.overlayRequests.get(id)===request)state.overlayRequests.delete(id);error(e.message);renderHistory();}
 }
 async function refreshOverlays(){
  const active=[...state.overlays.entries()].filter(([,run])=>activeStates.has(run.status));
  await Promise.all(active.map(async([id,previous])=>{const run=unwrap(await state.api(`/api/training/run/${encodeURIComponent(id)}`));if(state.overlays.get(id)===previous)state.overlays.set(id,run);}));
  if(active.length)renderCharts();
 }
 function renderCharts(){
  if(state.chartFrame)cancelAnimationFrame(state.chartFrame);
  state.chartFrame=requestAnimationFrame(()=>{
   state.chartFrame=null;if(!state.visible)return;
   const runs=[state.selected,...state.overlays.values()].filter(Boolean),legend=byId('training-graph-legend');legend.replaceChildren();
   runs.forEach((run,index)=>{const item=node('span'),swatch=node('i');swatch.style.setProperty('--legend-color',colors[index]);item.append(swatch,document.createTextNode(runName(run)));legend.append(item);});
   if(runs.length)legend.append(node('span',`Faint: raw · solid: ${state.selected?.movingAverageWindow||20}-episode average${runs.some(run=>run.metricsSampled)?' · graph downsampled':''}`));
   for(const [id,key,moving,isPercent] of [['reward','reward','movingAverageReward',false],['warning','warningTime','movingAverageWarningTime',false],['success','successRate','movingAverageSuccessRate',true],['budget','budgetUsed','movingAverageBudgetUsed',false]])drawChart(byId(`training-chart-${id}`),runs,key,moving,isPercent);
  });
 }
 function drawChart(canvas,runs,key,movingKey,isPercent){
  const bounds=canvas.getBoundingClientRect();if(bounds.width<1||bounds.height<1)return;
  const ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(bounds.width*ratio);canvas.height=Math.round(bounds.height*ratio);
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);
  const width=bounds.width,height=bounds.height,margin={left:49,right:13,top:12,bottom:34},plotW=width-margin.left-margin.right,plotH=height-margin.top-margin.bottom;
  const values=runs.flatMap(run=>(run.metrics||[]).flatMap(row=>[row[key],row[movingKey]])).filter(finite);
  const episodes=runs.flatMap(run=>(run.metrics||[]).map(row=>row.episode)).filter(finite);
  let low=values.length?Math.min(...values):0,high=values.length?Math.max(...values):1;
  if(isPercent){low=0;high=1;}else{const pad=Math.max((high-low)*.1,Math.abs(high)*.03,.1);low-=pad;high+=pad;if(key!=='reward'&&values.every(value=>value>=0))low=Math.max(0,low);}
  const maxEpisode=Math.max(1,...episodes),x=episode=>margin.left+((episode-1)/Math.max(1,maxEpisode-1))*plotW,y=value=>margin.top+(high-value)/(high-low)*plotH;
  ctx.font='10px "Segoe UI", sans-serif';ctx.textBaseline='middle';ctx.textAlign='right';ctx.lineWidth=1;
  const tickStep=(high-low)/4,tickPrecision=tickStep<1?Math.min(4,Math.ceil(-Math.log10(tickStep))+1):tickStep<10?1:0;
  for(let tick=0;tick<=4;tick++){const value=low+(high-low)*tick/4,py=y(value);ctx.strokeStyle='#263547';ctx.beginPath();ctx.moveTo(margin.left,py);ctx.lineTo(width-margin.right,py);ctx.stroke();ctx.fillStyle='#8298af';ctx.fillText(isPercent?`${Math.round(value*100)}%`:number(value,tickPrecision),margin.left-9,py);}
  ctx.textBaseline='top';ctx.textAlign='center';ctx.fillStyle='#8298af';
  const ticks=Math.min(maxEpisode,width<400?3:5);for(let tick=0;tick<ticks;tick++){const fraction=tick/Math.max(1,ticks-1),episode=Math.round(1+(maxEpisode-1)*fraction);ctx.fillText(String(episode),margin.left+plotW*fraction,height-margin.bottom+8);}
  ctx.fillText('Episode',margin.left+plotW/2,height-11);
  if(!values.length){ctx.fillStyle='#99adc1';ctx.textBaseline='middle';ctx.fillText('Waiting for recorded episode measurements',margin.left+plotW/2,margin.top+plotH/2);canvas.setAttribute('aria-label',`${canvas.parentElement.querySelector('figcaption span').textContent}. No recorded values yet.`);return;}
  ctx.save();ctx.beginPath();ctx.rect(margin.left-2,margin.top-2,plotW+4,plotH+4);ctx.clip();
  runs.forEach((run,index)=>{
   for(const [field,opacity,lineWidth] of [[key,.28,1],[movingKey,1,2]]){
    ctx.strokeStyle=colors[index];ctx.fillStyle=colors[index];ctx.globalAlpha=opacity;ctx.lineWidth=lineWidth;ctx.beginPath();let connected=false,count=0,last=null;
    for(const row of run.metrics||[]){if(!finite(row[field])||!finite(row.episode)){connected=false;continue;}const px=x(row.episode),py=y(row[field]);if(connected)ctx.lineTo(px,py);else ctx.moveTo(px,py);connected=true;count++;last=[px,py];}
    ctx.stroke();if(count===1&&last){ctx.beginPath();ctx.arc(last[0],last[1],3,0,Math.PI*2);ctx.fill();}
   }
  });ctx.restore();ctx.globalAlpha=1;
  const latest=latestMetric(runs[0]);canvas.setAttribute('aria-label',`${canvas.parentElement.querySelector('figcaption span').textContent}. ${runs.length} experiment${runs.length===1?'':'s'}, through episode ${maxEpisode}. Latest selected value ${isPercent?percent(latest[key]):number(latest[key])}.`);
 }
 async function start(event){
  event.preventDefault();if(state.busy||hasActiveRun())return;
  const enabledSensorIds=[...byId('training-catalogue').querySelectorAll('input:checked')].map(input=>input.value);
  if(!enabledSensorIds.length){error('Enable at least one sensor profile before starting training.');return;}
  const rawSeed=byId('training-seed').value.trim();
  const config={runName:byId('training-name').value.trim(),algorithm:byId('training-algorithm').value,episodes:Number(byId('training-episodes').value),budget:Number(byId('training-budget').value),seed:rawSeed===''?null:Number(rawSeed),enabledSensorIds,checkpointFrequency:Number(byId('training-checkpoint-frequency').value)};
  state.busy=true;error('');text('training-action-status','Starting native training and checking the environment…');updateControls();
  try{
   const run=unwrap(await state.api('/api/training/start',config));++state.selectionSequence;state.selected=run;state.receivedAt=Date.now();state.activeId=activeStates.has(run.status)?run.id:null;state.overlays.clear();state.overlayRequests.clear();
   byId('training-name').value=runName(run);renderRun();
   try{await refreshHistory();}catch(e){error(`Training started, but the run list could not be refreshed: ${e.message}`);}
  }catch(e){error(e.message);text('training-action-status','Training did not start. Resolve the reported issue and try again.');}
  finally{state.busy=false;updateControls();schedulePoll();}
 }
 async function stop(){
  if(!state.activeId||state.busy)return;state.busy=true;error('');updateControls();
  try{const result=await state.api('/api/training/stop',{runId:state.activeId});if(result?.run){state.selected=result.run;state.receivedAt=Date.now();}await refreshHistory();if(state.selected)await selectRun(state.selected.id);text('training-action-status','Stop requested. Waiting for safe simulator cleanup and checkpoint saving.');}
  catch(e){error(e.message);}finally{state.busy=false;updateControls();}
 }
 async function loadModel(){
  if(!state.selected||state.busy)return;state.busy=true;error('');updateControls();
  text('training-model-status','Checking the saved model against the current native environment…');
  try{const result=await state.api('/api/training/load',{runId:state.selected.id,checkpoint:byId('training-checkpoint').value});if(!result.model?.id)throw new Error('The manager did not return a compatible model.');text('training-model-status',`Loaded ${result.model.label}. Choose an evaluation seed in Live Unreal.`);await state.onLoadModel?.(result.model,result.status);}
  catch(e){error(e.message);text('training-model-status','Model was not loaded. The reported compatibility or connection issue must be resolved.');}
  finally{state.busy=false;updateControls();}
 }
 function schedulePoll(){clearTimeout(state.timer);if(state.visible&&state.api)state.timer=setTimeout(poll,1500);}
 async function poll(){
  if(state.polling||!state.visible||!state.api)return;state.polling=true;
  try{
   if(!state.busy){if(Date.now()-state.lastHistory>5000){await refreshHistory();await refreshOverlays();}if(state.selected&&activeStates.has(state.selected.status))await refreshSelected();else if(!state.selected&&state.activeId)await selectRun(state.activeId);}
   if(state.pollError&&byId('training-error').textContent===state.pollError&&!state.selected?.error)error('');state.pollError=null;
   renderElapsed();
  }catch(e){state.pollError=`Training status could not be refreshed: ${e.message}. The server may still be running the experiment.`;error(state.pollError);}
  finally{state.polling=false;schedulePoll();}
 }
 async function show(){
  state.visible=true;if(!state.api)return;
  try{if(!state.options)await refreshOptions(true);await refreshHistory();if(!state.selected&&state.runs.length)await selectRun(state.activeId||state.runs[0].id);else renderRun();}
  catch(e){error(e.message);}finally{schedulePoll();}
 }
 function hide(){state.visible=false;clearTimeout(state.timer);}
 function init({api,onLoadModel}){
  state.api=api;state.onLoadModel=onLoadModel;
  byId('training-form').addEventListener('submit',start);byId('training-stop').onclick=stop;byId('training-load-model').onclick=loadModel;
  byId('training-algorithm').onchange=renderAlgorithmNote;
  byId('training-checkpoint').onchange=renderCheckpointNote;
  for(const button of document.querySelectorAll('[data-episodes]'))button.onclick=()=>{byId('training-episodes').value=button.dataset.episodes;};
  byId('training-refresh').onclick=async()=>{error('');await refreshOptions();};
  byId('training-history-refresh').onclick=async()=>{try{await refreshHistory();if(state.selected)await selectRun(state.selected.id);}catch(e){error(e.message);}};
  new ResizeObserver(renderCharts).observe(byId('training-workbench'));
  if(state.visible)show();
 }
 return {init,show,hide};
})();
