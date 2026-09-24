'use strict';
const $ = id => document.getElementById(id);
let token='', catalog=[], view=null, frameIndex=0, mode='recorded', playing=false, busy=false, busyOperation='', connected=false;
let elapsed=0, lastWall=0, liveWall=0, loadSequence=0;
let nativeImage=null, eventSignature='';
const comparison={episodes:[],layouts:[],scenario:null,result:null,loading:false,episodeId:null,sequence:0,selectedPolicy:null};
const training={status:{running:false,phase:'idle',episode:0,totalEpisodes:0,history:[]},timer:null,lastViewerEpisode:-1,polling:false,replaySelection:'latest',replayLoading:false,replaySequence:0,replayViewer:null,historySignature:''};
const drawingCache=new WeakMap();
const fmt=(v,n=1)=>Number.isFinite(v)?v.toFixed(n):'—';
const pct=v=>Number.isFinite(v)?`${Math.round(v*100)}%`:'—';
function warningText(value){
 if(mode==='live'&&(!view||!view.ended))return 'Pending';
 return Number.isFinite(value)?`${fmt(value,2)} s`:'Unavailable';
}
function timeText(seconds){return `${String(Math.floor(seconds/60)).padStart(2,'0')}:${(seconds%60).toFixed(1).padStart(4,'0')}`;}
function showError(message){$('error').hidden=!message;$('error').textContent=message||'';}
async function api(path,body){
 const opts=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Console-Token':token},body:JSON.stringify(body)};
 const response=await fetch(path,opts);const data=await response.json();
 if(!response.ok)throw new Error(data.error||'Request failed');return data;
}
function connectionStatus(){
 const label=mode==='training'?(training.status.running?'Training · Unreal connected':`Training · ${training.status.phase}`):mode==='comparison'?'Comparison · native evaluation':mode==='recorded'?'Replay ready':connected?'Unreal connected':'Unreal disconnected';
 $('connection-label').textContent=label;
 $('connection').firstChild.style.background=(mode==='live'&&!connected)||(mode==='training'&&training.status.phase==='failed')?'#ffb18a':'#6ce8c8';
 $('connect').textContent=connected?'Disconnect':'Connect to Unreal';
}
function controls(){
 const blocking=(busy&&busyOperation!=='step')||comparison.loading, liveEnded=Boolean(mode==='live'&&view?.ended), draft=mode==='comparison'&&!comparison.result;
 $('play').textContent=playing?'Pause':mode==='live'?'Run episode':'Play replay';
 $('play').disabled=!view||blocking||liveEnded||draft;
 $('step').disabled=!view||playing||busy||blocking||liveEnded||draft;
 $('restart').disabled=!view||mode==='live'||blocking||draft;
 $('timeline').disabled=!view||blocking||draft;$('speed').disabled=!view||draft;
 $('connect').disabled=busy||playing||training.status.running;$('plan').disabled=!connected||busy||playing||training.status.running;
 $('training-start').disabled=busy||training.status.running;$('training-stop').disabled=!training.status.running;
 for(const id of ['training-name','training-algorithm','training-initialization','training-episodes','training-batch','training-sensors','training-seed','training-exploration','training-validation','training-checkpoint'])$(id).disabled=training.status.running;
 $('training-exploration').disabled=training.status.running||$('training-algorithm').value==='local_ppo';
 $('training-replay-load').disabled=training.replayLoading||!training.status.outputDirectory;
 $('training-episode-open').disabled=training.replayLoading||!training.status.episode;
 $('training-export').disabled=!training.status.outputDirectory;
 for(const id of ['recorded-mode','live-mode','comparison-mode','training-mode'])$(id).disabled=busy||playing||comparison.loading;
 comparisonControls();
}
function pause(){playing=false;lastWall=0;controls();}
function liveTimelineMax(){
 const fixed=Number(view?.fixedStepSeconds), limit=Number(view?.timeLimitSeconds);
 return Number.isFinite(fixed)&&fixed>0&&Number.isFinite(limit)&&limit>0
  ?Math.max(1,Math.ceil(limit/fixed)):Math.max(1,view?.frames.at(-1)?.completedSteps||0);
}
function syncTimeline(){
 if(!view){$('timeline').max=1;$('timeline').value=0;return;}
 if(mode==='live'){$('timeline').max=liveTimelineMax();$('timeline').value=view.frames[frameIndex]?.completedSteps||0;}
 else{$('timeline').max=Math.max(1,view.frames.length-1);$('timeline').value=frameIndex;}
}
function element(tag,text,cls){const el=document.createElement(tag);if(cls)el.className=cls;el.textContent=text;return el;}
function sensorDetails(){
 $('active-policy').textContent=mode==='training'&&view?'Active: trainable directional warning policy':mode==='live'&&view?`Active: ${view.policy}`:'No active planner.';
 $('surface-status').textContent=mode==='training'?'Every training placement is checked against native surface, spacing, FOV and budget constraints.':mode==='live'&&view?(view.surfaceMounted?'Surface-mounted · unsupported sites excluded.':'Legacy build: surface support not enforced. Rebuild Unreal.'):'Surface support is checked by Unreal before deployment.';
 $('sensors').replaceChildren();
 for(const [i,p] of (view?.placements||[]).entries()){
  const c=view.catalogue.find(c=>c.id===p.sensor_id)||{};
  const row=element('div','','sensor-row'),symbol=element('span',String(i+1).padStart(2,'0'),'sensor-symbol');
  const orientation=c.directional?` · yaw ${fmt(p.yaw_deg,0)}° pitch ${fmt(p.pitch_deg,0)}°`:'';
  const text=element('div','');text.append(element('strong',c.label||p.sensor_id),element('small',`${fmt(p.position[0],0)}, ${fmt(p.position[1],0)} m${orientation} · ${fmt(c.cost)} units`));row.append(symbol,text);$('sensors').append(row);
 }
 if(!view?.placements.length)$('sensors').append(element('p',view?'Policy chose STOP. No sensors.':'No layout committed.','helper'));
 $('sensor-count').textContent=view?view.placements.length:'—';
 const cost=view?.placements.reduce((sum,p)=>sum+(p.cost??view.catalogue.find(c=>c.id===p.sensor_id)?.cost??0),0)||0;
 $('cost').textContent=view?`${fmt(cost)} / ${fmt(view.budget)}`:'—';$('budget-bar').style.width=`${view?Math.min(100,cost/view.budget*100):0}%`;
}
function results(){
 const m=view?.metrics;$('result-detected').textContent=pct(m?.detected_fraction);$('result-confirmed').textContent=pct(m?.confirmed_fraction);
 const timely=m?.timely_fraction??(m?.target_results?.length?m.target_results.filter(t=>t.timely_confirmed).length/m.target_results.length:undefined);
 $('result-timely').textContent=pct(timely);$('result-return').textContent=fmt(m?.return,2);
 $('result-team-warning').textContent=warningText(m?.team_warning_seconds_lower_bound);
 $('result-mean-warning').textContent=warningText(m?.mean_drone_warning_seconds_lower_bound);
 $('results-tag').textContent=mode==='recorded'?'RECORDED FINAL':mode==='comparison'?(m?'NATIVE FINAL':'LOADING EPISODE'):mode==='training'?(m?'SELECTED NATIVE REPLAY':'TRAINING READY'):m?'MEASURED FINAL':'AWAITING EPISODE END';
 $('outcome-label').textContent=mode==='recorded'?'Recorded final result':mode==='comparison'?'Native final result':mode==='training'?'Replay status':'Episode status';
 $('detected-label').textContent=mode==='live'?'Public tracks':mode==='training'?'Replay detected':'Detected so far';
 $('result-note').textContent=mode==='recorded'?'Synthetic sensing outcomes. Timely confirmation means detection before the deadline; no interception is simulated. Warning is unavailable when the replay does not contain the exact metric.':mode==='comparison'?'Matched native sensing outcomes. Both layouts face the same adversary episode and deployment limits.':mode==='training'?'Results for the selected replay, which is one scenario. Compare the validation means above to judge policy improvement. RL optimizes mean per-drone warning time; undetected arrivals count as zero.':'Live metrics appear when the native episode ends. Directional thermal sensors use Unreal world-static line-of-sight; probability parameters remain simulator assumptions. Warning values are measured lower bounds from detection to 20 m objective-zone arrival.';
}
function setView(data){view=data;nativeImage=null;eventSignature='';if(data.nativePreview){const image=new Image();image.onload=()=>{if(view===data){nativeImage=image;render();}};image.src=data.nativePreview.image;}$('map').setAttribute('aria-label',data.nativePreview?'Native Unreal sensor placement preview':'Top-down simulation view');document.body.classList.toggle('native-preview',Boolean(data.nativePreview));frameIndex=0;elapsed=0;$('empty').hidden=true;syncTimeline();$('episode-title').textContent=view.label;$('coordinates').textContent=view.coordinateLabel;sensorDetails();results();render();controls();}
async function loadReplay(){
 if(mode==='comparison')return loadComparisonScenario();
 pause();const seq=++loadSequence;const row=catalog.find(r=>r.profile===$('profile').value&&String(r.policy)===$('policy').value&&String(r.case)===$('case').value);
 if(!row)return showError('This recorded case is unavailable.');
 try{const data=await api(`/api/replay/${row.id}`);if(seq!==loadSequence||mode!=='recorded')return;setView(data);showError('');}catch(e){showError(e.message);}
}
async function switchMode(next){
 pause();if(next!=='training'&&training.timer){clearTimeout(training.timer);training.timer=null;}mode=next;++loadSequence;showError('');
 if(mode==='recorded'&&$('policy').value.startsWith('trained-'))$('policy').value='406';
 if(mode!=='comparison')$('transport').hidden=false;
 for(const id of ['recorded','live','comparison','training'])$(id+'-mode').setAttribute('aria-pressed',String(mode===id));
 document.body.classList.toggle('comparison-mode',mode==='comparison');
 document.body.classList.toggle('training-mode',mode==='training');
 $('training-evidence').hidden=mode!=='training';
 syncComparisonPolicyOptions();
 if(mode==='comparison')$('comparison-panel').insertBefore($('transport'),$('comparison-results'));else $('single-map-container').after($('transport'));
 $('recorded-controls').hidden=mode==='live'||mode==='training';$('profile').hidden=mode==='comparison';$('profile-label').hidden=mode==='comparison';$('deployment-controls').hidden=mode==='comparison';$('single-map-container').hidden=mode==='comparison';
 $('case-label').textContent=mode==='comparison'?'Episode':'Recorded case';$('policy-label').textContent=mode==='comparison'?'RL policy':'Trained checkpoint';$('episode-help').textContent=mode==='comparison'?'Both layouts face the same recorded native scenario using realistic limited-FOV sensors.':'All three checkpoints and all 18 published cases are available, including failures.';fillEpisodeOptions();$('live-controls').hidden=mode!=='live';$('training-controls').hidden=mode!=='training';$('comparison-controls').hidden=mode!=='comparison';$('comparison-panel').hidden=mode!=='comparison';
 $('source-badge').textContent=mode==='recorded'?'RECORDED · SYNTHETIC':mode==='comparison'?'COMPARISON · NATIVE UNREAL':mode==='training'?'TRAINING · NATIVE UNREAL':'LIVE UNREAL · SYNTHETIC';
 $('provenance').textContent=mode==='recorded'?'Published temporal-v6 evidence · 18 recorded cases':mode==='comparison'?'Matched native Unreal evaluation · fixed common-sense baseline':mode==='training'?'Local native training · directional policy · retained checkpoints':'Local Unreal bridge · scripted Red · experimental Blue';
 if(mode==='recorded')await loadReplay();else if(mode==='comparison')await loadComparisonScenario();else if(mode==='training'){$('empty-title').textContent='Start a native training run';$('empty-copy').textContent='Launch the scene with -TrainingWorkbench -DelayedDetectionDemo, then choose a starting placement and start training.';$('empty-note').textContent='The latest completed placement will appear here.';if(view?.mode!=='training')view=null;await refreshTraining();if(!view){$('empty').hidden=false;$('episode-title').textContent='Training workbench ready';$('clock').textContent='00:00.0';$('progress').textContent='No completed training episode';sensorDetails();results();render();controls();}}else{$('empty-title').textContent='Connect your Unreal scene';$('empty-copy').textContent='Launch Istana with -IstanaBlueLive, then connect from the left panel.';$('empty-note').textContent='Recorded replays remain available.';view=null;$('empty').hidden=false;$('episode-title').textContent='Awaiting live episode';$('clock').textContent='00:00.0';$('progress').textContent='Fixed-step simulation';sensorDetails();results();render();controls();}
 connectionStatus();
}
function render(){
 const frame=view?.frames[frameIndex];syncTimeline();
 $('clock').textContent=timeText(frame?.time||0);
 $('progress').textContent=view?mode==='live'?`Step ${frame?.completedSteps??0} / ${liveTimelineMax()} · ${view.ended?'ended':playing?`running at ${$('speed').value}×`:'paused'}`:mode==='training'?`${view.trainingReplayLabel||trainingReplayLabel(training.replaySelection)} · frame ${frameIndex+1} / ${view.frames.length}`:`Frame ${frameIndex+1} / ${view.frames.length} · ${fmt(view.frames.at(-1).time)} s`:'No episode';
 $('threat-count').textContent=mode==='training'?(view?.metrics?.targets??'—'):frame?frame.threats.length:'—';
 $('detected-count').textContent=mode==='training'&&Number.isFinite(view?.metrics?.detected_fraction)?Math.round(view.metrics.targets*view.metrics.detected_fraction):frame?(mode==='live'?(frame.tracks||[]).length:frame.threats.filter(t=>t.ever_detected).length):'—';
 $('confirmed-count').textContent=frame?(mode==='live'?(frame.tracks||[]).filter(t=>t.confirmed).length:frame.threats.filter(t=>t.tracked).length):'—';
 $('outcome').textContent=mode==='training'?(view?.outcome||training.status.phase).replaceAll('_',' '):view?(view.outcome==='defended'?'Timely sensing':view.outcome==='running'?'In progress':mode==='live'?'Episode ended':view.outcome.replaceAll('_',' ')):'—';
 $('outcome').title=view?.outcome||'';
 if(mode==='comparison'&&!comparison.result){$('progress').textContent='Loading paired episode';for(const id of ['threat-count','detected-count','confirmed-count','outcome'])$(id).textContent='—';}
 draw(frame);if(mode!=='comparison')events(frame);
}
function events(frame){
 const log=[];
 for(const [i,p] of (mode==='comparison'&&!comparison.result?[]:view?.placements||[]).entries()){const c=view.catalogue.find(c=>c.id===p.sensor_id);log.push([0,`Blue deployed ${c?.label||p.sensor_id} at sensor ${i+1}`]);}
 if(frame&&(mode==='recorded'||(mode==='comparison'&&comparison.result))){
  for(const t of view.metrics.target_results||[]){
   if(t.first_detection!==null&&t.first_detection<=frame.time)log.push([t.first_detection,`${t.id} detected`]);
   if(t.first_confirmation!==null&&t.first_confirmation<=frame.time)log.push([t.first_confirmation,`${t.id} track confirmed`]);
   if(t.time_to_zone<=frame.time)log.push([t.time_to_zone,`${t.id} reached zone · ${t.timely_confirmed?'confirmed before deadline':'deadline missed'}`]);
  }
 }else if(frame)for(const t of frame.tracks||[])log.push([t.timestamp,mode==='comparison'?`${t.id} · initial public report (${pct(t.confidence)} confidence)`:`${t.id} · ${t.confirmed?'confirmed track':'detection report'}`]);
 log.sort((a,b)=>b[0]-a[0]);$('event-count').textContent=log.length;
 const visible=log.slice(0,40),signature=JSON.stringify(visible);if(signature===eventSignature)return;eventSignature=signature;$('events').replaceChildren();
 for(const [t,text] of visible){const row=element('div','','event');row.append(element('time',`${fmt(t)} s`),element('span',text));$('events').append(row);}
 if(!log.length)$('events').append(element('p',mode==='comparison'&&!comparison.result?'Evaluate to reveal adversary paths and sensing results.':'Detection reports will appear here.','helper'));
}
function mapData(data){
 if(!data)return {extent:150,paths:new Map()};
 if(drawingCache.has(data))return drawingCache.get(data);
 let extent=120;const paths=new Map();
 const include=p=>{extent=Math.max(extent,Math.abs(p[0]),Math.abs(p[1]));};
 for(const [index,frame] of (data.frames||[]).entries())for(const t of frame.threats||[]){include(t.position);if(!paths.has(t.id))paths.set(t.id,[]);paths.get(t.id).push({index,position:t.position});}
 for(const p of data.placements||[])include(p.position);
 for(const p of data.sites||[])include(p);
 const result={extent:extent*1.25,paths};drawingCache.set(data,result);return result;
}
function draw(frame){
 if(mode!=='comparison'){drawMap($('map'),view,frame,frameIndex);return;}
 const pair=comparison.result,sharedExtent=Math.max(mapData(pair?.rl).extent,mapData(pair?.baseline).extent),time=frame?.time||0;
 for(const method of ['rl','baseline']){
  const data=pair?.[method],index=data?Math.max(0,data.frames.findLastIndex(f=>f.time<=time)):0,current=data?.frames[index];
  drawMap($(`comparison-map-${method}`),data,current,index,sharedExtent);
  const detected=current?.threats.filter(t=>t.ever_detected??t.detected).length||0,confirmed=current?.threats.filter(t=>t.tracked??t.confirmed).length||0;
  $(`comparison-${method}-now`).textContent=pair?.layoutOnly&&data?`${data.placements.length} sensors · placement preview`:current?`${detected} detected · ${confirmed} confirmed at ${fmt(time,1)} s`:'Awaiting episode';
 }
}
function drawMap(canvas,renderView,frame,renderIndex,sharedExtent=null){
 const bounds=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 const pixelWidth=Math.max(1,Math.round(bounds.width*dpr)),pixelHeight=Math.max(1,Math.round(bounds.height*dpr));
 if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
 const w=bounds.width,h=bounds.height;ctx.fillStyle='#0b141e';ctx.fillRect(0,0,w,h);
 if(renderView?.nativePreview){if(nativeImage){const s=Math.min(w/1920,h/1080),ox=(w-1920*s)/2,oy=(h-1080*s)/2;ctx.drawImage(nativeImage,ox,oy,1920*s,1080*s);for(const a of renderView.nativePreview.anchors){const x=ox+a.x*s,y=oy+a.y*s;ctx.strokeStyle='#9fffe3';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(x,y+2,12,6,0,0,Math.PI*2);ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#ccfff0';ctx.fillText('Sensor',x+16,y-10);}}return;}
 const extent=sharedExtent??mapData(renderView).extent;
 const scale=Math.max(.01,Math.min(w-70,h-105)/(extent*2)),cx=w/2,cy=h/2+1;
 const xy=p=>[cx+p[0]*scale,cy-p[1]*scale];
 const line=(a,b,color,width=1)=>{ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
 const circle=(p,r,fill,stroke)=>{ctx.beginPath();ctx.arc(...p,r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}};
 const label=(txt,x,y,color='#8399ae',align='left')=>{ctx.fillStyle=color;ctx.font='10px Consolas,monospace';ctx.textAlign=align;ctx.fillText(txt,x,y);};
 const interval=extent>300?100:50;
 for(let m=-Math.floor(extent/interval)*interval;m<=extent;m+=interval){const [x,y]=xy([m,m]);line([x,45],[x,h-40],m===0?'#2c4055':'#192a3a');line([25,y],[w-25,y],m===0?'#2c4055':'#192a3a');if(m!==0&&y>65&&y<h-60)label(`${m}`,30,y-4);if(x>50&&x<w-50)label(`${m}`,x,h-42,undefined,'center');}
 const radius=(renderView?.objectiveRadius||20)*scale;circle([cx,cy],radius,'#6ce8c815','#52867e');circle([cx,cy],3,'#6ce8c8');label('OBJECTIVE',cx,cy+radius+16,'#a4c7bd','center');
 label(mode==='live'?'Y+':'N',w-30,60,'#b1c5d9','center');line([w-30,82],[w-30,66],'#b1c5d9');
 const bar=50*scale;line([w-30-bar,h-62],[w-30,h-62],'#708aa3',2);label('50 m',w-30,h-69,undefined,'right');
 if(!renderView||!frame)return;
 if($('sites').checked)for(const [i,site] of (renderView.sites||[]).entries())if(!(renderView.blockedSites||[]).includes(i))circle(xy(site),2,'#566d87');
 for(const [i,p] of renderView.placements.entries()){
  const c=renderView.catalogue.find(c=>c.id===p.sensor_id),[x,y]=xy(p.position),range=Math.max(...Object.values(c?.ranges||{r:0}));
  if($('ranges').checked&&c?.directional){const assumptions=c.simulation_assumptions||{},hardware=c.manufacturer_specifications||{};
   const radius=(assumptions.max_evaluation_distance_m||range)*scale,yaw=(p.yaw_deg||0)*Math.PI/180,half=(hardware.horizontal_fov_deg||24)*Math.PI/360;
   ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,radius,-yaw-half,-yaw+half);ctx.closePath();ctx.fillStyle='#70b7ff0c';ctx.fill();ctx.strokeStyle='#70b7ff70';ctx.setLineDash([5,5]);ctx.stroke();ctx.setLineDash([]);
  }else if($('ranges').checked){ctx.setLineDash([4,5]);circle([x,y],range*scale,'#70b7ff06','#70b7ff45');ctx.setLineDash([]);}
  ctx.fillStyle='#70b7ff';ctx.fillRect(x-4,y-4,8,8);label(`S${i+1}`,x+9,y-7,'#8fc9ff');
 }
 if($('trails').checked)for(const t of frame.threats){ctx.beginPath();let first=true;for(const point of mapData(renderView).paths.get(t.id)||[]){if(point.index>renderIndex)break;const q=xy(point.position);if(first){ctx.moveTo(...q);first=false;}else ctx.lineTo(...q);}ctx.strokeStyle='#ff8b8045';ctx.lineWidth=1.2;ctx.stroke();}
 for(const d of frame.detections||[]){const p=renderView.placements[d.sensor_index],t=frame.threats.find(t=>t.id===d.target_id);if(p&&t){ctx.setLineDash([3,4]);line(xy(p.position),xy(t.position),'#f4cf7866');ctx.setLineDash([]);}}
 for(const t of frame.threats){const [x,y]=xy(t.position),id=Number(String(t.id).replace('drone-','')),diag=(frame.directionalDiagnostics||[]).find(d=>d.droneId===id);if($('drone-rings').checked&&(t.detected||t.tracked||t.confirmed))circle([x,y],10,null,(t.tracked??t.confirmed)?'#6ce8c8':'#f4cf78');ctx.save();ctx.translate(x,y);ctx.rotate(Math.PI/4);ctx.fillStyle=t.breached?'#ff5757':(t.tracked??t.confirmed)?'#6ce8c8':t.detected?'#f4cf78':'#ff8b80';ctx.fillRect(-3.5,-3.5,7,7);ctx.restore();if(w>470&&frame.threats.length<=8){label(t.id,x+12,y+4,'#c2ccda');if(diag)label(`${diag.insideFov?'IN':'OUT'} · ${diag.lineOfSight?'LOS':'BLOCKED'} · ${fmt(diag.pixelsOnTarget,2)} px · P ${fmt(diag.detectionProbability,3)}`,x+12,y+17,diag.blockedByGeometry?'#ff756b':'#9adbc9');}}
 for(const t of frame.tracks||[]){const p=xy(t.position),publicReport=mode==='comparison'&&!comparison.result;if(publicReport){circle(p,7,'#f4cf7818','#f4cf78');circle(p,2,'#f4cf78');label(`${t.id} · public report`,p[0]+12,p[1]-9,'#f4cf78');}else{if($('drone-rings').checked)circle(p,9,null,t.confirmed?'#6ce8c8':'#f4cf78');if(w>470)label(t.id,p[0]+12,p[1]-9,'#f4cf78');}}
}
async function liveAction(op,payload={}){
 if(busy)return;busy=true;busyOperation=op;controls();showError('');
 try{const result=await api(`/api/action/${op}`,payload);
  if(op==='connect'||op==='disconnect'){connected=result.connected;if(op==='disconnect'){view=null;await switchMode('live');}}
  else if(op==='reset'||op==='preview'){setView(result);}
  else if(op==='step'){const follow=playing||frameIndex>=Math.max(0,(view?.frames.length||1)-1);view=result;frameIndex=follow?view.frames.length-1:Math.min(frameIndex,view.frames.length-1);results();render();if(view.ended)pause();}
 }catch(e){pause();connected=false;view=null;showError(`${e.message} ${mode==='live'?'Recorded replays remain available.':''}`);$('empty').hidden=false;sensorDetails();results();render();}
 finally{busy=false;busyOperation='';connectionStatus();controls();}
}
function animate(now){
 if(playing&&view){
  if(mode!=='live'){
   if(lastWall)elapsed+=(now-lastWall)/1000*Number($('speed').value);lastWall=now;
   while(frameIndex<view.frames.length-1&&view.frames[frameIndex+1].time<=elapsed){frameIndex++;render();}
   if(frameIndex>=view.frames.length-1)pause();
  }else{const speed=Math.max(.1,Number($('speed').value)||1),interval=1000*(Number(view.stepDurationSeconds)||.5)/speed;if(!busy&&now-liveWall>=interval){liveWall=now;liveAction('step');}}
 }
 requestAnimationFrame(animate);
}
function fillEpisodeOptions(){
 const select=$('case'),previous=select.value,previousCase=comparison.episodes.find(row=>String(row.id)===previous)?.case;select.replaceChildren();
 const rows=mode==='comparison'?comparison.episodes.filter(row=>String(row.policy)===$('policy').value):[{id:'1',label:'Case 01'},{id:'2',label:'Case 02'}];
 for(const row of rows){const option=element('option',row.label);option.value=row.id;select.append(option);}
 if(rows.some(row=>String(row.id)===previous))select.value=previous;
 else if(mode==='comparison'&&previousCase!==undefined){const equivalent=rows.find(row=>row.case===previousCase);if(equivalent)select.value=equivalent.id;}
 const chosen=rows.find(row=>String(row.id)===select.value);if(mode==='comparison'&&chosen?.defaultLayout){for(const option of $('comparison-layout').options){option.hidden=!(chosen.availableLayouts||[chosen.defaultLayout]).includes(option.value);option.textContent=chosen.trainedModel&&option.value==='directional_balanced_8'?'Matched training sensors · contractor layout':comparison.layouts.find(row=>row.id===option.value)?.label||option.textContent;}$('comparison-layout').value=chosen.defaultLayout;}
}

function isSavedLayout(value){return value.startsWith('saved-')||value.startsWith('trained-')||value.startsWith('observed-trained-');}
function syncComparisonPolicyOptions(){
 const policies=new Set(comparison.episodes.map(row=>String(row.policy))),select=$('policy');
 for(const option of select.options)option.hidden=mode==='comparison'?!policies.has(option.value):isSavedLayout(option.value);
 if(mode==='comparison'){
  const preferred=comparison.episodes.find(row=>row.trainedModel&&!row.bestObservedEpisode)||comparison.episodes[0];
  select.value=policies.has(comparison.selectedPolicy)?comparison.selectedPolicy:String(preferred?.policy||'');
 }else if(isSavedLayout(select.value))select.value='406';
}
function syncModelCatalog(session){
 comparison.episodes=session.comparisonEpisodes||comparison.episodes;
 const layoutSelect=$('comparison-layout'),previousLayout=layoutSelect.value;
 comparison.layouts=session.comparisonLayouts||comparison.layouts;
 layoutSelect.replaceChildren();for(const item of comparison.layouts){const option=element('option',item.label);option.value=item.id;layoutSelect.append(option);}
 layoutSelect.value=comparison.layouts.some(item=>item.id===previousLayout)?previousLayout:comparison.layouts[0]?.id||'';
 $('trained-comparison-models')?.remove();
 if(session.trainedModels?.length){const group=document.createElement('optgroup');group.id='trained-comparison-models';group.label='Validated models and retained episode layouts';for(const model of session.trainedModels){const suffix=model.kind==='observedEpisode'?` · ${fmt(model.bestWarningSeconds,2)} s observed`:` · selected at episode ${model.bestEpisode}`;const option=element('option',`${model.label}${suffix}`);option.value=model.id;group.append(option);}$('policy').append(group);}
 $('saved-live-models')?.remove();
 if(session.savedModels?.length){const group=document.createElement('optgroup');group.id='saved-live-models';group.label='Saved native layouts';for(const model of session.savedModels){const option=element('option',model.label);option.value=model.id;group.append(option);}$('live-policy').append(group);}
 syncComparisonPolicyOptions();
 if(mode==='comparison'){fillEpisodeOptions();const previousOption=[...layoutSelect.options].find(option=>option.value===previousLayout&&!option.hidden);if(previousOption)layoutSelect.value=previousLayout;}
}
function comparisonControls(){
 const locked=comparison.loading||busy;
 for(const id of ['profile','policy','case','comparison-layout'])$(id).disabled=locked;
}
function comparisonClearResults(){
 for(const method of ['rl','baseline','delta'])for(const metric of ['detected','confirmed','timely','cost','first-detection','detection-time','confirmation-time','warning'])$(`comparison-${method}-${metric}`).textContent='—';
 for(const method of ['rl','baseline'])$(`comparison-${method}-layout`).textContent='Loading layout…';
 $('comparison-results').hidden=false;$('comparison-layout-note').hidden=true;$('transport').hidden=false;
 $('comparison-verdict').textContent='Loading the paired native episode…';$('comparison-timing-note').textContent='';
}
function comparisonCost(data){return (data.placements||[]).reduce((sum,p)=>sum+(p.cost??data.catalogue.find(c=>c.id===p.sensor_id)?.cost??0),0);}
function comparisonSummary(data,native={}){
 const metrics=data.metrics||{},targets=metrics.target_results||[],total=native.target_count??(targets.length||data.frames?.[0]?.threats?.length||0);
 const detected=native.detected_count??(Number.isFinite(metrics.detected_fraction)?Math.round(metrics.detected_fraction*total):null);
 const confirmed=native.confirmed_count??(Number.isFinite(metrics.confirmed_fraction)?Math.round(metrics.confirmed_fraction*total):null);
 const timely=native.timely_confirmed_count??(Number.isFinite(metrics.timely_fraction)?Math.round(metrics.timely_fraction*total):targets.filter(t=>t.timely_confirmed).length);
 return {detected,confirmed,timely,total,cost:native.cost??comparisonCost(data),warning:native.mean_warning_s??metrics.mean_drone_warning_seconds_lower_bound};
}
function pairedTimes(rl,baseline,key){
 const right=new Map((baseline.metrics?.target_results||[]).map(target=>[String(target.id),target]));
 const pairs=(rl.metrics?.target_results||[]).map(left=>[left[key],right.get(String(left.id))?.[key]]).filter(pair=>pair.every(Number.isFinite));
 return {count:pairs.length,rl:pairs.length?pairs.reduce((sum,p)=>sum+p[0],0)/pairs.length:null,baseline:pairs.length?pairs.reduce((sum,p)=>sum+p[1],0)/pairs.length:null};
}
function signed(value,unit=''){return Number.isFinite(value)?`${value>0?'+':''}${fmt(value,unit?2:0)}${unit}`:'Unavailable';}
function comparisonResults(result){
 if(result.layoutOnly){
  $('comparison-results').hidden=true;$('comparison-layout-note').hidden=false;$('transport').hidden=true;
  $('comparison-layout-note').textContent='Placement geometry only · the archived RL layout uses its original three-sensor contract, while the new workbench start uses five directional sensors. Detection, warning-time and reward metrics are intentionally not compared.';
  for(const method of ['rl','baseline']){
   const data=result[method],cost=comparisonCost(data),name=method==='rl'?'Archived reinforcement-learning':'Eight-directional-sensor';
   $(`comparison-${method}-layout`).textContent=`${data.placements.length} sensors · ${fmt(cost)} / ${fmt(data.budget)} cost`;
   $(`comparison-map-${method}`).setAttribute('aria-label',`${name} placement preview with ${data.placements.length} sensors. No performance metrics are shown.`);
  }
  return;
 }
 $('comparison-results').hidden=false;$('comparison-layout-note').hidden=true;$('transport').hidden=false;
 const summaries={rl:comparisonSummary(result.rl,result.metrics?.rl),baseline:comparisonSummary(result.baseline,result.metrics?.baseline)};
   const sectorBenchmark=result.method==='directional_balanced_8',trainedModel=Boolean(result.trainedModel),trainedSensorCount=result.rl.maxSensors??result.rl.placements?.length??8;
   $('comparison-rl-column').textContent=sectorBenchmark?'Archived RL layout':trainedModel?result.policyLabel:'RL';
   $('comparison-baseline-column').textContent=trainedModel?`${trainedSensorCount} sensors`:sectorBenchmark?'Eight sensors':'Common sense';
   $('comparison-delta-column').textContent=sectorBenchmark||trainedModel?'Difference':'RL difference';
   $('comparison-delta-caption').textContent=trainedModel?`Difference = ${result.policyLabel} − ${trainedSensorCount}-sensor placement`:sectorBenchmark?'Difference = archived RL layout − eight-sensor placement':'RL difference = RL − common sense';
 for(const method of ['rl','baseline']){
  const summary=summaries[method],data=result[method];
  for(const metric of ['detected','confirmed','timely'])$(`comparison-${method}-${metric}`).textContent=Number.isFinite(summary[metric])?`${summary[metric]} / ${summary.total}`:'Unavailable';
  $(`comparison-${method}-cost`).textContent=`${fmt(summary.cost)} / ${fmt(data.budget)}`;
  $(`comparison-${method}-warning`).textContent=Number.isFinite(summary.warning)?`${fmt(summary.warning,2)} s`:'Unavailable';
  $(`comparison-${method}-layout`).textContent=`${data.placements.length} sensors · ${fmt(summary.cost)} / ${fmt(data.budget)} cost`;
    const layoutName=method==='rl'?(sectorBenchmark?'Archived reinforcement-learning':trainedModel?result.policyLabel:'Reinforcement learning'):(trainedModel?`${trainedSensorCount}-directional-sensor`:sectorBenchmark?'Eight-directional-sensor':'Fixed common-sense');
  $(`comparison-map-${method}`).setAttribute('aria-label',`${layoutName} layout: ${data.placements.length} sensors, ${summary.detected} of ${summary.total} adversaries detected by episode end. Shared playback shows sensing over time.`);
 }
 for(const metric of ['detected','confirmed','timely','cost','warning']){
  const left=summaries.rl[metric],right=summaries.baseline[metric],delta=Number.isFinite(left)&&Number.isFinite(right)?left-right:null;
  $(`comparison-delta-${metric}`).textContent=signed(delta,metric==='warning'?' s':metric==='cost'?' units':'');
 }
 const firstDetection={rl:result.metrics?.rl?.first_detection_s,baseline:result.metrics?.baseline?.first_detection_s};
 for(const method of ['rl','baseline'])$(`comparison-${method}-first-detection`).textContent=Number.isFinite(firstDetection[method])?`${fmt(firstDetection[method],2)} s`:'No detection';
 const firstDelta=Number.isFinite(firstDetection.rl)&&Number.isFinite(firstDetection.baseline)?firstDetection.rl-firstDetection.baseline:null;
 $('comparison-delta-first-detection').textContent=Number.isFinite(firstDelta)?`${signed(firstDelta,' s')}${firstDelta<0?' · earlier':firstDelta>0?' · later':' · same time'}`:'Not comparable';
 const timing=result.timing;
 const detection=timing?{count:timing.shared_detected_count,rl:timing.mean_rl_detection_s,baseline:timing.mean_baseline_detection_s,delta:timing.mean_detection_delta_s}:pairedTimes(result.rl,result.baseline,'first_detection');
 const confirmation=timing?{count:timing.shared_confirmed_count,rl:timing.mean_rl_confirmation_s,baseline:timing.mean_baseline_confirmation_s,delta:timing.mean_confirmation_delta_s}:pairedTimes(result.rl,result.baseline,'first_confirmation');
 for(const [metric,pair] of [['detection-time',detection],['confirmation-time',confirmation]]){
  for(const method of ['rl','baseline'])$(`comparison-${method}-${metric}`).textContent=pair.count?`${fmt(pair[method],2)} s`:'No shared targets';
  const delta=pair.delta??(pair.rl-pair.baseline);
  $(`comparison-delta-${metric}`).textContent=pair.count?`${signed(delta,' s')}${delta<0?' · earlier':delta>0?' · later':' · same time'}`:'Not comparable';
 }
 const difference=summaries.rl.detected-summaries.baseline.detected;
   const subject=sectorBenchmark?'The archived RL layout':trainedModel?result.policyLabel:'RL';
 $('comparison-verdict').textContent=difference===0?'Both layouts detected the same number of adversaries in this episode.':`${subject} detected ${Math.abs(difference)} ${difference>0?'more':'fewer'} adversar${Math.abs(difference)===1?'y':'ies'} in this episode.`;
   $('comparison-timing-note').textContent=sectorBenchmark||trainedModel?`Detection timing compares the same ${detection.count} adversar${detection.count===1?'y':'ies'} detected by both layouts; confirmation timing compares the same ${confirmation.count} confirmed by both. Differences are ${trainedModel?result.policyLabel:'archived RL layout'} minus ${trainedModel?`${trainedSensorCount}-sensor`:'eight-sensor'} placement. One episode does not establish overall performance.`:`Detection timing compares the same ${detection.count} adversar${detection.count===1?'y':'ies'} detected by both layouts; confirmation timing compares the same ${confirmation.count} confirmed by both. Negative timing differences mean RL was earlier. Positive warning differences mean more advance notice. One episode does not establish overall performance.`;
}
async function loadComparisonScenario(){
 pause();const sequence=++comparison.sequence,row=comparison.episodes.find(row=>String(row.id)===$('case').value&&String(row.policy)===$('policy').value);
 const layoutId=$('comparison-layout').value||row?.defaultLayout||'directional_balanced_8';
 comparison.scenario=null;comparison.result=null;comparison.episodeId=row?.id;comparison.loading=true;view=null;eventSignature='';nativeImage=null;document.body.classList.remove('native-preview');$('empty').hidden=true;
 $('comparison-budget').textContent='';$('comparison-reasoning').textContent='Loading the sensor model and evaluation details…';$('comparison-status').textContent='Loading both layouts and native sensing results…';$('episode-title').textContent='Preparing comparison';
 for(const id of ['comparison-forecast','comparison-weather','comparison-reports'])$(id).textContent='';
 comparisonClearResults();results();render();controls();showError('');
 try{
  if(!row)throw new Error('No eligible limited-FOV comparison is available. Complete a directional training run to create one; historical Recorded replays remain available.');
  const result=await api('/api/comparison/run',{episodeId:row.id,layoutId});
  if(sequence!==comparison.sequence||mode!=='comparison')return;
    comparison.result=result;comparison.scenario=result.scenario||{};const sectorBenchmark=result.method==='directional_balanced_8',trainedModel=Boolean(result.trainedModel),bestObserved=Boolean(result.bestObservedEpisode),trainedSensorCount=result.rl.maxSensors??result.rl.placements?.length??8,trainedBaseline=result.baseline.label||`${trainedSensorCount} directional sensors`;
    $('comparison-rl-label').textContent=sectorBenchmark?`Archived ${row.policyLabel||$('policy').selectedOptions[0].textContent} layout`:row.policyLabel||$('policy').selectedOptions[0].textContent;
    $('comparison-baseline-label').textContent=trainedModel?trainedBaseline:sectorBenchmark?'Eight directional sensors':'Common-sense coverage';
    $('comparison-control-heading').textContent=trainedModel?trainedBaseline:sectorBenchmark?'Eight directional sensors':'Common-sense coverage';
    $('comparison-control-copy').textContent=trainedModel?`${trainedSensorCount} surface-mounted, limited-FOV thermal camera${trainedSensorCount===1?'':'s'} cover representative bearings before randomized Red approaches are sampled.`:sectorBenchmark?'Eight surface-mounted, limited-FOV thermal cameras are centered on the benchmark sectors before five Red approaches are selected.':'Favor affordable sensors, spread coverage across approaches, and avoid redundant coverage. The baseline stays fixed for each episode.';
    $('episode-help').textContent=bestObserved?'This exact retained placement and baseline use the same sampled training episode, paths, speeds, seed and sensing draws. It is not a general policy estimate.':trainedModel?'The named best checkpoint and baseline use the same held-out native episode, paths, speeds, seed and sensing draws.':sectorBenchmark?'Each episode uses the same five native drones, paths, speeds and seed for both layouts. The archived RL layout has not yet been retrained for eight sensors.':'Each episode uses 60 adversaries in the native Istana environment. Both layouts face the same episode.';
    $('comparison-heading').textContent=trainedModel?`${row.policyLabel} vs ${trainedSensorCount}-sensor directional start`:sectorBenchmark?'Archived RL layout vs eight-directional start':'Reinforcement learning vs common sense';
    $('comparison-budget').textContent=trainedModel?`Same held-out workbench limit: ${fmt(result.rl.budget)} budget units, up to ${result.rl.maxSensors??8} sensors.`:sectorBenchmark?`Same workbench limit: ${fmt(result.rl.budget)} budget units, up to ${result.rl.maxSensors??8} sensors. The archived RL layout still contains only the sensors selected under its older contract.`:`Same deployment limits: ${fmt(result.rl.budget)} budget units, up to ${result.rl.maxSensors??result.baseline.maxSensors??3} sensors.`;
  $('comparison-reasoning').textContent=result.selection?.explanation||result.scenario?.selection?.explanation||'The fixed baseline favors affordable sensors that add new coverage, using approved sites and the same sensor catalogue, spacing and budget as RL. It chooses its layout before observing the realised adversary paths.';
  $('comparison-fairness').textContent=result.fairness?.description||'Both layouts use the current native sensor models, the same adversary episode, and the same deployment limits.';
  $('comparison-forecast').textContent=result.description||row.description||`${row.label} · ${result.rl.frames?.[0]?.threats?.length||60} adversaries · scripted Red approaches in the native Istana environment.`;
  $('comparison-weather').textContent=`Sensor profiles: ${(result.rl.catalogue||[]).filter(c=>(result.rl.availableSensorIds||result.rl.catalogue.map(c=>c.id)).includes(c.id)).map(c=>c.label||c.name||c.id).join(' · ')}.`;
  $('comparison-reports').textContent=bestObserved?(result.observedReplayExact?'This is the exact highest-warning completed training episode and its matched replay. It is retained for inspection and illustration, not presented as a deployable-policy or generalization result. No interception is simulated.':`This older run retains the exact logged best-episode score (${fmt(result.loggedObservedWarningSeconds,2)} s) and layout. Its viewer reconstructs the replay from the original seed because full episode frames were not saved at the time; the viewer's regenerated metric may differ slightly and is not a deployable-policy or generalization result.`) :trainedModel?'This is the automatically saved held-out-selected checkpoint and its matched native evaluation. Warning values are measured; one episode does not establish overall performance. No interception is simulated.':sectorBenchmark?'Warning time, detection and confirmation values are measured from this native workbench replay. Both layouts share the episode, but the archived RL placement was trained for an earlier three-sensor contract, so this is not evidence of a fair trained-policy advantage. No interception is simulated.':'Boson thermal uses directional fields of view and Unreal world line-of-sight; the other native sensor profiles currently use radial coverage. Existing temporal RL policies; sensor directions use the current public-forecast adapter. These checkpoints were not retrained for the directional sensor update. Timely confirmation means a confirmed track at least 4 seconds before objective entry. Sensor probability parameters are simulator assumptions; no interception is simulated.';
    $('comparison-tag').textContent=bestObserved?'RETAINED TRAINING EPISODE':trainedModel?'TRAINED MODEL RESULT':sectorBenchmark?'MATCHED WORKBENCH RESULTS':'MATCHED NATIVE RESULTS';$('comparison-status').textContent=sectorBenchmark||trainedModel?'Measured comparison ready. Play, step or scrub to inspect both layouts and their warning-time results.':'Comparison ready. Play, step or scrub to watch both layouts at the same time.';
  $('comparison-timeline-note').textContent='Both maps share the playback time and map scale.';
    $('provenance').textContent=bestObserved?`Retained highest-warning sampled episode · fixed ${trainedSensorCount}-sensor layout`:trainedModel?`Completed named training run · matched ${trainedSensorCount}-sensor held-out native evaluation`:sectorBenchmark?'Matched native workbench evaluation · archived RL layout vs eight directional sensors':'Matched native Unreal evaluation · fixed common-sense baseline';
  setView({...result.rl,label:row.label,coordinateLabel:'Objective-relative metres'});comparisonResults(result);
 }catch(e){if(sequence===comparison.sequence){comparison.result=null;view=null;showError(`Could not load comparison: ${e.message}`);$('comparison-status').textContent='Choose another episode or select Compare placements again to retry.';$('comparison-verdict').textContent='Comparison unavailable.';render();}}
 finally{if(sequence===comparison.sequence){comparison.loading=false;controls();}}
}
function drawTrainingSeries(canvasId,history,valueFor,{empty,unit,color,baseline}){
 const canvas=$(canvasId),ratio=window.devicePixelRatio||1,width=Math.max(180,canvas.clientWidth),height=Math.max(80,canvas.clientHeight);
 canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
 ctx.strokeStyle='#263343';ctx.lineWidth=1;for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,height*i/4);ctx.lineTo(width,height*i/4);ctx.stroke();}
 const points=(history||[]).map(row=>({episode:row.episode,value:valueFor(row)})).filter(point=>Number.isFinite(point.value));if(!points.length){ctx.fillStyle='#7890a8';ctx.font='11px Segoe UI';ctx.fillText(empty,10,height/2);return;}
 const values=points.map(point=>point.value);if(Number.isFinite(baseline))values.push(baseline);
 const low=Math.min(...values),high=Math.max(...values),span=Math.max(1e-6,high-low),first=points[0].episode,last=points.at(-1).episode;ctx.beginPath();
 points.forEach((point,index)=>{const x=first===last?width/2:8+(point.episode-first)*(width-16)/(last-first),y=8+(high-point.value)*(height-16)/span;if(index)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
 if(points.length===1){ctx.beginPath();ctx.arc(width/2,8+(high-points[0].value)*(height-16)/span,3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
 if(Number.isFinite(baseline)){const y=8+(high-baseline)*(height-16)/span;ctx.beginPath();ctx.setLineDash([5,4]);ctx.moveTo(8,y);ctx.lineTo(width-8,y);ctx.strokeStyle='#f4cf78';ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);}
 ctx.fillStyle='#97a7b9';ctx.font='9px Consolas';ctx.fillText(`${fmt(high,2)}${unit}`,5,10);ctx.fillText(`${fmt(low,2)}${unit}`,5,height-4);
}
function drawTrainingCharts(history){
 drawTrainingSeries('training-chart',history,row=>row.meanWarningSeconds??row.reward,{empty:'Warning-time history appears after episode 1',unit:' s',color:'#6ce8c8'});
 drawTrainingRewardChart(history);
 drawTrainingSeries('training-validation-chart',history,row=>row.validationWarningSeconds,{empty:'Fixed-scenario validation appears after the first policy update',unit:' s',color:'#70b7ff',baseline:training.status.baselineEvaluation?.meanWarningSeconds});
}
function drawTrainingRewardChart(history){
 const canvas=$('training-reward-chart'),ratio=window.devicePixelRatio||1,width=Math.max(180,canvas.clientWidth),height=Math.max(80,canvas.clientHeight);
 canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
 ctx.strokeStyle='#263343';ctx.lineWidth=1;for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,height*i/4);ctx.lineTo(width,height*i/4);ctx.stroke();}
 const rows=history||[],blue=rows.map(row=>Number(row.blueNativeReward??row.nativeReward)),red=rows.map(row=>Number(row.redNativeReward)),values=[...blue,...red].filter(Number.isFinite);
 if(!values.length){ctx.fillStyle='#7890a8';ctx.font='11px Segoe UI';ctx.fillText('Separate Blue/Red rewards appear after episode 1',10,height/2);return;}
 const low=Math.min(...values),high=Math.max(...values),span=Math.max(1e-6,high-low);
 const plot=(series,color)=>{ctx.beginPath();let started=false;series.forEach((value,index)=>{if(!Number.isFinite(value))return;const x=series.length===1?width/2:8+index*(width-16)/(series.length-1),y=8+(high-value)*(height-16)/span;if(started)ctx.lineTo(x,y);else{ctx.moveTo(x,y);started=true;}});ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();};
 plot(blue,'#f4cf78');plot(red,'#ff8b80');ctx.fillStyle='#97a7b9';ctx.font='9px Consolas';ctx.fillText(fmt(high,2),5,10);ctx.fillText(fmt(low,2),5,height-4);ctx.fillStyle='#f4cf78';ctx.fillText('BLUE',width-76,10);ctx.fillStyle='#ff8b80';ctx.fillText('RED',width-38,10);
}
function drawTrainingRewardLog(history){
 const log=$('training-reward-log');log.replaceChildren();const rows=(history||[]).slice(-3).reverse();
 if(!rows.length){log.append(element('p','Reward details appear after episode 1.','helper'));return;}
 for(const entry of rows){
  const section=element('div','','reward-episode');section.append(element('strong',`Episode ${entry.episode} · ${entry.rewardMode==='paired_contractor_delta'?'Warning gain vs matched contractor':'Blue warning objective'} ${fmt(entry.trainingReward??entry.reward,2)} s`));
  for(const component of entry.rewardBreakdown?.components||[]){const line=element('div','',`reward-line${component.value<0?' negative':''}`);line.append(element('b',`${component.value>=0?'+':''}${fmt(component.value,2)}`),element('span',`Blue · ${component.label}`));section.append(line);}
  const red=element('div','','reward-line red');red.append(element('b',`${entry.redNativeReward>=0?'+':''}${fmt(entry.redNativeReward,2)}`),element('span','Red · terminal sensing return (logged only; opponent is not trained)'));section.append(red);
  if(entry.redScenario){const scenario=element('div','','reward-line scenario');scenario.append(element('b',`${fmt(entry.redScenario.spawnRadiusM,0)} m`),element('span',`Red spawn · five distinct seeded sectors · bearings ${entry.redScenario.spawnBearingsDeg.map(value=>`${fmt(value,0)}°`).join(', ')}`));section.append(scenario);}
  for(const change of entry.validationChanges||[]){const line=element('div','','reward-line validation');line.append(element('b',`${change.value>=0?'+':''}${fmt(change.value,2)}`),element('span',change.label));section.append(line);}
  log.append(section);
 }
}
function trainingReplayLabel(selection){
 if(selection.startsWith('checkpoint-'))return `Validated policy checkpoint · episode ${selection.slice(11)}`;
 const labels={latest:`Latest sampled episode ${training.status.episode||'—'}`,initial:'Initial policy · before training',baseline:'Contractor baseline',best:`Best validated policy · episode ${training.status.bestEpisode??'—'}`};
 return labels[selection]||`Sampled episode ${selection}`;
}
function trainingMean(evaluation){return evaluation?.meanWarningSeconds??evaluation?.mean_drone_warning_s;}
function trainingDelta(value){return Number.isFinite(value)?`${value>0?'+':''}${fmt(value,2)} s`:'—';}
function renderTrainingEvidence(status){
 const latest=status.history?.at(-1),baseline=trainingMean(status.baselineEvaluation),initial=trainingMean(status.initialEvaluation),best=status.bestValidationWarningSeconds??status.bestWarningSeconds;
 $('training-baseline-warning').textContent=Number.isFinite(baseline)?`${fmt(baseline,2)} s`:'Awaiting evaluation';
 $('training-initial-warning').textContent=Number.isFinite(initial)?`Initial policy: ${fmt(initial,2)} s on the same scenarios`:'Initial policy: awaiting evaluation';
 $('training-sample-warning').textContent=latest?`${fmt(latest.meanWarningSeconds??latest.reward,2)} s`:'—';
 $('training-sample-note').textContent=latest?`Episode ${latest.episode} · one sampled scenario${latest.pairedTraining?` · contractor ${fmt(latest.pairedTraining.contractorWarningSeconds,2)} s · gain ${trainingDelta(latest.trainingReward)}`:''}`:'No completed episode';
 $('training-best-warning').textContent=Number.isFinite(best)?`${fmt(best,2)} s`:'Awaiting evaluation';
 $('training-validation-delta').textContent=Number.isFinite(best)&&Number.isFinite(baseline)?`${trainingDelta(best-baseline)} vs contractor · episode ${status.bestEpisode??0}`:'Awaiting fixed-scenario evaluation';
 const test=status.testEvaluation,testMean=trainingMean(test);
 $('training-test-warning').textContent=Number.isFinite(testMean)?`${fmt(testMean,2)} s`:'Not tested yet';
 $('training-test-delta').textContent=Number.isFinite(testMean)?`${trainingDelta(test.deltaSeconds)} vs contractor · ${test.cases?.length??test.cases??'separate'} scenarios`:'Runs after policy selection';
 $('training-evaluation-note').textContent=`Policy selection uses the mean of ${status.outputDirectory?status.validationCases:(Number($('training-validation').value)||8)} fixed validation scenarios. A sampled episode can score better or worse just because its scenario differs. Higher warning is better; missed detections count as zero.`;
 const exploration=status.exploration;
 $('training-exploration-summary').textContent=exploration&&Object.keys(exploration).length?`${exploration.uniqueLayouts??0} distinct layouts sampled (placement order ignored). Latest: ${exploration.sensorsChangedFromInitial??latest?.sensorsChangedFromInitial??'—'} sensors changed from the initial layout. ${exploration.localEdits?'Local edits start with equal probabilities':`Initial exploration target ${pct(exploration.explorationProbability)}`} · action diversity ${pct(exploration.meanNormalizedEntropy)} (0% = certain, 100% = uniform).`:'Layout diversity and exploration measurements appear during training.';
 const probe=status.baselineProbe;$('training-probe-summary').hidden=!probe;
 if(probe)$('training-probe-summary').textContent=`Baseline check: ${probe.candidates?.length??probe.candidates??0} nearby layouts evaluated. Best measured difference: ${trainingDelta(probe.bestDeltaSeconds)}. ${probe.improvesBaseline?'A tested alternative improved the baseline.':'These tested alternatives did not improve the baseline.'} This check is separate from the trained policy.`;
 syncTrainingReplayOptions(status);renderTrainingHistory(status);
}
function syncTrainingReplayOptions(status){
 const select=$('training-replay'),selected=training.replaySelection,options=[['latest','Follow latest sampled episode']];
 if(status.baselineEvaluation)options.push(['baseline','Contractor baseline · before training']);
 if(status.initialEvaluation||status.initialLayout)options.push(['initial','Initial policy · before training']);
 if(Number.isFinite(status.bestValidationWarningSeconds??status.bestWarningSeconds))options.push(['best',`Best validated policy · episode ${status.bestEpisode??0}`]);
 for(const episode of [...new Set(status.checkpoints||[])].filter(value=>Number.isInteger(value)&&value>0&&value<=status.episode).sort((a,b)=>a-b)){options.push([`checkpoint-${episode}`,`Policy checkpoint · episode ${episode}`]);options.push([String(episode),`Sampled layout · episode ${episode}`]);}
 if(!options.some(([value])=>value===selected))options.push([selected,trainingReplayLabel(selected)]);
 const signature=JSON.stringify(options);if(select.dataset.options!==signature){select.replaceChildren();for(const [value,label] of options){const option=element('option',label);option.value=value;select.append(option);}select.dataset.options=signature;}
 select.value=selected;$('training-episode-number').max=String(Math.max(1,status.episode||0));
}
function renderTrainingHistory(status){
 const history=status.history||[],signature=`${status.outputDirectory}-${status.episode}-${history.length}-${history.at(-1)?.validationWarningSeconds}`;
 if(signature===training.historySignature)return;training.historySignature=signature;
 const count=status.historyCount??status.episode??history.length,rows=history.slice(-100).reverse();
 $('training-history-count').textContent=`${count} episodes${count>100?' · latest 100 shown':''}`;
 const body=$('training-history-rows');body.replaceChildren();
 for(const row of rows){
  const tr=element('tr','');tr.append(element('td',String(row.episode)),element('td',`${fmt(row.meanWarningSeconds??row.reward,2)} s`),element('td',pct(row.detectedFraction)),element('td',Number.isFinite(row.sensorsChangedFromInitial)?String(row.sensorsChangedFromInitial):'—'),element('td',Number.isFinite(row.validationWarningSeconds)?`${fmt(row.validationWarningSeconds,2)} s`:'Not evaluated'));
  const action=element('td',''),button=element('button','Replay');button.className='training-history-replay';button.setAttribute('aria-label',`Replay training episode ${row.episode}`);button.onclick=()=>selectTrainingReplay(String(row.episode));action.append(button);tr.append(action);body.append(tr);
 }
}
async function selectTrainingReplay(selection){
 if(training.replayLoading)return;
 const checkpoint=/^checkpoint-\d+$/.test(selection);
 if(checkpoint&&!(training.status.checkpoints||[]).includes(Number(selection.slice(11))))return showError('Choose a saved policy checkpoint.');
 if(!checkpoint&&!['latest','initial','baseline','best'].includes(selection)&&(!/^\d+$/.test(selection)||Number(selection)<1||Number(selection)>training.status.episode))return showError(`Choose a completed episode between 1 and ${training.status.episode||0}.`);
 pause();showError('');const sequence=++training.replaySequence;
 if(selection==='latest'){
  training.replaySelection='latest';training.replayViewer=null;training.lastViewerEpisode=-1;applyTrainingStatus(training.status);$('training-replay-note').textContent='Following the latest sampled episode. Selected earlier replays stay on screen while training continues.';return;
 }
 training.replayLoading=true;controls();$('training-replay-note').textContent=`Loading ${trainingReplayLabel(selection).toLowerCase()}…`;
 try{
  const result=await api(`/api/training/replay/${encodeURIComponent(selection)}`);if(sequence!==training.replaySequence)return;
  training.replaySelection=selection;training.replayViewer=result.viewer||result;syncTrainingReplayOptions(training.status);
  if(mode==='training')setView(training.replayViewer);
  $('training-replay-note').textContent=`Showing ${result.label||trainingReplayLabel(selection)}. This replay stays selected while training continues. ${checkpoint||['initial','baseline','best'].includes(selection)?'Replay metrics describe one scenario; the cards show scenario averages.':''}`;
 }catch(e){showError(`Could not load training replay: ${e.message}`);$('training-replay-note').textContent=`Still showing ${trainingReplayLabel(training.replaySelection).toLowerCase()}.`;syncTrainingReplayOptions(training.status);}
 finally{training.replayLoading=false;controls();}
}
async function exportTrainingData(){
 $('training-export').disabled=true;showError('');
 try{const data=await api('/api/training/export'),blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`${(training.status.modelName||'training-run').replace(/[^a-z0-9_-]+/gi,'-')}-evidence.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 catch(e){showError(`Could not download run data: ${e.message}`);}finally{controls();}
}
function updateTrainingInitializationHelp(status=training.status){
 const algorithm=status.running?status.algorithm:($('training-algorithm').value||status.algorithm);
 if(algorithm==='local_ppo')$('training-initialization-help').textContent='PPO learns nearby moves and rotations for each sensor in the selected contractor layout. Every edit preserves the selected sensor inventory and full-layout legality. The starting layout has no probability advantage.';
 else if(status.running&&status.initialLayout?.coverage_intent)$('training-initialization-help').textContent=status.initialLayout.coverage_intent+' The layout initializes trainable logits and is not locked.';
 else $('training-initialization-help').textContent='The selected layout gives the policy a starting preference. Training also explores other legal placements.';
}
function applyTrainingStatus(status){
 training.status=status;const total=status.totalEpisodes||0,episode=status.episode||0,latest=status.history?.at(-1);
 $('training-progress').textContent=`${episode} / ${total}`;$('training-progress-bar').style.width=`${total?Math.min(100,episode/total*100):0}%`;
 $('training-warning').textContent=latest?`${fmt(latest.meanWarningSeconds??latest.reward,2)} s`:'—';$('training-blue-reward').textContent=latest?fmt(latest.blueNativeReward??latest.nativeReward,2):'—';$('training-red-reward').textContent=latest?fmt(latest.redNativeReward,2):'—';$('training-detected').textContent=latest?pct(latest.detectedFraction):'—';
 const labels={idle:'Ready to train',connecting:'Connecting to native Unreal…',initializing:'Evaluating the initial policy and contractor baseline…',probing:'Checking nearby baseline layouts…',training:'Native episodes are running',validating:'Evaluating fixed validation scenarios…',evaluating:'Evaluating native scenarios…',stopping:'Stopping after the current native action…',stopped:'Training stopped; recordings retained without a new final test or automatic publication',complete:'Training complete',failed:'Training failed'};
 $('training-status').textContent=status.phase==='idle'?'Ready to train. Choose a starting placement and settings above.':`${labels[status.phase]||status.phase}. ${status.modelName?`Model: ${status.modelName}.`:''} ${status.algorithmLabel||''}. ${status.sensorCount?`${status.sensorCount} sensors.`:''} ${status.initializationLabel||''}`.trim();
 const checkpointText=status.checkpoints?.length?` Policy snapshots: ${status.checkpoints.join(', ')}.`:'';
 $('training-output').textContent=status.registeredModel?`Saved “${status.registeredModel.name}” (${status.registeredModel.algorithmLabel}) as a validation-selected policy from episode ${status.registeredModel.bestEpisode} (${fmt(status.registeredModel.bestWarningSeconds,2)} s validation warning), plus exact best observed episode ${status.registeredModel.bestObservedEpisode} (${fmt(status.registeredModel.bestObservedWarningSeconds,2)} s sampled warning). Both are in Compare placements.${checkpointText}`:status.outputDirectory?`Artifacts: ${status.outputDirectory}.${checkpointText}`:'';
 updateTrainingInitializationHelp(status);
 renderTrainingEvidence(status);drawTrainingCharts(status.history);drawTrainingRewardLog(status.history);
 const signature=`${status.outputDirectory}-${status.episode}-${status.phase}`;if(mode==='training'&&training.replaySelection==='latest'&&status.viewer&&(training.lastViewerEpisode!==signature||view?.mode!=='training')&&!playing&&!training.replayLoading){const autoReplay=status.running&&(status.viewer.frames?.length||0)>1;training.lastViewerEpisode=signature;setView({...status.viewer,trainingReplayLabel:trainingReplayLabel(status.viewerSelection||'latest')});if(autoReplay){$('speed').value=String(Math.max(4,Number($('speed').value)||1));elapsed=status.viewer.frames[0].time;lastWall=0;playing=true;controls();}}
 else if(mode==='training'&&training.replaySelection!=='latest'&&training.replayViewer&&view!==training.replayViewer)setView(training.replayViewer);
 if(status.phase==='failed'&&status.error)showError(status.error);controls();connectionStatus();
}
async function refreshTraining(){
 if(training.polling)return;training.polling=true;
 try{const wasRunning=training.status.running,status=await api('/api/training/status');applyTrainingStatus(status);if(wasRunning&&status.registeredModel&&(status.phase==='complete'||status.phase==='stopped')){const session=await api('/api/session');syncModelCatalog(session);}}
 catch(e){showError(`Could not read training status: ${e.message}`);}
 finally{training.polling=false;if(mode==='training'&&training.status.running)training.timer=setTimeout(refreshTraining,750);}
}
async function trainingAction(action){
 if(busy)return;busy=true;busyOperation=`training-${action}`;controls();showError('');
 try{
  const payload=action==='start'?{name:$('training-name').value,algorithm:$('training-algorithm').value,episodes:Number($('training-episodes').value),batchSize:Number($('training-batch').value),sensorCount:Number($('training-sensors').value),seed:Number($('training-seed').value),initialization:$('training-initialization').value,explorationProbability:Number($('training-exploration').value),validationCases:Number($('training-validation').value),checkpointInterval:Number($('training-checkpoint').value)}:{};
  const status=await api(`/api/training/${action}`,payload);connected=false;
  if(action==='start'){++training.replaySequence;training.replaySelection='latest';training.replayViewer=null;training.lastViewerEpisode=-1;training.historySignature='';pause();view=null;$('training-replay-note').textContent='Following the latest sampled episode. Selected earlier replays stay on screen while training continues.';$('empty').hidden=false;$('empty-title').textContent='Preparing the training run';$('empty-copy').textContent='The initial policy and contractor baseline are evaluated before training begins.';$('empty-note').textContent='Recorded layouts appear as their evaluations finish.';$('episode-title').textContent='Preparing initial evaluation';sensorDetails();results();render();}
  applyTrainingStatus(status);
  if(status.running){if(training.timer)clearTimeout(training.timer);training.timer=setTimeout(refreshTraining,250);}
 }catch(e){showError(e.message);}finally{busy=false;busyOperation='';controls();connectionStatus();}
}
for(const id of ['profile','case'])$(id).addEventListener('change',loadReplay);
$('policy').addEventListener('change',()=>{if(mode==='comparison'){comparison.selectedPolicy=$('policy').value;fillEpisodeOptions();}loadReplay();});
$('comparison-layout').addEventListener('change',loadComparisonScenario);
for(const id of ['ranges','trails','drone-rings','sites'])$(id).addEventListener('change',()=>render());
$('recorded-mode').onclick=()=>switchMode('recorded');$('live-mode').onclick=()=>switchMode('live');$('comparison-mode').onclick=()=>switchMode('comparison');$('training-mode').onclick=()=>switchMode('training');
$('training-start').onclick=()=>trainingAction('start');$('training-stop').onclick=()=>trainingAction('stop');
$('training-replay-load').onclick=()=>selectTrainingReplay($('training-replay').value);
$('training-episode-open').onclick=()=>selectTrainingReplay($('training-episode-number').value.trim());
$('training-episode-number').onkeydown=event=>{if(event.key==='Enter')selectTrainingReplay($('training-episode-number').value.trim());};
$('training-export').onclick=exportTrainingData;
$('training-algorithm').onchange=()=>{$('training-algorithm-help').textContent=$('training-algorithm').selectedOptions[0]?.dataset.description||'';const local=$('training-algorithm').value==='local_ppo',untrained=$('training-initialization').querySelector('[value="untrained"]');if(untrained)untrained.disabled=local;if(local&&$('training-initialization').value==='untrained')$('training-initialization').value='directional_balanced_8';updateTrainingInitializationHelp();controls();};
$('connect').onclick=()=>{pause();liveAction(connected?'disconnect':'connect');};
$('plan').onclick=()=>{pause();liveAction(isSavedLayout($('live-policy').value)?'preview':'reset',{seed:Number($('seed').value),policy:$('live-policy').value});};
$('live-policy').onchange=()=>{$('plan').textContent=isSavedLayout($('live-policy').value)?'Apply saved layout':'Plan new episode';};
$('presentation').onchange=()=>{document.body.classList.toggle('presentation',$('presentation').checked);$('live-policy').size=$('presentation').checked?9:1;render();};
$('play').onclick=()=>{if(playing)return pause();if(!view)return;if(mode!=='live'&&frameIndex===view.frames.length-1){frameIndex=0;render();}else if(mode==='live'&&frameIndex<view.frames.length-1){frameIndex=view.frames.length-1;render();}elapsed=view.frames[frameIndex].time;lastWall=0;liveWall=0;playing=true;controls();render();};
$('step').onclick=()=>{pause();if(mode==='live')liveAction('step');else if(view){frameIndex=Math.min(frameIndex+1,view.frames.length-1);render();}};
$('restart').onclick=()=>{pause();frameIndex=0;elapsed=0;render();};
$('timeline').oninput=()=>{pause();const requested=Number($('timeline').value);if(mode==='live'){frameIndex=0;for(let i=1;i<view.frames.length&&view.frames[i].completedSteps<=requested;i++)frameIndex=i;}else frameIndex=requested;render();};
$('speed').onchange=()=>render();
const mapResizeObserver=new ResizeObserver(()=>draw(view?.frames[frameIndex]));
for(const id of ['map','comparison-map-rl','comparison-map-baseline'])mapResizeObserver.observe($(id).parentElement);
api('/api/session').then(async session=>{token=session.token;catalog=session.replays;connected=session.status.connected;training.status=session.training||training.status;const limits=session.trainingLimits||{};for(const [id,min,max] of [['training-episodes',limits.minEpisodes,limits.maxEpisodes],['training-sensors',limits.minSensors,limits.maxSensors]]){if(Number.isInteger(min))$(id).min=String(min);if(Number.isInteger(max))$(id).max=String(max);}for(const item of session.trainingAlgorithms||[]){const option=element('option',item.label);option.value=item.id;option.dataset.description=item.description;$('training-algorithm').append(option);}if($('training-algorithm').querySelector('[value="local_ppo"]'))$('training-algorithm').value='local_ppo';for(const item of session.trainingInitializations||[]){const option=element('option',item.label);option.value=item.id;$('training-initialization').append(option);}if($('training-initialization').querySelector('[value="directional_balanced_8"]'))$('training-initialization').value='directional_balanced_8';$('training-algorithm').onchange();syncModelCatalog(session);applyTrainingStatus(training.status);await loadReplay();connectionStatus();}).catch(e=>showError(e.message));
controls();requestAnimationFrame(animate);
