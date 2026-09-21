'use strict';
const $ = id => document.getElementById(id);
let token='', catalog=[], view=null, frameIndex=0, mode='recorded', playing=false, busy=false, busyOperation='', connected=false;
let elapsed=0, lastWall=0, liveWall=0, loadSequence=0;
let nativeImage=null, eventSignature='';
const comparison={episodes:[],scenario:null,result:null,loading:false,episodeId:null,sequence:0};
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
 const label=mode==='comparison'?'Comparison · native evaluation':mode==='recorded'?'Replay ready':connected?'Unreal connected':'Unreal disconnected';
 $('connection-label').textContent=label;
 $('connection').firstChild.style.background=mode==='live'&&!connected?'#ffb18a':'#6ce8c8';
 $('connect').textContent=connected?'Disconnect':'Connect to Unreal';
}
function controls(){
 const blocking=(busy&&busyOperation!=='step')||comparison.loading, liveEnded=Boolean(mode==='live'&&view?.ended), draft=mode==='comparison'&&!comparison.result;
 $('play').textContent=playing?'Pause':mode==='live'?'Run episode':'Play replay';
 $('play').disabled=!view||blocking||liveEnded||draft;
 $('step').disabled=!view||playing||busy||blocking||liveEnded||draft;
 $('restart').disabled=!view||mode==='live'||blocking||draft;
 $('timeline').disabled=!view||blocking||draft;$('speed').disabled=!view||draft;
 $('connect').disabled=busy||playing;$('plan').disabled=!connected||busy||playing;
 for(const id of ['recorded-mode','live-mode','comparison-mode'])$(id).disabled=busy||playing||comparison.loading;
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
 $('active-policy').textContent=mode==='live'&&view?`Active: ${view.policy}`:'No active planner.';
 $('surface-status').textContent=mode==='live'&&view?(view.surfaceMounted?'Surface-mounted · unsupported sites excluded.':'Legacy build: surface support not enforced. Rebuild Unreal.'):'Surface support is checked by Unreal before deployment.';
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
 $('results-tag').textContent=mode==='recorded'?'RECORDED FINAL':mode==='comparison'?(m?'NATIVE FINAL':'LOADING EPISODE'):m?'MEASURED FINAL':'AWAITING EPISODE END';
 $('outcome-label').textContent=mode==='recorded'?'Recorded final result':mode==='comparison'?'Native final result':'Episode status';
 $('detected-label').textContent=mode!=='live'?'Detected so far':'Public tracks';
 $('result-note').textContent=mode==='recorded'?'Synthetic sensing outcomes. Timely confirmation means detection before the deadline; no interception is simulated. Warning is unavailable when the replay does not contain the exact metric.':mode==='comparison'?'Matched native sensing outcomes. Both layouts face the same adversary episode and deployment limits.':'Live metrics appear when the native episode ends. Directional thermal sensors use Unreal world-static line-of-sight; probability parameters remain simulator assumptions. Warning values are measured lower bounds from detection to 20 m objective-zone arrival.';
}
function setView(data){view=data;nativeImage=null;eventSignature='';if(data.nativePreview){const image=new Image();image.onload=()=>{if(view===data){nativeImage=image;render();}};image.src=data.nativePreview.image;}$('map').setAttribute('aria-label',data.nativePreview?'Native Unreal sensor placement preview':'Top-down simulation view');document.body.classList.toggle('native-preview',Boolean(data.nativePreview));frameIndex=0;elapsed=0;$('empty').hidden=true;syncTimeline();$('episode-title').textContent=view.label;$('coordinates').textContent=view.coordinateLabel;sensorDetails();results();render();controls();}
async function loadReplay(){
 if(mode==='comparison')return loadComparisonScenario();
 pause();const seq=++loadSequence;const row=catalog.find(r=>r.profile===$('profile').value&&String(r.policy)===$('policy').value&&String(r.case)===$('case').value);
 if(!row)return showError('This recorded case is unavailable.');
 try{const data=await api(`/api/replay/${row.id}`);if(seq!==loadSequence||mode!=='recorded')return;setView(data);showError('');}catch(e){showError(e.message);}
}
async function switchMode(next){
 pause();mode=next;++loadSequence;showError('');
 for(const id of ['recorded','live','comparison'])$(id+'-mode').setAttribute('aria-pressed',String(mode===id));
 document.body.classList.toggle('comparison-mode',mode==='comparison');
 if(mode==='comparison')$('comparison-panel').insertBefore($('transport'),$('comparison-results'));else $('single-map-container').after($('transport'));
 $('recorded-controls').hidden=mode==='live';$('profile').hidden=mode==='comparison';$('profile-label').hidden=mode==='comparison';$('deployment-controls').hidden=mode==='comparison';$('single-map-container').hidden=mode==='comparison';
 $('case-label').textContent=mode==='comparison'?'Episode':'Recorded case';$('policy-label').textContent=mode==='comparison'?'RL policy':'Trained checkpoint';$('episode-help').textContent=mode==='comparison'?'Each episode uses 60 adversaries in the native Istana environment. Both layouts face the same episode.':'All three checkpoints and all 18 published cases are available, including failures.';fillEpisodeOptions();$('live-controls').hidden=mode!=='live';$('comparison-controls').hidden=mode!=='comparison';$('comparison-panel').hidden=mode!=='comparison';
 $('source-badge').textContent=mode==='recorded'?'RECORDED · SYNTHETIC':mode==='comparison'?'COMPARISON · NATIVE UNREAL':'LIVE UNREAL · SYNTHETIC';
 $('provenance').textContent=mode==='recorded'?'Published temporal-v6 evidence · 18 recorded cases':mode==='comparison'?'Matched native Unreal evaluation · fixed common-sense baseline':'Local Unreal bridge · scripted Red · experimental Blue';
 if(mode==='recorded')await loadReplay();else if(mode==='comparison')await loadComparisonScenario();else{view=null;$('empty').hidden=false;$('episode-title').textContent='Awaiting live episode';$('clock').textContent='00:00.0';$('progress').textContent='Fixed-step simulation';sensorDetails();results();render();controls();}
 connectionStatus();
}
function render(){
 const frame=view?.frames[frameIndex];syncTimeline();
 $('clock').textContent=timeText(frame?.time||0);
 $('progress').textContent=view?mode==='live'?`Step ${frame?.completedSteps??0} / ${liveTimelineMax()} · ${view.ended?'ended':playing?`running at ${$('speed').value}×`:'paused'}`:`Frame ${frameIndex+1} / ${view.frames.length} · ${fmt(view.frames.at(-1).time)} s`:'No episode';
 $('threat-count').textContent=frame?frame.threats.length:'—';
 $('detected-count').textContent=frame?(mode==='live'?(frame.tracks||[]).length:frame.threats.filter(t=>t.ever_detected).length):'—';
 $('confirmed-count').textContent=frame?(mode==='live'?(frame.tracks||[]).filter(t=>t.confirmed).length:frame.threats.filter(t=>t.tracked).length):'—';
 $('outcome').textContent=view?(view.outcome==='defended'?'Timely sensing':view.outcome==='running'?'In progress':mode==='live'?'Episode ended':view.outcome.replaceAll('_',' ')):'—';
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
  $(`comparison-${method}-now`).textContent=current?`${detected} detected · ${confirmed} confirmed at ${fmt(time,1)} s`:'Awaiting episode';
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
}
function comparisonControls(){
 const locked=comparison.loading||busy;
 for(const id of ['profile','policy','case'])$(id).disabled=locked;
}
function comparisonClearResults(){
 for(const method of ['rl','baseline','delta'])for(const metric of ['detected','confirmed','timely','cost','first-detection','detection-time','confirmation-time','warning'])$(`comparison-${method}-${metric}`).textContent='—';
 for(const method of ['rl','baseline'])$(`comparison-${method}-layout`).textContent='Loading layout…';
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
 const summaries={rl:comparisonSummary(result.rl,result.metrics?.rl),baseline:comparisonSummary(result.baseline,result.metrics?.baseline)};
 for(const method of ['rl','baseline']){
  const summary=summaries[method],data=result[method];
  for(const metric of ['detected','confirmed','timely'])$(`comparison-${method}-${metric}`).textContent=Number.isFinite(summary[metric])?`${summary[metric]} / ${summary.total}`:'Unavailable';
  $(`comparison-${method}-cost`).textContent=`${fmt(summary.cost)} / ${fmt(data.budget)}`;
  $(`comparison-${method}-warning`).textContent=Number.isFinite(summary.warning)?`${fmt(summary.warning,2)} s`:'Unavailable';
  $(`comparison-${method}-layout`).textContent=`${data.placements.length} sensors · ${fmt(summary.cost)} / ${fmt(data.budget)} cost`;
  $(`comparison-map-${method}`).setAttribute('aria-label',`${method==='rl'?'Reinforcement learning':'Fixed common-sense'} layout: ${data.placements.length} sensors, ${summary.detected} of ${summary.total} adversaries detected by episode end. Shared playback shows sensing over time.`);
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
 $('comparison-verdict').textContent=difference===0?'Both layouts detected the same number of adversaries in this episode.':`RL detected ${Math.abs(difference)} ${difference>0?'more':'fewer'} adversar${Math.abs(difference)===1?'y':'ies'} in this episode.`;
 $('comparison-timing-note').textContent=`Detection timing compares the same ${detection.count} adversar${detection.count===1?'y':'ies'} detected by both layouts; confirmation timing compares the same ${confirmation.count} confirmed by both. Negative timing differences mean RL was earlier. Positive warning differences mean more advance notice. One episode does not establish overall performance.`;
}
async function loadComparisonScenario(){
 pause();const sequence=++comparison.sequence,row=comparison.episodes.find(row=>String(row.id)===$('case').value&&String(row.policy)===$('policy').value);
 comparison.scenario=null;comparison.result=null;comparison.episodeId=row?.id;comparison.loading=true;view=null;eventSignature='';nativeImage=null;document.body.classList.remove('native-preview');$('empty').hidden=true;
 $('comparison-budget').textContent='';$('comparison-reasoning').textContent='Loading the sensor model and evaluation details…';$('comparison-status').textContent='Loading both layouts and native sensing results…';$('episode-title').textContent='Preparing comparison';
 for(const id of ['comparison-forecast','comparison-weather','comparison-reports'])$(id).textContent='';
 comparisonClearResults();results();render();controls();showError('');
 try{
  if(!row)throw new Error('No paired native evaluation is available for this policy. Choose another policy.');
  const result=await api('/api/comparison/run',{episodeId:row.id});
  if(sequence!==comparison.sequence||mode!=='comparison')return;
  comparison.result=result;comparison.scenario=result.scenario||{};
  $('comparison-rl-label').textContent=row.policyLabel||$('policy').selectedOptions[0].textContent;
  $('comparison-budget').textContent=`Same deployment limits: ${fmt(result.rl.budget)} budget units, up to ${result.rl.maxSensors??result.baseline.maxSensors??3} sensors.`;
  $('comparison-reasoning').textContent=result.selection?.explanation||result.scenario?.selection?.explanation||'The fixed baseline favors affordable sensors that add new coverage, using approved sites and the same sensor catalogue, spacing and budget as RL. It chooses its layout before observing the realised adversary paths.';
  $('comparison-fairness').textContent=result.fairness?.description||'Both layouts use the current native sensor models, the same adversary episode, and the same deployment limits.';
  $('comparison-forecast').textContent=result.description||row.description||`${row.label} · ${result.rl.frames?.[0]?.threats?.length||60} adversaries · scripted Red approaches in the native Istana environment.`;
  $('comparison-weather').textContent=`Sensor profiles: ${(result.rl.catalogue||[]).filter(c=>(result.rl.availableSensorIds||result.rl.catalogue.map(c=>c.id)).includes(c.id)).map(c=>c.label||c.name||c.id).join(' · ')}.`;
  $('comparison-reports').textContent='Boson thermal uses directional fields of view and Unreal world line-of-sight; the other native sensor profiles currently use radial coverage. Existing temporal RL policies; sensor directions use the current public-forecast adapter. These checkpoints were not retrained for the directional sensor update. Timely confirmation means a confirmed track at least 4 seconds before objective entry. Sensor probability parameters are simulator assumptions; no interception is simulated.';
  $('comparison-tag').textContent='MATCHED NATIVE RESULTS';$('comparison-status').textContent='Comparison ready. Play, step or scrub to watch both layouts at the same time.';
  setView({...result.rl,label:row.label,coordinateLabel:'Objective-relative metres'});comparisonResults(result);
 }catch(e){if(sequence===comparison.sequence){comparison.result=null;view=null;showError(`Could not load comparison: ${e.message}`);$('comparison-status').textContent='Choose another episode or select Compare placements again to retry.';$('comparison-verdict').textContent='Comparison unavailable.';render();}}
 finally{if(sequence===comparison.sequence){comparison.loading=false;controls();}}
}
for(const id of ['profile','case'])$(id).addEventListener('change',loadReplay);
$('policy').addEventListener('change',()=>{if(mode==='comparison')fillEpisodeOptions();loadReplay();});
for(const id of ['ranges','trails','drone-rings','sites'])$(id).addEventListener('change',()=>render());
$('recorded-mode').onclick=()=>switchMode('recorded');$('live-mode').onclick=()=>switchMode('live');$('comparison-mode').onclick=()=>switchMode('comparison');
$('connect').onclick=()=>{pause();liveAction(connected?'disconnect':'connect');};
$('plan').onclick=()=>{pause();liveAction($('live-policy').value.startsWith('saved-')?'preview':'reset',{seed:Number($('seed').value),policy:$('live-policy').value});};
$('live-policy').onchange=()=>{$('plan').textContent=$('live-policy').value.startsWith('saved-')?'Apply saved layout':'Plan new episode';};
$('presentation').onchange=()=>{document.body.classList.toggle('presentation',$('presentation').checked);$('live-policy').size=$('presentation').checked?9:1;render();};
$('play').onclick=()=>{if(playing)return pause();if(!view)return;if(mode!=='live'&&frameIndex===view.frames.length-1){frameIndex=0;render();}else if(mode==='live'&&frameIndex<view.frames.length-1){frameIndex=view.frames.length-1;render();}elapsed=view.frames[frameIndex].time;lastWall=0;liveWall=0;playing=true;controls();render();};
$('step').onclick=()=>{pause();if(mode==='live')liveAction('step');else if(view){frameIndex=Math.min(frameIndex+1,view.frames.length-1);render();}};
$('restart').onclick=()=>{pause();frameIndex=0;elapsed=0;render();};
$('timeline').oninput=()=>{pause();const requested=Number($('timeline').value);if(mode==='live'){frameIndex=0;for(let i=1;i<view.frames.length&&view.frames[i].completedSteps<=requested;i++)frameIndex=i;}else frameIndex=requested;render();};
$('speed').onchange=()=>render();
const mapResizeObserver=new ResizeObserver(()=>draw(view?.frames[frameIndex]));
for(const id of ['map','comparison-map-rl','comparison-map-baseline'])mapResizeObserver.observe($(id).parentElement);
api('/api/session').then(async session=>{token=session.token;catalog=session.replays;comparison.episodes=session.comparisonEpisodes||[];connected=session.status.connected;if(session.savedModels?.length){const group=document.createElement('optgroup');group.label='Archived native layouts (no inference)';for(const model of session.savedModels){const option=document.createElement('option');option.value=model.id;option.textContent=model.label;group.append(option);}$('live-policy').append(group);}await loadReplay();connectionStatus();}).catch(e=>showError(e.message));
controls();requestAnimationFrame(animate);
