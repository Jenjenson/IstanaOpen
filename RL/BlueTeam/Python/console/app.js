'use strict';
const $ = id => document.getElementById(id);
let token='', catalog=[], view=null, frameIndex=0, mode='recorded', playing=false, busy=false, busyOperation='', connected=false;
let elapsed=0, lastWall=0, liveWall=0, loadSequence=0;
let nativeImage=null, eventSignature='';
const comparison={scenario:null,result:null,placements:[],method:'baseline',loading:false,replayId:null,mapTransform:null,sequence:0,revealed:false};
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
 const label=mode==='comparison'?'Comparison · local simulation':mode==='recorded'?'Replay ready':connected?'Unreal connected':'Unreal disconnected';
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
 $('results-tag').textContent=mode==='recorded'?'RECORDED FINAL':mode==='comparison'?(m?'RESCORED FINAL':'LAYOUT PREVIEW'):m?'MEASURED FINAL':'AWAITING EPISODE END';
 $('outcome-label').textContent=mode==='recorded'?'Recorded final result':mode==='comparison'?'Rescored final result':'Episode status';
 $('detected-label').textContent=mode!=='live'?'Detected so far':'Public tracks';
 $('result-note').textContent=mode==='recorded'?'Synthetic sensing outcomes. Timely confirmation means detection before the deadline; no interception is simulated. Warning is unavailable when the replay does not contain the exact metric.':mode==='comparison'?'Matched synthetic sensing outcomes. Both layouts use the same archived scenario and sensing draws. Timely confirmation is before the objective-zone deadline; no interception is simulated.':'Live metrics appear when the native episode ends. Directional thermal sensors use Unreal world-static line-of-sight; probability parameters remain simulator assumptions. Warning values are measured lower bounds from detection to 20 m objective-zone arrival.';
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
 $('recorded-controls').hidden=mode==='live';$('live-controls').hidden=mode!=='live';$('comparison-controls').hidden=mode!=='comparison';$('comparison-panel').hidden=mode!=='comparison';
 $('source-badge').textContent=mode==='recorded'?'RECORDED · SYNTHETIC':mode==='comparison'?'COMPARISON · SYNTHETIC':'LIVE UNREAL · SYNTHETIC';
 $('provenance').textContent=mode==='recorded'?'Published temporal-v6 evidence · 18 recorded cases':mode==='comparison'?'Matched synthetic evaluation · archived scenarios and RL layouts':'Local Unreal bridge · scripted Red · experimental Blue';
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
 if(mode==='comparison'&&!comparison.result){$('progress').textContent='Layout preview · evaluate to reveal paths';for(const id of ['threat-count','detected-count','confirmed-count','outcome'])$(id).textContent='—';}
 draw(frame);events(frame);
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
function draw(frame){
 const canvas=$('map'),bounds=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 const pixelWidth=Math.max(1,Math.round(bounds.width*dpr)),pixelHeight=Math.max(1,Math.round(bounds.height*dpr));
 if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
 const w=bounds.width,h=bounds.height;ctx.fillStyle='#0b141e';ctx.fillRect(0,0,w,h);
 if(view?.nativePreview){if(nativeImage){const s=Math.min(w/1920,h/1080),ox=(w-1920*s)/2,oy=(h-1080*s)/2;ctx.drawImage(nativeImage,ox,oy,1920*s,1080*s);for(const a of view.nativePreview.anchors){const x=ox+a.x*s,y=oy+a.y*s;ctx.strokeStyle='#9fffe3';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(x,y+2,12,6,0,0,Math.PI*2);ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#ccfff0';ctx.fillText('Sensor',x+16,y-10);}}return;}
 const pts=view?view.frames.flatMap(f=>f.threats.map(t=>t.position)).concat(view.placements.map(p=>p.position),mode==='comparison'?view.sites||[]:[]):[];
 if(mode==='comparison'&&!comparison.result)for(const track of frame?.tracks||[])pts.push(track.position);
 let extent=120;for(const p of pts)extent=Math.max(extent,Math.abs(p[0]),Math.abs(p[1]));extent*=1.25;
 const scale=Math.min(w-70,h-105)/(extent*2),cx=w/2,cy=h/2+1;
 const xy=p=>[cx+p[0]*scale,cy-p[1]*scale];
 comparison.mapTransform=mode==='comparison'?{cx,cy,scale}:null;
 const line=(a,b,color,width=1)=>{ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
 const circle=(p,r,fill,stroke)=>{ctx.beginPath();ctx.arc(...p,r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}};
 const label=(txt,x,y,color='#8399ae',align='left')=>{ctx.fillStyle=color;ctx.font='10px Consolas,monospace';ctx.textAlign=align;ctx.fillText(txt,x,y);};
 const interval=extent>300?100:50;
 for(let m=-Math.floor(extent/interval)*interval;m<=extent;m+=interval){const [x,y]=xy([m,m]);line([x,45],[x,h-40],m===0?'#2c4055':'#192a3a');line([25,y],[w-25,y],m===0?'#2c4055':'#192a3a');if(m!==0&&y>65&&y<h-60)label(`${m}`,30,y-4);if(x>50&&x<w-50)label(`${m}`,x,h-42,undefined,'center');}
 const radius=(view?.objectiveRadius||20)*scale;circle([cx,cy],radius,'#6ce8c815','#52867e');circle([cx,cy],3,'#6ce8c8');label('OBJECTIVE',cx,cy+radius+16,'#a4c7bd','center');
 label(mode==='live'?'Y+':'N',w-30,60,'#b1c5d9','center');line([w-30,82],[w-30,66],'#b1c5d9');
 const bar=50*scale;line([w-30-bar,h-62],[w-30,h-62],'#708aa3',2);label('50 m',w-30,h-69,undefined,'right');
 if(!view||!frame)return;
 if($('sites').checked||mode==='comparison')for(const [i,site] of (view.sites||[]).entries())if(!(view.blockedSites||[]).includes(i)){
  const editing=mode==='comparison'&&$('comparison-baseline').value==='manual';
  const legal=!editing||!comparisonPlacementError($('comparison-sensor').value,i);
  circle(xy(site),editing?4:2,legal?'#566d87':'#3d3840');
  if(editing&&String(i)===$('comparison-site').value)circle(xy(site),8,null,'#f4cf78');
 }
 for(const [i,p] of view.placements.entries()){
  const c=view.catalogue.find(c=>c.id===p.sensor_id),[x,y]=xy(p.position),range=Math.max(...Object.values(c?.ranges||{r:0}));
  if($('ranges').checked&&c?.directional){const assumptions=c.simulation_assumptions||{},hardware=c.manufacturer_specifications||{};
   const radius=(assumptions.max_evaluation_distance_m||range)*scale,yaw=(p.yaw_deg||0)*Math.PI/180,half=(hardware.horizontal_fov_deg||24)*Math.PI/360;
   ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,radius,-yaw-half,-yaw+half);ctx.closePath();ctx.fillStyle='#70b7ff0c';ctx.fill();ctx.strokeStyle='#70b7ff70';ctx.setLineDash([5,5]);ctx.stroke();ctx.setLineDash([]);
  }else if($('ranges').checked){ctx.setLineDash([4,5]);circle([x,y],range*scale,'#70b7ff06','#70b7ff45');ctx.setLineDash([]);}
  ctx.fillStyle='#70b7ff';ctx.fillRect(x-4,y-4,8,8);label(`S${i+1}`,x+9,y-7,'#8fc9ff');
 }
 if($('trails').checked)for(const t of frame.threats){ctx.beginPath();let first=true;for(const f of view.frames.slice(0,frameIndex+1)){const p=f.threats.find(v=>v.id===t.id);if(!p)continue;const q=xy(p.position);if(first){ctx.moveTo(...q);first=false;}else ctx.lineTo(...q);}ctx.strokeStyle='#ff8b8045';ctx.lineWidth=1.2;ctx.stroke();}
 for(const d of frame.detections||[]){const p=view.placements[d.sensor_index],t=frame.threats.find(t=>t.id===d.target_id);if(p&&t){ctx.setLineDash([3,4]);line(xy(p.position),xy(t.position),'#f4cf7866');ctx.setLineDash([]);}}
 for(const t of frame.threats){const [x,y]=xy(t.position),id=Number(String(t.id).replace('drone-','')),diag=(frame.directionalDiagnostics||[]).find(d=>d.droneId===id);if($('drone-rings').checked&&(t.detected||t.tracked))circle([x,y],10,null,t.tracked?'#6ce8c8':'#f4cf78');ctx.save();ctx.translate(x,y);ctx.rotate(Math.PI/4);ctx.fillStyle=t.breached?'#ff5757':t.tracked?'#6ce8c8':'#ff8b80';ctx.fillRect(-3.5,-3.5,7,7);ctx.restore();if(w>470&&frame.threats.length<=8){label(t.id,x+12,y+4,'#c2ccda');if(diag)label(`${diag.insideFov?'IN':'OUT'} · ${diag.lineOfSight?'LOS':'BLOCKED'} · ${fmt(diag.pixelsOnTarget,2)} px · P ${fmt(diag.detectionProbability,3)}`,x+12,y+17,diag.blockedByGeometry?'#ff756b':'#9adbc9');}}
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
function comparisonName(){return $('comparison-baseline').value==='manual'?'Your manual layout':'Common-sense heuristic';}
function comparisonCost(placements=comparison.placements){return placements.reduce((sum,p)=>sum+(comparison.scenario?.catalogue.find(c=>c.id===p.sensor_id)?.cost||0),0);}
function comparisonControls(){
 const locked=comparison.loading||busy, available=Boolean(comparison.scenario);
 for(const id of ['profile','policy','case','comparison-baseline'])$(id).disabled=locked;
 for(const id of ['comparison-sensor','comparison-site','comparison-suggest','comparison-clear'])$(id).disabled=locked||!available;
 $('comparison-evaluate').disabled=locked||!available;
 $('comparison-evaluate').textContent=comparison.loading?'Evaluating / loading…':'Evaluate both layouts';
 $('comparison-add').disabled=locked||!available||Boolean(comparisonPlacementError($('comparison-sensor').value,Number($('comparison-site').value)))||!$('comparison-site').value;
 for(const button of $('comparison-placements').querySelectorAll('button'))button.disabled=locked;
 $('comparison-show-rl').disabled=locked||!comparison.result;$('comparison-show-baseline').disabled=locked||!comparison.result;
}
function comparisonPlacementError(sensorId,siteIndex){
 const s=comparison.scenario;if(!s)return 'Load a scenario first.';
 const c=s.catalogue.find(c=>c.id===sensorId),site=s.sites[siteIndex];
 if(!c||!site)return 'Choose a sensor and an approved site.';
 if(s.availableSensorIds&&!s.availableSensorIds.includes(sensorId))return 'This sensor is unavailable for the selected case.';
 if((s.blockedSites||[]).includes(siteIndex)||(s.eligibleSites&&!s.eligibleSites.includes(siteIndex)))return 'This site is outside the permitted deployment area.';
 if(comparison.placements.some(p=>p.site_index===siteIndex))return 'This site already has a sensor. Remove it before adding another.';
 if(comparison.placements.length>=s.maxSensors)return `The limit is ${s.maxSensors} sensors. Remove a sensor first.`;
 if(comparisonCost()+c.cost>s.budget+1e-8)return 'This sensor exceeds the remaining budget. Remove a sensor or choose a lower-cost sensor.';
 if(comparison.placements.some(p=>Math.hypot(site[0]-s.sites[p.site_index][0],site[1]-s.sites[p.site_index][1])<(s.minSeparation||0)-1e-8))return `Sensors need at least ${fmt(s.minSeparation,0)} m of separation. Choose a different site.`;
 return '';
}
function comparisonFillEditor(){
 const s=comparison.scenario;$('comparison-placements').replaceChildren();
 $('comparison-editor').hidden=$('comparison-baseline').value!=='manual';
 $('comparison-baseline-heading').textContent=comparisonName();
 $('comparison-show-baseline').textContent=$('comparison-baseline').value==='manual'?'My layout':'Common sense';
 if(!s)return;
 const selectedSensor=s.catalogue.find(c=>c.id===$('comparison-sensor').value);
 $('comparison-sensor-info').textContent=selectedSensor?Object.entries(selectedSensor.ranges||{}).filter(([,range])=>range>0).map(([modality,range])=>`${modality.toUpperCase()} range ${fmt(range,0)} m`).join(' · '):'';
 for(const [index,p] of comparison.placements.entries()){
  const c=s.catalogue.find(c=>c.id===p.sensor_id),row=element('div','','manual-placement');
  row.append(element('span',`${c?.label||c?.name||p.sensor_id} · site ${p.site_index+1}`));
  const remove=element('button','×','icon-button');remove.setAttribute('aria-label',`Remove ${c?.label||c?.name||p.sensor_id} from site ${p.site_index+1}`);
  remove.onclick=()=>{comparison.placements.splice(index,1);comparisonChanged();};row.append(remove);$('comparison-placements').append(row);
 }
 if(!comparison.placements.length)$('comparison-placements').append(element('p','No sensors placed. An empty layout can be evaluated.','helper'));
 $('comparison-budget').textContent=`${comparison.placements.length} / ${s.maxSensors} sensors · ${fmt(comparisonCost())} / ${fmt(s.budget)} budget units · ${fmt(s.minSeparation||0,0)} m minimum spacing`;
 for(const option of $('comparison-site').options){const error=comparisonPlacementError($('comparison-sensor').value,Number(option.value));option.disabled=Boolean(error);}
 if(!$('comparison-site').selectedOptions[0]||$('comparison-site').selectedOptions[0].disabled){const option=Array.from($('comparison-site').options).find(o=>!o.disabled);$('comparison-site').value=option?.value??'';}
 $('comparison-site-help').textContent=$('comparison-site').value?'Choose a site above or click a site on the map. Each site holds one sensor; budget and spacing limits apply.':'No legal site remains for this sensor. Remove a sensor or choose a lower-cost sensor to make room.';
 comparisonControls();
}
function comparisonDraft(){
 const s=comparison.scenario;if(!s)return;
 const placements=comparison.placements.map(p=>({...p,position:s.sites[p.site_index],cost:s.catalogue.find(c=>c.id===p.sensor_id)?.cost||0}));
 setView({...s,label:s.label||'Placement comparison',coordinateLabel:'Objective-relative metres · approved sites',placements,frames:[{time:0,threats:[],detections:[],tracks:s.tracks||[]}],metrics:null,outcome:'layout_preview'});
 $('map').setAttribute('aria-label',`Layout editor showing the objective, approved sensor sites, your current sensors, and ${(s.tracks||[]).length} initial public reports marked in amber. Adversary paths are hidden until evaluation.`);
 $('comparison-timeline-note').textContent='Layout preview · paths hidden';
 $('comparison-tag').textContent='LAYOUT PREVIEW';
}
function comparisonClearResults(){
 for(const method of ['rl','baseline'])for(const metric of ['detected','confirmed','timely','cost'])$(`comparison-${method}-${metric}`).textContent='—';
 $('comparison-verdict').textContent='Evaluate both layouts to see their results.';
 $('comparison-fairness').textContent='Build a layout before revealing adversary paths. No Unreal connection needed.';
 $('comparison-tag').textContent='LAYOUT PREVIEW';$('comparison-timeline-note').textContent='Layout preview · paths hidden';
 $('comparison-show-rl').setAttribute('aria-pressed','false');$('comparison-show-baseline').setAttribute('aria-pressed','true');
}
function comparisonChanged(){
 pause();showError('');comparison.result=null;comparison.method='baseline';comparisonClearResults();comparisonFillEditor();comparisonDraft();
 $('comparison-status').textContent=comparison.revealed?'Layout changed. Previous scores cleared. This is practice on a scenario whose paths you have already seen.':'Layout ready. Evaluate both layouts to reveal adversary paths and matched sensing results.';
}
async function loadComparisonScenario(){
 pause();const sequence=++comparison.sequence;
 const row=catalog.find(r=>r.profile===$('profile').value&&String(r.policy)===$('policy').value&&String(r.case)===$('case').value);
 comparison.scenario=null;comparison.result=null;comparison.placements=[];comparison.revealed=false;comparison.method='baseline';comparison.replayId=row?.id;
 comparison.loading=true;view=null;eventSignature='';nativeImage=null;document.body.classList.remove('native-preview');$('empty').hidden=true;
 $('comparison-sensor').replaceChildren();$('comparison-site').replaceChildren();$('comparison-placements').replaceChildren();$('comparison-budget').textContent='';
 $('comparison-forecast').textContent='Loading scenario information…';$('comparison-weather').textContent='';$('comparison-reports').textContent='';
 $('comparison-reasoning').textContent='Loading public scenario information…';$('comparison-status').textContent='Loading approved sites and the common-sense suggestion…';$('episode-title').textContent='Preparing comparison';
 comparisonClearResults();sensorDetails();results();render();controls();showError('');
 try{
  if(!row)throw new Error('This recorded case is unavailable. Choose another case.');
  const s=await api('/api/comparison/scenario',{replayId:row.id});
  if(sequence!==comparison.sequence||mode!=='comparison')return;
  comparison.scenario=s;comparison.placements=(s.suggestedPlacements||[]).map(p=>({sensor_id:p.sensor_id,site_index:p.site_index}));
  for(const c of s.catalogue){if(s.availableSensorIds&&!s.availableSensorIds.includes(c.id))continue;const option=element('option',`${c.label||c.name||c.id} · ${fmt(c.cost)} units`);option.value=c.id;$('comparison-sensor').append(option);}
  for(const [i,site] of s.sites.entries()){if((s.blockedSites||[]).includes(i)||(s.eligibleSites&&!s.eligibleSites.includes(i)))continue;const option=element('option',`Site ${i+1} · ${fmt(site[0],0)}, ${fmt(site[1],0)} m`);option.value=i;$('comparison-site').append(option);}
  $('comparison-reasoning').textContent=s.selection?.explanation||'The suggestion spreads affordable sensors over likely approaches using the public forecast and sensor coverage. It respects the same available sensors, approved sites, spacing and budget as RL, without seeing the realised adversary paths.';
  const directions=['E','NE','N','NW','W','SW','S','SE'],forecast=s.forecast||{},weather=s.weather||{};
  $('comparison-forecast').textContent=`Approach likelihood: ${(forecast.approach_weights||[]).map((weight,i)=>`${directions[i]} ${pct(weight)}`).join(' · ')}. Expected altitude ${fmt(forecast.altitude,0)} m; speed ${fmt(forecast.speed,0)} m/s; emitter likelihood ${pct(forecast.emitter_probability)}.`;
  $('comparison-weather').textContent=`Visibility ${pct(weather.visibility)} · light ${pct(weather.illumination)} · rain ${pct(weather.rain)} · RF noise ${pct(weather.rf_noise)}. Forecasts are public estimates; ranges and sensing are synthetic.`;
  const reports=s.tracks||[];
  $('comparison-reports').textContent=`${reports.length} initial public report${reports.length===1?'':'s'}${reports.length?' (amber rings on the layout preview): '+reports.map(t=>`${t.id} at ${fmt(t.position[0],0)}, ${fmt(t.position[1],0)} m; ${pct(t.confidence)} confidence`).join(' · '):''}. These are reported observations, not future trajectories.`;
  comparisonChanged();
 }catch(e){if(sequence===comparison.sequence){showError(`Could not prepare comparison: ${e.message}`);$('comparison-status').textContent='Choose another case or select Compare placements again to retry.';}}
 finally{if(sequence===comparison.sequence){comparison.loading=false;controls();}}
}
function comparisonAdd(siteIndex){
 if(comparison.loading||$('comparison-baseline').value!=='manual')return;
 const sensorId=$('comparison-sensor').value,error=comparisonPlacementError(sensorId,siteIndex);if(error)return showError(error);
 comparison.placements.push({sensor_id:sensorId,site_index:siteIndex});comparisonChanged();
}
function comparisonSummary(data){
 const metrics=data.metrics||{},total=metrics.target_results?.length||data.frames?.[0]?.threats?.length||0;
 const count=(fraction)=>Number.isFinite(fraction)?`${Math.round(fraction*total)} / ${total} (${pct(fraction)})`:'Unavailable';
 return {detected:count(metrics.detected_fraction),confirmed:count(metrics.confirmed_fraction),timely:count(metrics.timely_fraction??(total?metrics.target_results?.filter(t=>t.timely_confirmed).length/total:undefined)),cost:`${fmt(data.placements.reduce((sum,p)=>sum+(p.cost??data.catalogue.find(c=>c.id===p.sensor_id)?.cost??0),0))} / ${fmt(data.budget)}`,total};
}
function comparisonShow(method){
 if(!comparison.result)return;
 const previousTime=view?.frames[frameIndex]?.time||0;
 comparison.method=method;view=comparison.result[method];nativeImage=null;eventSignature='';
 frameIndex=Math.max(0,view.frames.findLastIndex(f=>f.time<=previousTime));
 $('comparison-show-rl').setAttribute('aria-pressed',String(method==='rl'));$('comparison-show-baseline').setAttribute('aria-pressed',String(method==='baseline'));
 $('episode-title').textContent=`${comparison.scenario.label} · ${method==='rl'?'RL layout':comparisonName()}`;
 $('map').setAttribute('aria-label',`${method==='rl'?'Reinforcement learning':comparisonName()} sensor layout and adversary sensing, at the shared replay time.`);
 $('comparison-timeline-note').textContent='Shared time · switch layouts to compare';
 sensorDetails();results();render();controls();
}
async function runComparison(){
 if(comparison.loading||!comparison.scenario)return;
 pause();comparison.loading=true;controls();showError('');$('comparison-status').textContent='Evaluating both layouts against the same adversaries and sensing draws…';
 const sequence=comparison.sequence,baseline=$('comparison-baseline').value;
 try{
  const body={replayId:comparison.replayId,baseline};if(baseline==='manual')body.placements=comparison.placements.map(p=>({sensor_id:p.sensor_id,site_index:p.site_index}));
  const result=await api('/api/comparison/run',body);if(sequence!==comparison.sequence||mode!=='comparison')return;
  comparison.result=result;comparison.revealed=true;
  for(const method of ['rl','baseline']){const summary=comparisonSummary(result[method]);for(const metric of ['detected','confirmed','timely','cost'])$(`comparison-${method}-${metric}`).textContent=summary[metric];}
  const total=comparisonSummary(result.rl).total,rlDetected=Math.round(result.rl.metrics.detected_fraction*total),baselineDetected=Math.round(result.baseline.metrics.detected_fraction*total),difference=rlDetected-baselineDetected;
  $('comparison-verdict').textContent=`${difference===0?'Both layouts detected the same number of adversaries':`RL detected ${Math.abs(difference)} ${difference>0?'more':'fewer'} adversar${Math.abs(difference)===1?'y':'ies'}`} in this case. This single synthetic case does not establish overall performance.`;
  $('comparison-tag').textContent='MATCHED FINAL RESULTS';
  $('comparison-fairness').textContent='Both layouts rescored with the same adversaries, sensing draws, sensor catalogue and limits. Timely means confirmed before the objective-zone deadline.';
  $('comparison-status').textContent='Evaluation complete. Switch map layouts at any replay time. Editing clears these scores; repeat edits are practice on this revealed scenario.';
  frameIndex=0;elapsed=0;view=result.baseline;comparisonShow('baseline');
 }catch(e){showError(`Comparison failed: ${e.message}`);$('comparison-status').textContent='Check the layout and scenario limits, then evaluate again.';}
 finally{comparison.loading=false;controls();}
}
for(const id of ['profile','policy','case'])$(id).addEventListener('change',loadReplay);
 $('comparison-baseline').onchange=()=>{if(comparison.scenario){comparison.placements=(comparison.scenario.suggestedPlacements||[]).map(p=>({sensor_id:p.sensor_id,site_index:p.site_index}));comparisonChanged();}};
 $('comparison-sensor').onchange=()=>{comparisonFillEditor();render();};$('comparison-site').onchange=()=>{comparisonControls();render();};
 $('comparison-add').onclick=()=>comparisonAdd(Number($('comparison-site').value));
 $('comparison-suggest').onclick=()=>{comparison.placements=(comparison.scenario?.suggestedPlacements||[]).map(p=>({sensor_id:p.sensor_id,site_index:p.site_index}));comparisonChanged();};
 $('comparison-clear').onclick=()=>{comparison.placements=[];comparisonChanged();};
 $('comparison-evaluate').onclick=runComparison;$('comparison-show-rl').onclick=()=>comparisonShow('rl');$('comparison-show-baseline').onclick=()=>comparisonShow('baseline');
 $('map').addEventListener('click',event=>{
  if(mode!=='comparison'||$('comparison-baseline').value!=='manual'||comparison.loading||!comparison.scenario||!comparison.mapTransform)return;
  const {cx,cy,scale}=comparison.mapTransform,bounds=$('map').getBoundingClientRect(),x=event.clientX-bounds.left,y=event.clientY-bounds.top;
  let nearest=-1,distance=16;for(const [i,site] of comparison.scenario.sites.entries()){const d=Math.hypot(cx+site[0]*scale-x,cy-site[1]*scale-y);if(d<distance){nearest=i;distance=d;}}
  if(nearest>=0)comparisonAdd(nearest);
 });
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
new ResizeObserver(()=>draw(view?.frames[frameIndex])).observe($('map').parentElement);
api('/api/session').then(async session=>{token=session.token;catalog=session.replays;connected=session.status.connected;if(session.savedModels?.length){const group=document.createElement('optgroup');group.label='Archived native layouts (no inference)';for(const model of session.savedModels){const option=document.createElement('option');option.value=model.id;option.textContent=model.label;group.append(option);}$('live-policy').append(group);}await loadReplay();connectionStatus();}).catch(e=>showError(e.message));
controls();requestAnimationFrame(animate);
