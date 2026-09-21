'use strict';
const $ = id => document.getElementById(id);
let token='', catalog=[], view=null, frameIndex=0, mode='recorded', playing=false, busy=false, busyOperation='', connected=false;
let elapsed=0, lastWall=0, liveWall=0, loadSequence=0;
let nativeImage=null, eventSignature='';
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
 const label=mode==='recorded'?'Replay ready':connected?'Unreal connected':'Unreal disconnected';
 $('connection-label').textContent=label;
 $('connection').firstChild.style.background=mode==='live'&&!connected?'#ffb18a':'#6ce8c8';
 $('connect').textContent=connected?'Disconnect':'Connect to Unreal';
}
function controls(){
 const blocking=busy&&busyOperation!=='step', liveEnded=Boolean(mode==='live'&&view?.ended);
 $('play').textContent=playing?'Pause':mode==='live'?'Run episode':'Play replay';
 $('play').disabled=!view||blocking||liveEnded;
 $('step').disabled=!view||playing||busy||liveEnded;
 $('restart').disabled=!view||mode==='live'||blocking;
 $('timeline').disabled=!view;$('speed').disabled=!view;
 $('connect').disabled=busy||playing;$('plan').disabled=!connected||busy||playing;
 $('recorded-mode').disabled=busy||playing;$('live-mode').disabled=busy||playing;
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
  const text=element('div','');text.append(element('strong',c.label||p.sensor_id),element('small',`${fmt(p.position[0],0)}, ${fmt(p.position[1],0)} m · ${fmt(c.cost)} units`));row.append(symbol,text);$('sensors').append(row);
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
 $('results-tag').textContent=mode==='recorded'?'RECORDED FINAL':m?'MEASURED FINAL':'AWAITING EPISODE END';
 $('outcome-label').textContent=mode==='recorded'?'Recorded final result':'Episode status';
 $('detected-label').textContent=mode==='recorded'?'Detected so far':'Public tracks';
 $('result-note').textContent=mode==='recorded'?'Synthetic sensing outcomes. Timely confirmation means detection before the deadline; no interception is simulated. Warning is unavailable when the replay does not contain the exact metric.':'Live metrics appear when the native episode ends. Synthetic analytical sensors; terrain occlusion is not modeled. Warning values are measured lower bounds from detection to 20 m objective-zone arrival.';
}
function setView(data){view=data;nativeImage=null;eventSignature='';if(data.nativePreview){const image=new Image();image.onload=()=>{if(view===data){nativeImage=image;render();}};image.src=data.nativePreview.image;}$('map').setAttribute('aria-label',data.nativePreview?'Native Unreal sensor placement preview':'Top-down simulation view');document.body.classList.toggle('native-preview',Boolean(data.nativePreview));frameIndex=0;elapsed=0;$('empty').hidden=true;syncTimeline();$('episode-title').textContent=view.label;$('coordinates').textContent=view.coordinateLabel;sensorDetails();results();render();controls();}
async function loadReplay(){
 pause();const seq=++loadSequence;const row=catalog.find(r=>r.profile===$('profile').value&&String(r.policy)===$('policy').value&&String(r.case)===$('case').value);
 if(!row)return showError('This recorded case is unavailable.');
 try{const data=await api(`/api/replay/${row.id}`);if(seq!==loadSequence||mode!=='recorded')return;setView(data);showError('');}catch(e){showError(e.message);}
}
async function switchMode(next){
 pause();mode=next;++loadSequence;showError('');
 for(const id of ['recorded','live'])$(id+'-mode').setAttribute('aria-pressed',String(mode===id));
 $('recorded-controls').hidden=mode!=='recorded';$('live-controls').hidden=mode!=='live';$('source-badge').textContent=mode==='recorded'?'RECORDED · SYNTHETIC':'LIVE UNREAL · SYNTHETIC';
 $('provenance').textContent=mode==='recorded'?'Published temporal-v6 evidence · 18 recorded cases':'Local Unreal bridge · scripted Red · experimental Blue';
 if(mode==='recorded')await loadReplay();else{view=null;$('empty').hidden=false;$('episode-title').textContent='Awaiting live episode';$('clock').textContent='00:00.0';$('progress').textContent='Fixed-step simulation';sensorDetails();results();render();controls();}
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
 draw(frame);events(frame);
}
function events(frame){
 const log=[];
 for(const [i,p] of (view?.placements||[]).entries()){const c=view.catalogue.find(c=>c.id===p.sensor_id);log.push([0,`Blue deployed ${c?.label||p.sensor_id} at sensor ${i+1}`]);}
 if(frame&&mode==='recorded'){
  for(const t of view.metrics.target_results||[]){
   if(t.first_detection!==null&&t.first_detection<=frame.time)log.push([t.first_detection,`${t.id} detected`]);
   if(t.first_confirmation!==null&&t.first_confirmation<=frame.time)log.push([t.first_confirmation,`${t.id} track confirmed`]);
   if(t.time_to_zone<=frame.time)log.push([t.time_to_zone,`${t.id} reached zone · ${t.timely_confirmed?'confirmed before deadline':'deadline missed'}`]);
  }
 }else if(frame)for(const t of frame.tracks||[])log.push([t.timestamp,`${t.id} · ${t.confirmed?'confirmed track':'detection report'}`]);
 log.sort((a,b)=>b[0]-a[0]);$('event-count').textContent=log.length;
 const visible=log.slice(0,40),signature=JSON.stringify(visible);if(signature===eventSignature)return;eventSignature=signature;$('events').replaceChildren();
 for(const [t,text] of visible){const row=element('div','','event');row.append(element('time',`${fmt(t)} s`),element('span',text));$('events').append(row);}
 if(!log.length)$('events').append(element('p','Detection reports will appear here.','helper'));
}
function draw(frame){
 const canvas=$('map'),bounds=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 const pixelWidth=Math.max(1,Math.round(bounds.width*dpr)),pixelHeight=Math.max(1,Math.round(bounds.height*dpr));
 if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
 const w=bounds.width,h=bounds.height;ctx.fillStyle='#0b141e';ctx.fillRect(0,0,w,h);
 if(view?.nativePreview){if(nativeImage){const s=Math.min(w/1920,h/1080),ox=(w-1920*s)/2,oy=(h-1080*s)/2;ctx.drawImage(nativeImage,ox,oy,1920*s,1080*s);for(const a of view.nativePreview.anchors){const x=ox+a.x*s,y=oy+a.y*s;ctx.strokeStyle='#9fffe3';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(x,y+2,12,6,0,0,Math.PI*2);ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#ccfff0';ctx.fillText('Sensor',x+16,y-10);}}return;}
 const pts=view?view.frames.flatMap(f=>f.threats.map(t=>t.position)).concat(view.placements.map(p=>p.position)):[];
 let extent=120;for(const p of pts)extent=Math.max(extent,Math.abs(p[0]),Math.abs(p[1]));extent*=1.25;
 const scale=Math.min(w-70,h-105)/(extent*2),cx=w/2,cy=h/2+1;
 const xy=p=>[cx+p[0]*scale,cy-p[1]*scale];
 const line=(a,b,color,width=1)=>{ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
 const circle=(p,r,fill,stroke)=>{ctx.beginPath();ctx.arc(...p,r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}};
 const label=(txt,x,y,color='#8399ae',align='left')=>{ctx.fillStyle=color;ctx.font='10px Consolas,monospace';ctx.textAlign=align;ctx.fillText(txt,x,y);};
 const interval=extent>300?100:50;
 for(let m=-Math.floor(extent/interval)*interval;m<=extent;m+=interval){const [x,y]=xy([m,m]);line([x,45],[x,h-40],m===0?'#2c4055':'#192a3a');line([25,y],[w-25,y],m===0?'#2c4055':'#192a3a');if(m!==0&&y>65&&y<h-60)label(`${m}`,30,y-4);if(x>50&&x<w-50)label(`${m}`,x,h-42,undefined,'center');}
 const radius=(view?.objectiveRadius||20)*scale;circle([cx,cy],radius,'#6ce8c815','#52867e');circle([cx,cy],3,'#6ce8c8');label('OBJECTIVE',cx,cy+radius+16,'#a4c7bd','center');
 label(mode==='live'?'Y+':'N',w-30,60,'#b1c5d9','center');line([w-30,82],[w-30,66],'#b1c5d9');
 const bar=50*scale;line([w-30-bar,h-62],[w-30,h-62],'#708aa3',2);label('50 m',w-30,h-69,undefined,'right');
 if(!view||!frame)return;
 if($('sites').checked)for(const [i,site] of (view.sites||[]).entries())if(!(view.blockedSites||[]).includes(i))circle(xy(site),2,'#566d87');
 for(const [i,p] of view.placements.entries()){
  const c=view.catalogue.find(c=>c.id===p.sensor_id),[x,y]=xy(p.position),range=Math.max(...Object.values(c?.ranges||{r:0}));
  if($('ranges').checked){ctx.setLineDash([4,5]);circle([x,y],range*scale,'#70b7ff06','#70b7ff45');ctx.setLineDash([]);}
  ctx.fillStyle='#70b7ff';ctx.fillRect(x-4,y-4,8,8);label(`S${i+1}`,x+9,y-7,'#8fc9ff');
 }
 if($('trails').checked)for(const t of frame.threats){ctx.beginPath();let first=true;for(const f of view.frames.slice(0,frameIndex+1)){const p=f.threats.find(v=>v.id===t.id);if(!p)continue;const q=xy(p.position);if(first){ctx.moveTo(...q);first=false;}else ctx.lineTo(...q);}ctx.strokeStyle='#ff8b8045';ctx.lineWidth=1.2;ctx.stroke();}
 for(const d of frame.detections||[]){const p=view.placements[d.sensor_index],t=frame.threats.find(t=>t.id===d.target_id);if(p&&t){ctx.setLineDash([3,4]);line(xy(p.position),xy(t.position),'#f4cf7866');ctx.setLineDash([]);}}
 for(const t of frame.threats){const [x,y]=xy(t.position);if($('drone-rings').checked&&(t.detected||t.tracked))circle([x,y],10,null,t.tracked?'#6ce8c8':'#f4cf78');ctx.save();ctx.translate(x,y);ctx.rotate(Math.PI/4);ctx.fillStyle=t.breached?'#ff5757':t.tracked?'#6ce8c8':'#ff8b80';ctx.fillRect(-3.5,-3.5,7,7);ctx.restore();if(w>470&&frame.threats.length<=8)label(t.id,x+12,y+4,'#c2ccda');}
 for(const t of frame.tracks||[]){const p=xy(t.position);if($('drone-rings').checked)circle(p,9,null,t.confirmed?'#6ce8c8':'#f4cf78');if(w>470)label(t.id,p[0]+12,p[1]-9,'#f4cf78');}
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
  if(mode==='recorded'){
   if(lastWall)elapsed+=(now-lastWall)/1000*Number($('speed').value);lastWall=now;
   while(frameIndex<view.frames.length-1&&view.frames[frameIndex+1].time<=elapsed){frameIndex++;render();}
   if(frameIndex>=view.frames.length-1)pause();
  }else{const speed=Math.max(.1,Number($('speed').value)||1),interval=1000*(Number(view.stepDurationSeconds)||.5)/speed;if(!busy&&now-liveWall>=interval){liveWall=now;liveAction('step');}}
 }
 requestAnimationFrame(animate);
}
for(const id of ['profile','policy','case'])$(id).addEventListener('change',loadReplay);
for(const id of ['ranges','trails','drone-rings','sites'])$(id).addEventListener('change',()=>render());
$('recorded-mode').onclick=()=>switchMode('recorded');$('live-mode').onclick=()=>switchMode('live');
$('connect').onclick=()=>{pause();liveAction(connected?'disconnect':'connect');};
$('plan').onclick=()=>{pause();liveAction($('live-policy').value.startsWith('saved-')?'preview':'reset',{seed:Number($('seed').value),policy:$('live-policy').value});};
$('live-policy').onchange=()=>{$('plan').textContent=$('live-policy').value.startsWith('saved-')?'Apply saved layout':'Plan new episode';};
$('presentation').onchange=()=>{document.body.classList.toggle('presentation',$('presentation').checked);$('live-policy').size=$('presentation').checked?9:1;render();};
$('play').onclick=()=>{if(playing)return pause();if(!view)return;if(mode==='recorded'&&frameIndex===view.frames.length-1){frameIndex=0;render();}else if(mode==='live'&&frameIndex<view.frames.length-1){frameIndex=view.frames.length-1;render();}elapsed=view.frames[frameIndex].time;lastWall=0;liveWall=0;playing=true;controls();render();};
$('step').onclick=()=>{pause();if(mode==='live')liveAction('step');else if(view){frameIndex=Math.min(frameIndex+1,view.frames.length-1);render();}};
$('restart').onclick=()=>{pause();frameIndex=0;elapsed=0;render();};
$('timeline').oninput=()=>{pause();const requested=Number($('timeline').value);if(mode==='live'){frameIndex=0;for(let i=1;i<view.frames.length&&view.frames[i].completedSteps<=requested;i++)frameIndex=i;}else frameIndex=requested;render();};
$('speed').onchange=()=>render();
new ResizeObserver(()=>draw(view?.frames[frameIndex])).observe($('map').parentElement);
api('/api/session').then(async session=>{token=session.token;catalog=session.replays;connected=session.status.connected;if(session.savedModels?.length){const group=document.createElement('optgroup');group.label='Archived native layouts (no inference)';for(const model of session.savedModels){const option=document.createElement('option');option.value=model.id;option.textContent=model.label;group.append(option);}$('live-policy').append(group);}await loadReplay();connectionStatus();}).catch(e=>showError(e.message));
controls();requestAnimationFrame(animate);
