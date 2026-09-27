'use strict';
const $ = id => document.getElementById(id);
const colors = {red:'#df5245',orange:'#e9952b',yellow:'#d6b83c',green:'#65a662',cyan:'#51b5bf',blue:'#4c88d1',purple:'#9b65b6',pink:'#da80aa'};
const names = {red:'אדום',orange:'כתום',yellow:'צהוב',green:'ירוק',cyan:'תכלת',blue:'כחול',purple:'סגול',pink:'ורוד'};
const statuses = {ready:'מוכן',queued:'בתור',running:'מעבד',complete:'הושלם',failed:'נכשל',cancelled:'בוטל'};
const stages = {queued:'ממתין לעיבוד',pose:'מעקב גוף וצבעי אחיזות',metrics:'חישוב מדדי תנועה',render:'יצירת הווידאו',preview:'הכנת תצוגה לדפדפן',complete:'הניתוח הושלם'};
let selected=null, report=null, crop=null, cropMode=false, pollTimer=null, chartKey='speed', videoKind='source', uploadBusy=false;
let chartLayers={motion:true,contacts:true};
let cropImage=new Image(), cropBox=null, dragStart=null;
const fmt=(n,d=2)=>n==null?'—':Number(n).toFixed(d);
const pct=n=>n==null?'—':`${Math.round(n*100)}%`;
const time=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`;
const url=(kind,id=selected?.id)=>`/api/jobs/${id}/files/${kind}`;
function showError(message){$('error').textContent=message;$('error').hidden=!message;}
async function api(path, options={}){
  const response=await fetch(path,options);
  if(!response.ok){let body;try{body=await response.json();}catch{body={detail:response.statusText};}throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));}
  return response.json();
}
function safeAction(fn){return (...args)=>Promise.resolve().then(()=>fn(...args)).catch(e=>showError(e.message));}
function updateControls(){
  const ready=selected?.status==='ready'&&!uploadBusy;
  $('analyze').disabled=!ready;$('crop-toggle').disabled=!ready;
  for(const id of ['smoothing','threshold','visibility','roi','contact-radius','skin'])$(id).disabled=!ready;
  const active=selected&&['running','queued'].includes(selected.status);
  $('cancel').hidden=!active;$('progress-section').hidden=!selected||selected.status==='ready';
  $('stage').textContent=stages[selected?.stage]||statuses[selected?.status]||'';
  $('percent').textContent=pct(selected?.progress||0);$('progress').value=selected?.progress||0;
  $('analyze').textContent=active?'הניתוח בעבודה…':selected?.status==='complete'?'הניתוח הושלם':'התחל ניתוח ↖';
}
async function refreshHistory(){
  const list=await api('/api/jobs');$('history').replaceChildren();
  if(!list.length){const p=document.createElement('p');p.className='hint';p.textContent='הניתוחים שלך יופיעו כאן.';$('history').append(p);return;}
  for(const job of list.slice(0,8)){
    const b=document.createElement('button');b.className='history-item'+(selected?.id===job.id?' active':'');
    const n=document.createElement('span');n.textContent=job.name;const s=document.createElement('small');s.textContent=statuses[job.status];b.append(n,s);
    b.onclick=safeAction(()=>selectJob(job));$('history').append(b);
  }
}
function setVideo(kind,restoreTime=false){
  if(!selected)return;const current=$('video').currentTime||0;videoKind=kind;
  $('video').src=url(kind);$('video').poster=url(selected.status==='complete'?'poster':'thumbnail');
  if(restoreTime)$('video').addEventListener('loadedmetadata',()=>{$('video').currentTime=Math.min(current,$('video').duration||current);},{once:true});
  document.querySelectorAll('[data-video]').forEach(b=>b.classList.toggle('active',b.dataset.video===kind));
}
async function selectJob(job){
  clearTimeout(pollTimer);setCropMode(false);selected=job;report=null;crop=job.options?.crop||null;
  showError('');$('empty').hidden=true;$('video').hidden=false;$('clip-name').textContent=job.name;
  $('results').hidden=true;$('pre-results').hidden=false;$('video-tabs').hidden=job.status!=='complete';
  $('video-meta').textContent=`${job.metadata.width} × ${job.metadata.height} · ${fmt(job.metadata.fps,1)} FPS · ${fmt(job.metadata.duration_s,1)} שנ׳`;
  cropImage=new Image();cropImage.src=url('thumbnail');updateCropLabel();
  if(job.options){for(const [id,key] of [['smoothing','smoothing_seconds'],['threshold','static_speed'],['visibility','visibility_threshold'],['roi','roi_size'],['contact-radius','contact_radius_px']]){$(id).value=job.options[key];$(id).dispatchEvent(new Event('input'));}$('skin').checked=job.options.mask_skin;}
  setVideo(job.status==='complete'?'preview':'source');updateControls();
  await refreshHistory();
  if(job.status==='complete')await loadReport(job.id);
  else if(['running','queued'].includes(job.status))poll(job.id);
  else if(job.status==='failed')showError(job.error||'הניתוח נכשל. נסה להעלות סרטון חדש.');
}
async function uploadFile(file){
  if(!file||uploadBusy)return;
  if(file.size>500*1024*1024)throw new Error('יש לבחור סרטון קטן מ־500MB.');
  uploadBusy=true;updateControls();showError('');$('stage').textContent='מעלה סרטון…';$('progress-section').hidden=false;
  try{const data=new FormData();data.append('file',file);const job=await api('/api/upload',{method:'POST',body:data});await selectJob(job);}
  finally{uploadBusy=false;updateControls();$('file').value='';}
}
$('file').onchange=safeAction(e=>uploadFile(e.target.files[0]));
$('empty-upload').onclick=()=>$('file').click();
$('dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click();}};
for(const event of ['dragenter','dragover'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('dragover');});
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('dragover');});
$('dropzone').addEventListener('drop',safeAction(e=>uploadFile(e.dataTransfer.files[0])));
$('samples').onchange=safeAction(async e=>{if(!e.target.value)return;const job=await api(`/api/samples/${encodeURIComponent(e.target.value)}`,{method:'POST'});await selectJob(job);e.target.value='';});
for(const [id,format] of [['smoothing',n=>`${Number(n).toFixed(2)} שנ׳`],['threshold',n=>`${Number(n).toFixed(3)} diag/s`],['visibility',n=>Number(n).toFixed(2)],['roi',n=>`${n} px`],['contact-radius',n=>`${n} px`]]){$(id).oninput=()=>$(`${id}-value`).textContent=format($(id).value);}
function closeSettingHelp(except=null){for(const tip of document.querySelectorAll('.help-tip')){if(tip===except)continue;tip.classList.remove('active');tip.setAttribute('aria-expanded','false');$(tip.dataset.help).hidden=true;}}
for(const tip of document.querySelectorAll('.help-tip')){
  const toggle=event=>{event.preventDefault();event.stopPropagation();const panel=$(tip.dataset.help),willOpen=panel.hidden;closeSettingHelp(tip);panel.hidden=!willOpen;tip.classList.toggle('active',willOpen);tip.setAttribute('aria-expanded',String(willOpen));};
  tip.onclick=toggle;tip.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){toggle(event);}};
}
$('analysis-settings').ontoggle=()=>{if(!$('analysis-settings').open)closeSettingHelp();};
$('analyze').onclick=safeAction(async()=>{
  if(!selected)return;setCropMode(false);$('analysis-settings').open=false;
  const body={crop,static_speed:+$('threshold').value,smoothing_seconds:+$('smoothing').value,visibility_threshold:+$('visibility').value,roi_size:+$('roi').value,contact_radius_px:+$('contact-radius').value,mask_skin:$('skin').checked};
  $('analyze').disabled=true;
  try{selected=await api(`/api/jobs/${selected.id}/analyze`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});showError('');poll(selected.id);}
  finally{updateControls();}
});
$('cancel').onclick=safeAction(async()=>{if(selected){await api(`/api/jobs/${selected.id}/cancel`,{method:'POST'});$('stage').textContent='מבטל…';}});
async function poll(id){
  if(selected?.id!==id)return;
  try{const job=await api(`/api/jobs/${id}`);if(selected?.id!==id)return;selected=job;updateControls();
    if(['running','queued'].includes(job.status)){pollTimer=setTimeout(()=>poll(id),900);return;}
    await refreshHistory();
    if(job.status==='complete'){$('video-tabs').hidden=false;setVideo('preview');await loadReport(id);}
    else if(job.status==='failed')showError(job.error||'הניתוח נכשל.');
  }catch(error){showError(`החיבור לשרת נותק: ${error.message}`);pollTimer=setTimeout(()=>poll(id),2500);}
}
async function loadReport(id){
  const data=await api(url('report',id));if(selected?.id!==id)return;report=data;
  $('results').hidden=false;$('pre-results').hidden=true;
  const s=data.summary;
  $('static-metric').textContent=pct(s.static_fraction);$('static-detail').textContent=`${fmt(s.static_s,1)} שנ׳ מתוך ${fmt(s.static_s+s.moving_s,1)} מדידות`;
  $('speed-metric').textContent=fmt(s.mean_speed,3);$('jerk-metric').textContent=fmt(s.median_jerk,2);$('coverage-metric').textContent=pct(s.pose_coverage);
  $('coverage-detail').textContent=`${s.frames} פריימים · ${s.segments} רצפי מעקב`;
  $('ratio').textContent=s.static_to_move_ratio==null?'—':`${fmt(s.static_to_move_ratio,2)} : 1`;$('metric-coverage').textContent=pct(s.metric_coverage);
  renderTechnique(s);
  $('route-color').textContent=names[s.route.color]||'אין מספיק ראיות';$('route-swatch').style.background=colors[s.route.color]||'#b8bdb3';$('route-status').textContent=s.route.color?'הערכה':'לא ודאי';
  $('route-detail').textContent=`${s.route.distinct_holds} אחיזות שונות · ${s.route.contact_events} אירועי צבע`+(s.route.candidate?` · ${pct(s.route.evidence_share)} מהראיות לצבע ${names[s.route.candidate]}`:'');
  $('route-bars').replaceChildren();const total=Object.values(s.route.votes).reduce((a,b)=>a+b,0);
  for(const [name,vote] of Object.entries(s.route.votes).sort((a,b)=>b[1]-a[1])){
    const row=document.createElement('div');row.className='route-row';const label=document.createElement('span');label.textContent=names[name];const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=`${100*vote/total}%`;fill.style.background=colors[name];bar.append(fill);const number=document.createElement('small');number.textContent=pct(vote/total);row.append(label,bar,number);$('route-bars').append(row);
  }
  $('time-bar').replaceChildren();$('time-legend').replaceChildren();
  for(const [label,value,color] of [['סטטי',s.static_s,'#cc592d'],['תנועה',s.moving_s,'#708461'],['ללא מדידה',s.unknown_s,'#d5d9ce']]){
    const span=document.createElement('span');span.style.width=`${100*value/s.duration_s}%`;span.style.background=color;$('time-bar').append(span);const l=document.createElement('span'),i=document.createElement('i');i.style.background=color;l.append(i,`${label} ${fmt(value,1)} שנ׳`);$('time-legend').append(l);
  }
  $('contact-count').textContent=`${data.contacts.length} אירועים`;$('contacts').replaceChildren();
  const limbs={15:'יד שמאל',16:'יד ימין',27:'רגל שמאל',28:'רגל ימין'};
  for(const c of data.contacts){const b=document.createElement('button');const i=document.createElement('i');i.style.background=colors[c.color];b.append(i,`${time(c.time_s)} · ${limbs[c.landmark]} · ${names[c.color]}`);b.onclick=()=>$('video').currentTime=c.time_s;$('contacts').append(b);}
  if(!data.contacts.length)$('contacts').textContent='לא זוהו גפיים יציבות עם צבע אחיזה ברור.';
  chartLayers={motion:true,contacts:true};updateLayerButtons();setChartExpanded(true);$('contact-details').open=false;
  const translations={
    'Possible camera movement: speed, jerk and hold deduplication may be unreliable. Use a fixed camera.':'זוהתה אפשרות לתנועת מצלמה. היא עלולה להשפיע על המהירות, ה־jerk וספירת האחיזות. מומלץ צילום קבוע.',
    'Low hip visibility: much of the clip is untracked. Try a clearer view or a climber crop.':'נראות האגן נמוכה בחלק ניכר מהסרטון. נסה צילום ברור יותר או סימון אזור המטפס.',
    'Limited continuous tracking: less than half the clip supports reliable derivatives.':'פחות ממחצית הסרטון מאפשרת חישוב רציף של מדדי תנועה. יש להתייחס לתוצאות כמדגם חלקי.',
    'Not enough consistent hold evidence to estimate the route color.':'אין מספיק ראיות עקביות להערכת צבע המסלול.',
    'Input decoding ended before the reported frame count; results cover decoded frames only.':'קריאת הסרטון הסתיימה מוקדם מהצפוי. התוצאות מתייחסות לפריימים שנקראו בלבד.'
  };
  $('warnings').replaceChildren();for(const warning of data.warnings){const p=document.createElement('p');p.textContent=translations[warning]||warning;$('warnings').append(p);}
  $('download-video').href=url('video');$('download-csv').href=url('csv');$('download-json').href=url('report');
  $('seek').max=s.duration_s;$('seek').value=0;drawChart();
}
const NS='http://www.w3.org/2000/svg';
function svgEl(name,attrs={},text=''){const e=document.createElementNS(NS,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);e.textContent=text;return e;}
function renderTechnique(summary){
  const t=summary.technique;
  if(!t){
    $('drive-metric').textContent=$('arms-metric').textContent=$('pauses-metric').textContent=$('holds-metric').textContent='—';
    $('drive-detail').textContent='יש להריץ את הסרטון מחדש';$('pauses-detail').textContent='הדוח נוצר בגרסה קודמת';$('holds-detail').textContent='';
    $('technique-note').textContent='הניתוח הזה נוצר לפני הוספת מדדי הטכניקה. העלה את הסרטון מחדש כדי לקבל סיכום למטפס.';
    $('coaching-list').replaceChildren();return;
  }
  if(t.leg_signal_share==null){$('drive-metric').textContent='—';$('drive-detail').textContent=`${t.upward_phases} שלבי עלייה · אין מספיק שינויי מפרקים`;}
  else{$('drive-metric').textContent=`${Math.round(t.leg_signal_share*100)} / ${Math.round(t.arm_signal_share*100)}`;$('drive-detail').textContent=`רגליים / ידיים · ${t.resolved_upward_phases} שלבי עלייה`;}
  $('arms-metric').textContent=pct(t.straight_arm_static_fraction);
  $('pauses-metric').textContent=String(t.pause_count);$('pauses-detail').textContent=t.pause_count?`הארוכה ביותר ${fmt(t.longest_pause_s,1)} שנ׳`:'לא נמצאה עצירה מעל 0.5 שנ׳';
  $('holds-metric').textContent=`${t.hand_hold_uses} / ${t.foot_hold_uses}`;$('holds-detail').textContent='ידיים / רגליים · אחיזות שונות';
  $('technique-note').textContent='חלוקת הידיים והרגליים היא קשר קינמטי בין שינויי זוויות המפרקים לעליית האגן. היא אינה מדידת כוח, עומס או אנרגיה.';
  const texts={
    recording:()=>['שפר את תנאי הצילום','אין מספיק רצף אמין כדי להפיק הנחיית טכניקה מלאה. צלם את כל הגוף, ממצלמה קבועה ובזווית ישרה ככל האפשר לקיר.'],
    leg_drive_focus:()=>['נסה להתחיל את העלייה מהרגליים',`פשיטת הברכיים תרמה ${pct(t.leg_signal_share)} מאות התנועה שנלווה לעליית האגן. חפש מיקום רגל גבוה, העבר אליו משקל ורק אז משוך בידיים.`],
    leg_drive_positive:()=>['הרגליים משתתפות היטב בעלייה',`פשיטת הברכיים תרמה ${pct(t.leg_signal_share)} מאות התנועה שנלווה לעליית האגן. נסה לשמור על הדפוס גם בצעדים הקשים.`],
    straight_arm_focus:()=>['חפש מנוחות עם זרועות ישרות',`רק ${pct(t.straight_arm_static_fraction)} מתנוחות הזרוע שנמדדו בעצירה היו ישרות. שקיעת אגן וסיבוב הגוף עשויים להוריד עומס מהאמות.`],
    straight_arm_positive:()=>['עצירות יעילות יותר לידיים',`${pct(t.straight_arm_static_fraction)} מתנוחות הזרוע שנמדדו בעצירה היו ישרות, סימן אפשרי למנוחה חסכונית יותר.`],
    pauses_focus:()=>['תכנן לפני שאתה יוצא לדרך',`נמצאו ${t.pause_count} עצירות של חצי שנייה ומעלה; הארוכה נמשכה ${fmt(t.longest_pause_s,1)} שנ׳. נסה לקרוא את שני הצעדים הבאים מהקרקע.`],
    rhythm_focus:()=>['עבוד על קצב צפוי יותר','המהירות השתנתה מאוד בין מקטעים. בחזרה הבאה נסה לחבר שניים או שלושה צעדים בנשימה אחת, בלי להאיץ אחרי עצירה ארוכה.'],
    path_focus:()=>['בדוק אם אפשר לקצר את מסלול האגן',`יש פער גדול בין הדרך שעבר האגן לבין ההתקדמות נטו (ישירות ${pct(summary.path_directness)}). חפש סיבובי אגן והעברות משקל שמקרבים אותו לרגל התומכת; תנועה צדית מכוונת יכולה להיות חלק נכון מהפתרון.`],
    limited_evidence:()=>['נדרשת חזרה נוספת','המדדים שנאספו אינם מספיקים להמלצה יציבה. נתח ניסיון נוסף מאותה זווית והשווה בין שתי החזרות.']
  };
  $('coaching-list').replaceChildren();
  for(const advice of summary.coaching||[]){const copy=(texts[advice.code]||texts.limited_evidence)();const item=document.createElement('article');item.className=`coaching-item ${advice.kind||'recording'}`;const body=document.createElement('div'),title=document.createElement('strong'),p=document.createElement('p');title.textContent=copy[0];p.textContent=copy[1];body.append(title,p);item.append(body);$('coaching-list').append(item);}
}
function drawChart(){
  if(!report)return;
  const svg=$('chart');svg.replaceChildren();
  const metrics=report.metrics||[],contacts=report.contacts||[],duration=Math.max(report.summary.duration_s||0,0.001);
  const showMotion=chartLayers.motion,showContacts=chartLayers.contacts,x0=55,x1=885,width=x1-x0;
  const motionTop=15,motionBottom=showContacts?112:184,handY=showMotion?151:72,footY=showMotion?184:132;
  const graphBottom=showContacts?footY:motionBottom;
  const unit=showMotion?(chartKey==='speed'?'image diagonals / s':'image diagonals / s³'):'צבעי מגעים';
  $('chart-unit').textContent=unit;
  $('chart-hint').textContent=showMotion&&showContacts
    ?'פערים בקו הם זמן ללא מדידה. עיגול מסמן יד ומעוין מסמן רגל; צבע הנקודה הוא צבע האחיזה.'
    :showMotion
      ?(chartKey==='speed'?'פערים בקו הם זמן ללא מדידה. שיאים מציינים תנועה מהירה יותר.':'פערים בקו הם זמן ללא מדידה. Jerk נמוך ומתמשך פחות מצביע בדרך כלל על תנועה חלקה יותר.')
      :'עיגול מסמן מגע יד ומעוין מסמן מגע רגל. צבע הנקודה הוא צבע האחיזה שזוהה.';
  svg.setAttribute('aria-label',showMotion&&showContacts?'גרף תנועה ורצף מגעים לאורך הטיפוס':showMotion?'גרף תנועה לאורך הטיפוס':'רצף מגעים לאורך הטיפוס');

  for(let i=0;i<=4;i++){
    const x=x0+width*i/4;
    svg.append(svgEl('line',{x1:x,x2:x,y1:10,y2:graphBottom+6,stroke:'#e5e6df','stroke-width':1}));
    svg.append(svgEl('text',{x,y:228,'text-anchor':i===0?'start':i===4?'end':'middle'},`${(duration*i/4).toFixed(1)}s`));
  }

  if(showMotion){
    const values=metrics.map(metric=>metric[chartKey]).filter(value=>Number.isFinite(value));
    const max=Math.max(...values,chartKey==='speed'?0.02:1)*1.1;
    for(let i=0;i<4;i++){
      const y=motionTop+(motionBottom-motionTop)*i/3;
      svg.append(svgEl('line',{x1:x0,x2:x1,y1:y,y2:y,stroke:'#e1e3da','stroke-width':1}));
      svg.append(svgEl('text',{x:x0-9,y:y+4,'text-anchor':'end'},(max*(1-i/3)).toFixed(chartKey==='speed'?3:1)));
    }
    let path='',previous=-2;
    for(const metric of metrics){
      const value=metric[chartKey];
      if(!Number.isFinite(value)){previous=-2;continue;}
      const x=x0+Math.max(0,Math.min(1,metric.time_s/duration))*width;
      const y=motionBottom-value/max*(motionBottom-motionTop);
      path+=`${metric.frame_index===previous+1?'L':'M'}${x.toFixed(2)},${y.toFixed(2)} `;previous=metric.frame_index;
    }
    svg.append(svgEl('path',{class:'motion-line',d:path,stroke:'#cc592d','stroke-width':2,fill:'none','stroke-linecap':'round','stroke-linejoin':'round'}));
    if(!values.length)svg.append(svgEl('text',{x:470,y:(motionTop+motionBottom)/2,'text-anchor':'middle'},'אין מספיק פריימים רציפים לחישוב'));
  }

  if(showContacts){
    svg.append(svgEl('text',{x:x0-10,y:handY+4,'text-anchor':'end'},'ידיים'),svgEl('text',{x:x0-10,y:footY+4,'text-anchor':'end'},'רגליים'));
    for(const y of [handY,footY])svg.append(svgEl('line',{x1:x0,x2:x1,y1:y,y2:y,stroke:'#cfd4c9','stroke-width':2}));
    const sorted=[...contacts].sort((a,b)=>a.time_s-b.time_s),points=[];
    for(const contact of sorted){const hand=contact.landmark===15||contact.landmark===16,x=x0+Math.max(0,Math.min(1,contact.time_s/duration))*width,y=hand?handY:footY;points.push(`${x},${y}`);}
    if(points.length>1)svg.append(svgEl('polyline',{class:'contact-path',points:points.join(' '),fill:'none',stroke:'#aeb4aa','stroke-width':1.5,'stroke-dasharray':'3 4'}));
    const limbs={15:'יד שמאל',16:'יד ימין',27:'רגל שמאל',28:'רגל ימין'};
    for(const contact of sorted){
      const hand=contact.landmark===15||contact.landmark===16,x=x0+Math.max(0,Math.min(1,contact.time_s/duration))*width,y=hand?handY:footY;
      const attrs={fill:colors[contact.color]||'#888',stroke:'#fff','stroke-width':2};
      const node=hand?svgEl('circle',{...attrs,cx:x,cy:y,r:7}):svgEl('polygon',{...attrs,points:`${x},${y-8} ${x+8},${y} ${x},${y+8} ${x-8},${y}`});
      const description=`${time(contact.time_s)} · ${limbs[contact.landmark]} · ${names[contact.color]||contact.color}`;
      const seekToContact=()=>{$('video').currentTime=contact.time_s;updateTime();};
      node.classList.add('hold-node');node.setAttribute('tabindex','0');node.setAttribute('role','button');node.setAttribute('aria-label',description);node.append(svgEl('title',{},description));
      node.onclick=event=>{event.stopPropagation();seekToContact();};node.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();seekToContact();}};svg.append(node);
    }
    if(!contacts.length)svg.append(svgEl('text',{x:470,y:(handY+footY)/2+4,'text-anchor':'middle'},'לא זוהו מגעים צבעוניים יציבים בסרטון'));
  }
  svg.append(svgEl('line',{id:'playhead',x1:x0,x2:x0,y1:10,y2:graphBottom+6,stroke:'#69785c','stroke-width':1,'stroke-dasharray':'4 3'}));updateTime();
}
function updateTime(){const t=$('video').currentTime||0;$('timecode').textContent=time(t);$('seek').value=t;if(report&&$('playhead')){const x=55+Math.max(0,Math.min(1,t/Math.max(report.summary.duration_s,0.001)))*830;$('playhead').setAttribute('x1',x);$('playhead').setAttribute('x2',x);}}
$('video').ontimeupdate=updateTime;
$('video').onerror=()=>{if(selected)showError('הדפדפן לא הצליח לנגן את קובץ המקור. אפשר להמשיך בניתוח; התוצאה תומר לתצוגה נתמכת.');};
$('seek').oninput=()=>{$('video').currentTime=+$('seek').value;};
$('chart').onclick=e=>{if(report){const point=$('chart').createSVGPoint();point.x=e.clientX;point.y=e.clientY;const x=point.matrixTransform($('chart').getScreenCTM().inverse()).x;$('video').currentTime=Math.max(0,Math.min(1,(x-55)/830))*report.summary.duration_s;updateTime();}};
document.querySelectorAll('[data-chart]').forEach(b=>b.onclick=()=>{chartKey=b.dataset.chart;document.querySelectorAll('[data-chart]').forEach(x=>x.classList.toggle('active',x===b));drawChart();});
function updateLayerButtons(){for(const button of document.querySelectorAll('[data-layer]')){const active=chartLayers[button.dataset.layer];button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}$('chart-tabs').hidden=!chartLayers.motion;}
document.querySelectorAll('[data-layer]').forEach(button=>button.onclick=()=>{const layer=button.dataset.layer;if(chartLayers[layer]&&Object.values(chartLayers).filter(Boolean).length===1)return;chartLayers[layer]=!chartLayers[layer];updateLayerButtons();drawChart();});
function setChartExpanded(expanded){$('chart-panel').hidden=!expanded;$('chart-collapse').setAttribute('aria-expanded',String(expanded));$('chart-collapse').textContent=expanded?'הסתר גרף ↑':'פתח גרף ↓';}
$('chart-collapse').onclick=()=>setChartExpanded($('chart-panel').hidden);
document.querySelectorAll('[data-video]').forEach(b=>b.onclick=()=>setVideo(b.dataset.video,true));
function updateCropLabel(){$('crop-label').textContent=crop?'אזור המטפס סומן':'כל הפריים';$('crop-clear').hidden=!crop;}
function setCropMode(enabled){cropMode=enabled;$('crop-canvas').hidden=!enabled;$('crop-help').hidden=!enabled;$('crop-toggle').textContent=enabled?'סיום סימון ✓':'סימון אזור המטפס ⌗';if(enabled){$('video').pause();drawCrop();}}
$('crop-toggle').onclick=()=>{if(cropImage.complete&&cropImage.naturalWidth)setCropMode(!cropMode);};
$('crop-clear').onclick=()=>{crop=null;updateCropLabel();if(cropMode)drawCrop();};
function drawCrop(){
  const canvas=$('crop-canvas'),rect=$('video-stage').getBoundingClientRect();canvas.width=rect.width;canvas.height=rect.height;
  const ctx=canvas.getContext('2d'),scale=Math.min(canvas.width/cropImage.width,canvas.height/cropImage.height),w=cropImage.width*scale,h=cropImage.height*scale,x=(canvas.width-w)/2,y=(canvas.height-h)/2;
  cropBox={x,y,w,h};ctx.fillStyle='#212922';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(cropImage,x,y,w,h);
  if(crop){ctx.fillStyle='rgba(0,0,0,.35)';ctx.fillRect(x,y,w,h);const [x0,y0,x1,y1]=crop;ctx.save();ctx.beginPath();ctx.rect(x+x0*w,y+y0*h,(x1-x0)*w,(y1-y0)*h);ctx.clip();ctx.drawImage(cropImage,x,y,w,h);ctx.restore();ctx.strokeStyle='#ed824f';ctx.lineWidth=2;ctx.strokeRect(x+x0*w,y+y0*h,(x1-x0)*w,(y1-y0)*h);}
}
function cropPoint(event){const r=$('crop-canvas').getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-r.left-cropBox.x)/cropBox.w)),Math.max(0,Math.min(1,(event.clientY-r.top-cropBox.y)/cropBox.h))];}
$('crop-canvas').onpointerdown=e=>{dragStart=cropPoint(e);$('crop-canvas').setPointerCapture(e.pointerId);};
$('crop-canvas').onpointermove=e=>{if(!dragStart)return;const p=cropPoint(e);crop=[Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.max(p[0],dragStart[0]),Math.max(p[1],dragStart[1])];drawCrop();};
$('crop-canvas').onpointerup=()=>{dragStart=null;if(crop&&(crop[2]-crop[0]<.05||crop[3]-crop[1]<.05))crop=null;updateCropLabel();drawCrop();};
$('crop-canvas').onpointercancel=()=>{dragStart=null;};
window.addEventListener('resize',()=>{if(cropMode)drawCrop();});
async function init(){
  const [health,samples]=await Promise.all([api('/api/health'),api('/api/samples')]);$('model-status').textContent=health.model_ready?'המודל מוכן':'יש להוריד מודל';
  if(!health.model_ready)showError('מודל המעקב חסר. יש להריץ במחשב: python setup_model.py');
  const titles={'body-trajectory-input.mp4':'תנועה דינמית · 3.4 שנ׳','warp-dynamic-input.mp4':'טיפוס בתקרה · מצלמה בתנועה','warp-fixed-input.mp4':'קיר טיפוס · מצלמה קבועה'};
  for(const sample of samples){const o=document.createElement('option');o.value=sample.name;o.textContent=titles[sample.name]||sample.name;$('samples').append(o);}$('sample-credit').hidden=!samples.length;
  await refreshHistory();const list=await api('/api/jobs');const latest=list.find(j=>j.status==='complete')||list.find(j=>['running','queued','ready'].includes(j.status));if(latest)await selectJob(latest);
}
safeAction(init)();
