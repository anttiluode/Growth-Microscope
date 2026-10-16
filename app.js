import {W,H,GRAPH,REGIONS,pairSets,fresh,growStep,objective,material,causalMap,fingerprint,compare,ablate} from './engine.js';
const $=id=>document.getElementById(id);
const el={need:$('need'),network:$('network'),history:$('history'),mstep:$('m-step'),mmaterial:$('m-material'),mtrain:$('m-train'),mtest:$('m-test'),checkpoint:$('checkpoint'),regionGrid:$('regionGrid'),regionTitle:$('region-title'),regionDesc:$('region-description'),fingerprint:$('fingerprint'),predplot:$('predplot'),predict:$('predict'),reveal:$('reveal'),message:$('gate-message'),scoreRows:$('score-rows'),auditStatus:$('audit-status')};
const state={task:'bridge',h:fresh(),step:0,history:[],frames:[],region:5,gate:null,revealed:false,auto:null,signals:null};
const fmt=x=>Number(x).toFixed(4);
const COLORS={'Train causal fingerprint':'#d4ff8c','Weight only':'#ffb46e','Held-out activity':'#70c7cc','Held-out adjoint sensitivity':'#aeb3fa'};
const checkpointSteps=new Set([0,20,40,80,140,220,320,480,640]);
const safe=(n,a,b)=>Math.max(a,Math.min(b,n));
function historyPoint(){const p=pairSets(state.task);return {step:state.step,train:objective(state.h,p.train).loss,held:objective(state.h,p.held).loss,material:material(state.h)};}
function stop(){if(state.auto){clearInterval(state.auto);state.auto=null;}$('play').textContent='▶ Auto';}
function reset(){stop();state.h=fresh();state.step=0;state.history=[historyPoint()];state.frames=[{step:0,h:Array.from(state.h)}];invalidate();render();}
function invalidate(){state.gate=null;state.revealed=false;el.reveal.disabled=true;el.predict.disabled=false;el.auditStatus.textContent='HIDDEN';el.message.textContent='Grow the network, then predict what a held-out cut will do.';el.scoreRows.replaceChildren();}
function advance(count){if(count<1)return;const p=pairSets(state.task);for(let i=0;i<count;i++){
  state.h=growStep(state.h,p.train).h;state.step++;
  if(state.step%5===0)state.history.push(historyPoint());
  if(checkpointSteps.has(state.step))state.frames.push({step:state.step,h:Array.from(state.h)});
}invalidate();render();}
function render(){const p=pairSets(state.task),last=state.history.at(-1);el.mstep.textContent=state.step;el.mmaterial.textContent=material(state.h).toFixed(2);el.mtrain.textContent=fmt(last.train);el.mtest.textContent=fmt(last.held);
  renderNet(p.train);renderHistory();renderRegions(p.train);renderFingerprint(p.train);renderCheckpoint();renderPrediction();}
function line(a,b,w,color,opacity=1,dash=''){return `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${w}" opacity="${opacity}" ${dash?`stroke-dasharray="${dash}"`:''} stroke-linecap="round"/>`;}
function coords(index){const x=index%W,y=Math.floor(index/W);return [45+x*(570/(W-1)),43+y*(413/(H-1))];}
function renderNet(pairs){let s='';const region=state.region;
  s+=`<rect x="${40+(region%4)*570/4}" y="${34+Math.floor(region/4)*413/3}" width="${570/4}" height="${413/3}" fill="#f6b47c" opacity=".04" stroke="#ffc08a" stroke-width="2" stroke-dasharray="6 6"/>`;
  for(let i=0;i<GRAPH.edges.length;i++){
    const e=GRAPH.edges[i],h=state.h[i],marked=e.region===region;
    s+=line(coords(e.a),coords(e.b),.7+Math.sqrt(h)*4.4,marked?'#ffae76':'#a0d9bb', .18+Math.min(.82,h/1.4));
  }
  for(let i=0;i<GRAPH.n;i++){const [x,y]=coords(i);s+=`<circle cx="${x}" cy="${y}" r="2.3" fill="#6a8f93"/>`;}
  const sourceSet=new Set(pairs.map(x=>x[0])), sinkSet=new Set(pairs.map(x=>x[1]));
  for(const idx of sourceSet){const [x,y]=coords(idx);s+=`<circle cx="${x}" cy="${y}" r="8.5" fill="#70c7cc" stroke="#e2ffff" stroke-width="1.5"/>`;}
  for(const idx of sinkSet){const [x,y]=coords(idx);s+=`<circle cx="${x}" cy="${y}" r="8.5" fill="#ffae76" stroke="#ffe3c8" stroke-width="1.5"/>`;}
  s+=`<text x="16" y="491" fill="#95aaa9" font-size="11">${GRAPH.n} nodes · ${GRAPH.edges.length} recruitable edges · one grounded potential gauge</text>`;
  el.network.innerHTML=s;
}
function renderHistory(){const rows=state.history;
 let s='';const pad=27,bottom=128,right=630, width=right-pad,height=107;
 const max=Math.max(...rows.flatMap(r=>[r.train,r.held]))*1.1;
 const X=step=>pad+(step/Math.max(320,Math.ceil(state.step/80)*80))*width;
 const Y=val=>bottom-height*val/Math.max(max,1e-8);
 for(let i=0;i<5;i++){let y=bottom-i*height/4;s+=line([pad,y],[right,y],.65,'#46606a',.5);s+=`<text x="0" y="${y+3}" font-size="10" fill="#94aaa9">${(i*max/4).toFixed(1)}</text>`;}
 for(const [key,color,dash] of [['train','#d4ff8c',''],['held','#ffae76','7 5']]){
  const pts=rows.map(r=>`${X(r.step).toFixed(1)},${Y(r[key]).toFixed(1)}`).join(' ');
  s+=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.2" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
 }
 s+=`<text x="${right}" y="148" text-anchor="end" font-size="10" fill="#94aaa9">step ${state.step}</text>`;
 el.history.innerHTML=s;
}
function regionTrain(){return causalMap(state.h,pairSets(state.task).train,.5);}
function renderRegions(pairs){const harms=regionTrain(), max=Math.max(...harms,1e-8);
 el.regionGrid.innerHTML=REGIONS.map(r=>`<button type="button" class="region ${state.region===r.id?'active':''}" data-region="${r.id}" aria-pressed="${state.region===r.id}"><span class="rid">${r.id+1} / ${r.name}</span><span class="rval">+${harms[r.id].toFixed(3)}</span><span class="rbar" style="width:${100*harms[r.id]/max}%"></span></button>`).join('');
 el.regionGrid.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{state.region=Number(b.dataset.region);render();}));
 el.regionTitle.textContent=REGIONS[state.region].name;
 const selected=harms[state.region];
 el.regionDesc.textContent=`Cutting 50% of grown conductance here raises TRAINING resistance by ${selected.toFixed(4)}. This is a measured intervention, not an inference from appearance.`;
}
function renderFingerprint(train){const fp=fingerprint(state.h,train.map(p=>[p]))[state.region],max=Math.max(1e-9,...fp.profile);
 el.fingerprint.innerHTML=fp.profile.map((v,i)=>`<div class="fp-row"><span>Need ${i+1}</span><div class="track"><span style="width:${100*v/max}%"></span></div><small>+${v.toFixed(4)}</small></div>`).join('')+`<div class="caption">Current role: <strong>${fp.role}</strong> · intervention impact for three trained source–sink pairs.</div>`;
}
function renderCheckpoint(){const s=state.frames.map(f=>`<option value="${f.step}">${f.step} steps</option>`).join('');el.checkpoint.innerHTML=s;el.checkpoint.value=String(state.step);
 if(el.checkpoint.value!==String(state.step)){const o=document.createElement('option');o.value=String(state.step);o.textContent=`${state.step} · live`;o.selected=true;el.checkpoint.append(o);}}
function predict(){if(state.auto)stop();const p=pairSets(state.task);state.gate=compare(state.h,p.train,p.held);state.revealed=false;el.reveal.disabled=false;el.predict.disabled=true;el.message.textContent='Predictions frozen. You can examine the estimates now. Reveal to score them against the unseen terminal cuts.';renderPrediction();}
function reveal(){if(!state.gate)return;state.revealed=true;el.reveal.disabled=true;el.auditStatus.textContent='REVEALED';el.message.textContent='Held-out cut outcomes revealed. Check whether the causal fingerprint actually beats weight and activity baselines.';renderPrediction();}
function renderPrediction(){const q=state.gate,steps=12;
 if(!q){el.predplot.innerHTML=`<text x="345" y="124" fill="#96aead" text-anchor="middle" font-size="15">No predictions yet · press “Make predictions”</text>`;return;}
 let s='',y0=212,barH=165;const values=Object.values(q.predictions).flat();if(state.revealed)values.push(...q.actual);
 const max=Math.max(0.001,...values)*1.13;
 s+=line([32,y0],[671,y0],1,'#5e777b');
 const keys=Object.keys(COLORS);
 for(let i=0;i<steps;i++){
   const x=39+i*53,band=35;
   if(i===state.region)s+=`<rect x="${x-7}" y="24" width="${band+14}" height="194" fill="#d4ff8c" opacity=".07" rx="4"/>`;
   keys.forEach((key,j)=>{
     const h=Math.max(0,q.predictions[key][i]/max*barH),px=x+j*7.5;
     s+=`<rect x="${px}" y="${y0-h}" width="6.1" height="${h}" fill="${COLORS[key]}" rx="1.1"/>`;
   });
   if(state.revealed){const h=Math.max(0,q.actual[i]/max*barH);s+=`<circle cx="${x+13}" cy="${y0-h}" r="4.4" fill="#f7f9f6" stroke="#131d24" stroke-width="1"/>`;}
   s+=`<text x="${x+13}" y="232" font-size="10" text-anchor="middle" fill="#a3bbb9">${i+1}</text>`;
 }
 for(let k=1;k<5;k++){const yy=y0-barH*k/4;s+=line([32,yy],[675,yy],.5,'#5e777b',.33);s+=`<text x="4" y="${yy+3}" font-size="9" fill="#9ab0b0">${(max*k/4).toFixed(2)}</text>`;}
 el.predplot.innerHTML=s;
 el.scoreRows.innerHTML=keys.map(key=>{
   const row=q.scores[key];return `<tr><td><span style="color:${COLORS[key]}">●</span> ${key.replace('Held-out ','').replace('Train ','')}</td><td>${state.revealed?row.r.toFixed(3):'—'}</td><td>${state.revealed?row.rmse.toFixed(4):'—'}</td><td>${state.revealed?row.top3_overlap+'/3':'—'}</td></tr>`;
 }).join('');
}
function download(){const p=pairSets(state.task),data={protocol:'Growth Microscope v1',task:state.task,step:state.step,budget:24,
 scaffold:.06,algorithm:'projected negative compliance gradient',graph:{nodes:GRAPH.n,edges:GRAPH.edges},demands:p,
 snapshots:state.frames,history:state.history,finalConductance:Array.from(state.h),causalTraining:Array.from(regionTrain()),
 ...(state.gate?{audit:state.gate}:{}),caveat:'The group definitions and task objective are provided; not a learned explanation of neural semantics.'};
 const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`growth-microscope-${state.task}-${state.step}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
el.need.addEventListener('change',()=>{state.task=el.need.value;reset();});
$('advance40').addEventListener('click',()=>{stop();advance(40);});
$('growFull').addEventListener('click',()=>{stop();advance(Math.max(0,320-state.step));});
$('reset').addEventListener('click',reset);
$('play').addEventListener('click',()=>{if(state.auto){stop();return;}el.predict.disabled=true;$('play').textContent='Ⅱ Pause';state.auto=setInterval(()=>{if(state.step>=640){stop();el.predict.disabled=false;return;}advance(5);el.predict.disabled=true;},160);});
el.checkpoint.addEventListener('change',()=>{const f=state.frames.find(x=>x.step===Number(el.checkpoint.value));if(!f)return;stop();state.h=Float64Array.from(f.h);state.step=f.step;state.history=state.history.filter(r=>r.step<=f.step);state.frames=state.frames.filter(r=>r.step<=f.step);invalidate();render();});
el.predict.addEventListener('click',predict);el.reveal.addEventListener('click',reveal);$('export').addEventListener('click',download);
reset();
