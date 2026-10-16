import {W,H,GRAPH,pairSets,material} from './engine.js';
import {TASKS,CHECKPOINTS,needSplits,sweepPairs,trajectory,selectCheckpoint,commitDamagePrediction,revealDamagePrediction,equalMaterialUniform} from './gate2.js';
import {objective} from './engine.js';
const $=id=>document.getElementById(id);
const fmt=(x,d=3)=>Number(x).toFixed(d);
const color={specialist:'#ffb46e',diverse:'#70c7cc'};
let current=null,commit=null,revealed=null;
function coords(i){return [23+(i%W)*38,22+Math.floor(i/W)*29];}
function drawNet(h){let edges='';
 for(let i=0;i<GRAPH.edges.length;i++){
  const e=GRAPH.edges[i],a=coords(e.a),b=coords(e.b),v=h[i];
  edges+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${v>.08?'#8fce9a':'#627c89'}" stroke-width="${.7+3*Math.sqrt(v)}" opacity="${.2+Math.min(.8,v/1.4)}" stroke-linecap="round"/>`;
 }
 return `<svg viewBox="0 0 430 282" aria-label="Topology: green thickness represents grown conductance">${edges}</svg>`;
}
function lineChart(traces){const w=640,h=235,pad=37;
 const entries=traces.flatMap(([name,series])=>series.map(x=>({...x,arm:name})));
 const ymax=Math.max(1,...entries.map(x=>x.validation))*1.08;
 const x=n=>pad+(w-pad-12)*n/Math.max(320,entries.at(-1)?.step??320);
 const y=v=>h-30-(h-54)*v/ymax;
 let out='';
 for(let i=0;i<=4;i++){const val=ymax*i/4,Y=y(val);out+=`<line x1="${pad}" y1="${Y}" x2="${w}" y2="${Y}" stroke="#52616d" opacity=".34"/><text x="4" y="${Y+3}" fill="#9daeb6" font-size="10">${fmt(val,1)}</text>`;}
 for(const [name,series] of traces){const pts=series.map(f=>`${x(f.step).toFixed(1)},${y(f.validation).toFixed(1)}`).join(' ');
 out+=`<polyline points="${pts}" fill="none" stroke="${color[name]}" stroke-width="2.8"/>`;
 const chosen=selectCheckpoint(current[name]);out+=`<circle cx="${x(chosen.step)}" cy="${y(chosen.validation)}" r="5" fill="${color[name]}" stroke="#fff"/>`;
 }
 return `<svg viewBox="0 0 ${w} ${h}" aria-label="Validation cost across checkpoints: minimum determines selection">${out}</svg>`;
}
function dangerBars(predictions,actual){let s='';const keys=Object.keys(predictions),max=Math.max(.01,...Object.values(predictions).flat(),...(actual||[]))*1.1;
 for(let i=0;i<12;i++){
  const x=32+i*47,bottom=210;
  keys.forEach((k,j)=>{let height=Math.max(0,predictions[k][i])/max*171;
   s+=`<rect x="${x+j*7}" y="${bottom-height}" height="${height}" width="6.2" fill="${['#d4ff8c','#ffb46e','#70c7cc','#aeb3fa'][j]}" rx="1"/>`;});
  if(actual){let ht=Math.max(0,actual[i])/max*171;s+=`<circle cx="${x+12}" cy="${bottom-ht}" r="4.5" fill="#fff" stroke="#092129" stroke-width="1.5"/>`;}
  s+=`<text x="${x+13}" y="228" font-size="10" fill="#a9bdc2" text-anchor="middle">${i+1}</text>`;
 }
 return `<svg viewBox="0 0 610 242" aria-label="Predicted damage bars. White circles are revealed measured damage."><line x1="26" y1="210" x2="608" y2="210" stroke="#617983"/>${s}</svg>`;
}
function clear(){current=null;commit=null;revealed=null;$('commit').disabled=true;$('reveal').disabled=true;$('export').disabled=true;$('stage').textContent='Unprepared';$('chart').innerHTML='';$('network-s').innerHTML='';$('network-d').innerHTML='';$('comparison').innerHTML='';$('predictions').innerHTML='';$('test-panel').hidden=true;}
$('task').addEventListener('change',clear);
$('prepare').addEventListener('click',()=>{
 clear();$('prepare').disabled=true;$('stage').textContent='Growing 2 models and selecting checkpoints…';
 // Yield to paint the status first.
 setTimeout(()=>{
  try{
   const task=$('task').value,steps=Number($('steps').value);
   const specialist=trajectory(task,'specialist',{steps,checkpoints:CHECKPOINTS}),diverse=trajectory(task,'diverse',{steps,checkpoints:CHECKPOINTS});
   current={task,steps,specialist,diverse};
   $('chart').innerHTML=lineChart([['specialist',specialist.frames],['diverse',diverse.frames]]);
   $('network-s').innerHTML=drawNet(selectCheckpoint(specialist).h);
   $('network-d').innerHTML=drawNet(selectCheckpoint(diverse).h);
   $('comparison').innerHTML=['specialist','diverse'].map(a=>{
    const f=selectCheckpoint(current[a]),final=current[a].frames.at(-1);
    return `<tr><td style="color:${color[a]}">${a}</td><td>${f.step}</td><td>${fmt(f.validation)}</td><td>${fmt(final.validation)}</td><td>${fmt(f.material,1)}</td></tr>`;
   }).join('');
   $('stage').textContent='Checkpoints chosen from validation only. Test remains unqueried.';$('commit').disabled=false;
  }catch(e){$('stage').textContent='Error: '+e.message;console.error(e);}finally{$('prepare').disabled=false;}
 },30);
});
$('commit').addEventListener('click',()=>{
 if(!current)return;
 commit={}; for(const name of ['specialist','diverse']){
  const f=selectCheckpoint(current[name]);commit[name]=commitDamagePrediction(Float64Array.from(f.h),current.task);
 }
 $('commit').disabled=true;$('reveal').disabled=false;$('stage').textContent='Predictions committed on training demands only. Test labels and scores still hidden.';
 $('predictions').innerHTML=renderPredictions();
});
function renderPredictions(){const name=$('inspect').value;
 return `<div class="legend2"><span style="color:#d4ff8c">● Causal training</span><span style="color:#ffb46e">● Material</span><span style="color:#70c7cc">● Train flow</span><span style="color:#aeb3fa">● Train gradient</span>${revealed?'<span style="color:#fff">● Observed</span>':''}</div>`+
 dangerBars(commit[name].predictions,revealed?.[name]?.actual);
}
$('inspect').addEventListener('change',()=>{if(commit)$('predictions').innerHTML=renderPredictions();});
$('reveal').addEventListener('click',()=>{
 if(!commit||!current)return;
 revealed={};
 const sweep=sweepPairs(current.task),stress=needSplits(current.task).test;
 const metricRows=[];
 for(const name of ['specialist','diverse']){
  const traj=current[name],f=selectCheckpoint(traj),last=traj.frames.at(-1),h=Float64Array.from(f.h);
  const trueResult=revealDamagePrediction(h,commit[name],{split:'sweep'});
  const uniform=equalMaterialUniform(h);
  revealed[name]={...trueResult,
   selected:{step:f.step,validation:f.validation,stress:objective(h,stress).loss,sweep:objective(h,sweep).loss},
   final:{step:last.step,validation:last.validation,stress:objective(last.h,stress).loss,sweep:objective(last.h,sweep).loss},
   uniform:{stress:objective(uniform,stress).loss,sweep:objective(uniform,sweep).loss}};
  const z=revealed[name];
  metricRows.push(`<tr><td>${name}</td><td>${fmt(z.selected.sweep)}</td><td>${fmt(z.final.sweep)}</td><td>${fmt(z.uniform.sweep)}</td><td>${fmt(z.selected.stress)}</td></tr>`);
 }
 $('final-table').innerHTML=metricRows.join('');
 $('score-title').textContent=`24 new demands — measured damage versus four training-only predictions`;
 $('score-table').innerHTML=['specialist','diverse'].flatMap(name=>Object.entries(revealed[name].scores).map(([kind,r])=>`<tr><td>${name}</td><td>${kind}</td><td>${fmt(r.pearson)}</td><td>${fmt(r.rmse,4)}</td><td>${r.top3}/3</td></tr>`)).join('');
 $('test-panel').hidden=false;$('reveal').disabled=true;$('export').disabled=false;
 $('stage').textContent='Test revealed. No checkpoint or gain was selected from test data.';
 $('predictions').innerHTML=renderPredictions();
});
$('export').addEventListener('click',()=>{
 const data={protocol:'Growth Microscope V2 -- prospective model selection and held-out causal damage',
 timestamp:new Date().toISOString(),task:current.task,steps:current.steps,validation:'three shifted V1 held pairs',
 test:'three stress pairs plus 24 disjoint deterministic sampled pairs',
 constraints:{pairsPerTrainingGradient:3,budget:24,materialAllocation:'conductances on fixed 11x9 lattice',checkpointSelection:'validation minimum only',
 noCryptographicBlind:true},
 checkpoints:Object.fromEntries(['specialist','diverse'].map(a=>[a,current[a].frames])),
 committed:commit,unsealed:revealed};
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download=`growth-microscope-gate2-${current.task}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),700);
});
