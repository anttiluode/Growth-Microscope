import {GRAPH,W,xy} from './engine.js';
import {trajectory,selectCheckpoint} from './gate2.js';
import {K,commitInterpretation,revealInterpretation} from './gate3.js';
const $=id=>document.getElementById(id);
const fmt=(x,d=4)=>Number(x).toFixed(d);
const PALETTE=['#69d4c5','#efb86b','#d492db','#8eb6fb','#a1da87','#f08596','#e5dc75','#8dc6b6','#c2aafa','#d9a786','#90b6ec','#d1e0ba'];
const color=i=>PALETTE[i%PALETTE.length];
let prepared=null,committed=null,revealed=null,selectedGroup=-1;
function clear(){prepared=null;committed=null;revealed=null;selectedGroup=-1;for(const id of ['commit','reveal','export'])$(id).disabled=true; $('revealed').hidden=true;$('groups').innerHTML='<div class="empty">No groups yet.</div>'; $('network').innerHTML='';$('plot').innerHTML='';$('inspect-edge').innerHTML='';$('commit-note').textContent='Prepare, then lock the explanations before the test is evaluated.';$('evolution').innerHTML='';}
const coords=i=>[19+(i%W)*34,19+Math.floor(i/W)*30];
function draw(){if(!prepared)return;
 const groups=committed?.groupings?.[$('coloring').value];
 let s='<svg viewBox="0 0 380 280" role="img" aria-label="Conductive lattice, edge colors show discovered functional clusters">';
 for(let i=0;i<GRAPH.edges.length;i++){
  const e=GRAPH.edges[i],a=coords(e.a),b=coords(e.b),v=prepared.h[i];
  const group=groups?.[i]??e.region;
  s+=`<line class="edge ${selectedGroup>=0&&group!==selectedGroup?'dim':''}" data-edge="${i}" data-group="${group}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color(group)}" stroke-width="${.75+Math.sqrt(v)*4}" opacity="${Math.max(.26,Math.min(1,.3+v))}" stroke-linecap="round"><title>Edge ${i}, group ${group+1}, material ${fmt(v,5)}</title></line>`;
 }
 s+='</svg>';$('network').innerHTML=s;
 $('network').querySelectorAll('[data-edge]').forEach(el=>el.addEventListener('click',()=>{
  const e=Number(el.dataset.edge);selectedGroup=Number(el.dataset.group);draw();
  const p=committed?.trainingProfiles[e];
  $('inspect-edge').textContent=`Edge ${e} • group ${selectedGroup+1} • grown material ${fmt(prepared.h[e],5)} • train-cut harm ${p?.map(x=>fmt(x,5)).join(' / ')??'not yet measured'}${revealed&&committed.probeIds.includes(e)?` • HELD-OUT cut harm ${fmt(revealed.actual[committed.probeIds.indexOf(e)],5)}`:''}`;
  highlightCards();
 }));
}
function highlightCards(){ $('groups').querySelectorAll('[data-group]').forEach(el=>el.classList.toggle('active',Number(el.dataset.group)===selectedGroup)); }
function groupCards(){if(!committed)return;
 const means=committed.clusters.map(c=>c.signature),max=Math.max(1e-9,...means.flat());
 $('groups').innerHTML=committed.clusters.map(c=>`<button class="v3-group" data-group="${c.id}"><span class="v3-name" style="color:${color(c.id)}">■ Group ${c.id+1}</span><small>${c.edges.length} edges · material ${fmt(c.material,2)}</small><span class="v3-signature">${c.signature.map((x,j)=>`<i title="Train probe ${j+1}: ${fmt(x,5)}" style="height:${Math.max(5,Math.round(100*x/max))}%;background:${color(c.id)};opacity:${.55+j*.18}"></i>`).join('')}</span></button>`).join('');
 $('groups').querySelectorAll('[data-group]').forEach(el=>el.addEventListener('click',()=>{const c=Number(el.dataset.group);selectedGroup=selectedGroup===c?-1:c;draw();highlightCards();
 const info=committed.clusters[c];$('inspect-edge').textContent=`Group ${c+1}: ${info.edges.length} edges. Mean training cut harm across three stimuli: ${info.signature.map(x=>fmt(x,6)).join(' / ')}. Group assignments were discovered without coordinates.`;
 }));
}
function plotDevelopment(){if(!prepared||!committed)return;
 const series=committed.clusters.map(c=>prepared.traj.frames.map(f=>c.edges.reduce((sum,e)=>sum+f.h[e],0)));
 const max=Math.max(1,...series.flat())*1.1,w=550,h=130,left=28;
 let svg=`<span class="v3-plot-title">Retrospective history: how material entered each FINAL learned group (groups defined at selected checkpoint, not dynamically tracked identities)</span><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Growth history for groups">`;
 for(let k=0;k<K;k++){
  const pts=prepared.traj.frames.map((f,i)=>`${left+(w-left-12)*f.step/Math.max(1,prepared.traj.steps)},${h-10-(h-22)*series[k][i]/max}`).join(' ');
  svg+=`<polyline fill="none" stroke="${color(k)}" stroke-width="1.9" opacity=".84" points="${pts}"/>`;
 }
 svg+='</svg>';$('evolution').innerHTML=svg;
}
function predictedPlot(){if(!committed)return;
 const kind=$('predictor').value, all=committed.predictions[kind], ids=committed.probeIds;
 const pred=ids.map(e=>all[e]),trueValues=revealed?.actual;
 const max=Math.max(1e-8,...pred,...(trueValues||[]))*1.1;
 const width=650,height=245,pad=35;
 let svg=`<div class="v3-plot-title">${kind}: ${ids.length} locked predicted single-edge cuts ${trueValues?'· white dots = held-out measured effects':'· outcomes not revealed'}</div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Predicted and measured resistance changes for selected edges">`;
 svg+=`<line x1="${pad}" y1="214" x2="640" y2="214" stroke="#6f8a8f"/>`;
 ids.forEach((e,i)=>{const x=pad+((width-pad-15)*i/(ids.length-1)),hp=pred[i]/max*180;
  svg+=`<line x1="${x}" y1="214" x2="${x}" y2="${214-hp}" stroke="${color(committed.groupings.Functional[e])}" stroke-width="3.2" opacity=".8"><title>Edge ${e}: predicted ${fmt(pred[i],5)}</title></line>`;
  if(trueValues)svg+=`<circle cx="${x}" cy="${214-trueValues[i]/max*180}" r="3.1" fill="#fff" stroke="#193440" stroke-width="1.2"><title>Edge ${e}: observed ${fmt(trueValues[i],5)}</title></circle>`;
 });
 svg+=`<text x="5" y="15" font-size="11" fill="#b3c8cf">ΔR ≤ ${fmt(max,3)}</text><text x="510" y="239" font-size="11" fill="#b3c8cf">Edge interventions →</text></svg>`;
 $('plot').innerHTML=svg;
}
$('task').addEventListener('change',clear);$('arm').addEventListener('change',clear);$('steps').addEventListener('change',clear);
$('coloring').addEventListener('change',()=>{selectedGroup=-1;draw();highlightCards();});
$('predictor').addEventListener('change',predictedPlot);
$('prepare').addEventListener('click',()=>{
 clear();$('prepare').disabled=true;$('status').textContent='Growing a network, selecting by validation, and measuring per-edge TRAIN interventions…';
 setTimeout(()=>{try{
  const task=$('task').value,arm=$('arm').value,steps=Number($('steps').value);
  const traj=trajectory(task,arm,{steps,checkpoints:[0,10,20,40,80,160,320]}),chosen=selectCheckpoint(traj);
  prepared={task,arm,steps,traj,selectedStep:chosen.step,h:Float64Array.from(chosen.h)};
  $('status').textContent=`Selected checkpoint ${chosen.step} using validation alone. Press Commit to discover 12 causal groups from TRAIN intervention profiles. No unseen source/sink cut has run.`;
  $('coloring').value='Functional';$('commit').disabled=false;draw();
 }catch(e){$('status').textContent='Error: '+e.message;console.error(e);}finally{$('prepare').disabled=false;}},40);
});
$('commit').addEventListener('click',()=>{
 if(!prepared)return;$('commit').disabled=true;$('status').textContent='Measuring all training edge interventions and locking 12 groups plus 8 damage predictors…';
 setTimeout(()=>{try{
  committed=commitInterpretation(prepared.h,prepared.task);$('reveal').disabled=false;
  $('status').textContent=`Committed ${committed.probeIds.length} prospective individual-edge predictions, 12 groups and 20 shuffled nulls. Network digest ${committed.hash}. No test cuts evaluated.`;
  $('commit-note').textContent=`Locked to state ${committed.hash} (noncryptographic). ${committed.trainPairCount} training probes × ${committed.edges} edges; no held-out terminal pairs or spatial labels entered functional clustering.`;
  groupCards();draw();plotDevelopment();predictedPlot();
 }catch(e){$('status').textContent='Error: '+e.message;console.error(e);$('commit').disabled=false;}},40);
});
$('reveal').addEventListener('click',()=>{
 if(!committed)return;$('reveal').disabled=true;$('status').textContent='Evaluating frozen predictions against unseen individual edge cuts…';
 setTimeout(()=>{try{
  revealed=revealInterpretation(prepared.h,committed);$('revealed').hidden=false;$('export').disabled=false;
  const scores=revealed.scores, sorted=Object.entries(scores).sort((a,b)=>a[1].rmse-b[1].rmse);
  const best=sorted[0],functional=scores['Functional groups'];
  $('verdict').className='v3-summary '+(best[0]==='Functional groups'?'':'warn');
  $('verdict').textContent=best[0]==='Functional groups'?'The discovered functional groups had the lowest held-out damage RMSE in this trial.':'The functional groups did NOT win: '+best[0]+' achieved the lowest held-out prediction RMSE. This is a useful negative control, not a failed interface.';
  $('scores').innerHTML=sorted.map(([name,s])=>`<tr class="${name==='Functional groups'?'v3-highlight-row':''}"><td>${name}</td><td>${fmt(s.pearson,3)}</td><td>${fmt(s.rmse,5)}</td><td>${s.top5}/5</td></tr>`).join('');
  const coh=Object.entries(revealed.coherence),max=Math.max(1e-6,...coh.map(x=>x[1]));
  $('coherence').innerHTML='<div class="v3-mini-bars">'+coh.map(([name,val],i)=>`<div><b>${name}</b><span style="width:${Math.max(1,Math.round(val/max*100))}%;background:${color(i)}"></span><small>${fmt(val,4)}</small></div>`).join('')+'</div>';
  const n=revealed.randomSummary;
  $('null-stats').innerHTML=`<strong>${n.randomBeatingFunctional}/${n.n} shuffled explanations</strong> had lower held-out RMSE than discovered functional groups.<p>Median random RMSE ${fmt(n.medianRmse,5)}; range ${fmt(n.minRmse,5)}–${fmt(n.maxRmse,5)}. These 20 groups were generated before revealing outcomes.</p>`;
  $('status').textContent=`Revealed ${revealed.probeIds.length} unseen single-edge cuts across ${revealed.testCount} disjoint terminal demands. Training parameters and predictions frozen; nothing was refit.`;
  predictedPlot();draw();
 }catch(e){$('status').textContent='Error: '+e.message;console.error(e);$('reveal').disabled=false;}},40);
});
$('export').addEventListener('click',()=>{
 if(!committed||!revealed)return;const receipt={protocol:'gate3 causal groups',task:prepared.task,arm:prepared.arm,step:prepared.selectedStep,
 stateDigest:committed.hash,trainingProfiles:committed.trainingProfiles,clusterDefinitions:committed.clusters,
 partitions:committed.groupings,committedPredictions:committed.predictions,shuffledNullPredictions:committed.shuffledNullPredictions,
 testedEdgeIds:revealed.probeIds,actualDamages:revealed.actual,scores:revealed.scores,coherence:revealed.coherence,randomNull:revealed.randomSummary,limitations:revealed.limitations};
 const url=URL.createObjectURL(new Blob([JSON.stringify(receipt,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`growth-microscope-gate3-${prepared.task}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1200);
});
