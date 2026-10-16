/** Gate 3 — discover functional edge groups from TRAINING interventions, then
 * prospectively predict damage for individual edges on unseen terminal demands.
 * This is a deterministic Kirchhoff toy lab, NOT neuron interpretability.
 * All competing groupings use the same training ablation observations.
 */
import {GRAPH,REGIONS,xy,conductances,solve,objective,pearson,rmse,topK} from './engine.js';
import {needSplits,sweepPairs,trajectory,selectCheckpoint} from './gate2.js';

export const K=12, FRACTION=.5;
const EDGES=GRAPH.edges.length;
const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
const fit=(x,y)=>{let n=0,d=0;for(let i=0;i<x.length;i++){n+=x[i]*y[i];d+=x[i]*x[i];}return d>1e-24?n/d:0;};

export function cutEdge(h,e,fraction=FRACTION){
 if(!Number.isInteger(e)||e<0||e>=EDGES||fraction<0||fraction>1)throw Error('invalid edge cut');
 const out=Float64Array.from(h);out[e]*=(1-fraction);return out;
}
/** Finite-difference causal effect of each edge on each TRAIN stimulus.
 * No rectangular region IDs, node positions, semantic labels or TEST inputs here.
 */
export function trainEdgeInterventions(h,trainPairs,{fraction=FRACTION}={}){
 const baselines=trainPairs.map(pair=>objective(h,[pair]).loss);
 const profiles=Array.from({length:EDGES},()=>new Array(trainPairs.length));
 for(let e=0;e<EDGES;e++){
  if(h[e]===0){profiles[e].fill(0);continue;}
  const cut=cutEdge(h,e,fraction);
  for(let j=0;j<trainPairs.length;j++)profiles[e][j]=Math.max(0,objective(cut,[trainPairs[j]]).loss-baselines[j]);
 }
 return profiles;
}
export function trainSignals(h,trainPairs){
 const g=conductances(h), flows=new Float64Array(EDGES),gradient=new Float64Array(EDGES);
 for(const [source,sink] of trainPairs){
  const {v}=solve(g,source,sink);
  for(let e=0;e<EDGES;e++){
   const {a,b}=GRAPH.edges[e],dv=v[a]-v[b];
   flows[e]+=Math.abs(g[e]*dv)/trainPairs.length;
   gradient[e]+=FRACTION*h[e]*dv*dv/trainPairs.length;
  }
 }
 return {flows,gradient};
}
function seedDraw(seed){let s=seed>>>0;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return s>>>0;};}
/** Balanced ranks avoid trivially singleton groups for material/flow controls. */
export function rankGroups(values,k=K){
 const sorted=Array.from({length:values.length},(_,i)=>i).sort((a,b)=>values[a]-values[b]||a-b);
 const labels=new Array(values.length);
 sorted.forEach((idx,r)=>{labels[idx]=Math.min(k-1,Math.floor(r*k/values.length));});return labels;
}
export function randomGroups(n=EDGES,k=K,seed=41){
 const indices=Array.from({length:n},(_,i)=>i),rng=seedDraw(seed);
 for(let i=n-1;i>0;i--){const j=rng()%(i+1);[indices[i],indices[j]]=[indices[j],indices[i]];}
 const group=Array(n);
 indices.forEach((e,i)=>{group[e]=Math.min(k-1,Math.floor(i*k/n));});return group;
}
/** Standardized causal response vectors, seeded deterministic k-means, no spatial input.
 * Initializes by farthest-point selection; empty clusters are repaired by splitting
 * the largest cluster at its farthest sample. Returned membership is an explanation
 * candidate, not proof of a useful mechanism.
 */
export function functionalGroups(profiles,k=K,{iterations=35}={}){
 if(profiles.length<k||!profiles.length)throw Error('not enough profiles');
 const dims=profiles[0].length;
 if(!dims||profiles.some(row=>row.length!==dims||row.some(x=>!Number.isFinite(x))))throw Error('bad profile');
 const scale=Array.from({length:dims},(_,j)=>Math.max(1e-11,Math.sqrt(avg(profiles.map(row=>row[j]*row[j])))));
 // log compress huge local effects; preserve probe-to-probe shape and magnitude.
 const features=profiles.map(row=>row.map((x,j)=>Math.log1p(Math.max(0,x)/scale[j])));
 const dist=(a,b)=>a.reduce((s,v,j)=>s+(v-b[j])**2,0);
 const chosen=[features.reduce((best,v,i)=>v.reduce((s,x)=>s+x*x,0)>features[best].reduce((s,x)=>s+x*x,0)?i:best,0)];
 while(chosen.length<k){let best=-1,score=-1;
  for(let i=0;i<features.length;i++)if(!chosen.includes(i)){
   const nearest=Math.min(...chosen.map(j=>dist(features[i],features[j])));
   if(nearest>score+1e-14){best=i;score=nearest;}
  }
  chosen.push(best);
 }
 let centroids=chosen.map(i=>features[i].slice()),labels=new Array(features.length).fill(-1);
 for(let iter=0;iter<iterations;iter++){
  // Balanced assignment: every explanatory group must represent a similar
  // number of edges. Otherwise k-means creates tiny causal outlier labels and
  // dumps 100+ dormant edges into a meaningless catch-all group.
  const capacity=Array.from({length:k},(_,i)=>Math.floor(features.length/k)+(i<features.length%k?1:0));
  const order=features.map((row,i)=>{
   const ds=centroids.map(c=>dist(row,c)).sort((a,b)=>a-b);
   return {i,priority:ds[1]-ds[0]};
  }).sort((a,b)=>b.priority-a.priority||a.i-b.i);
  const updated=new Array(features.length);
  for(const {i} of order){
   let best=-1,score=Infinity;
   for(let c=0;c<k;c++)if(capacity[c]>0){
    const d=dist(features[i],centroids[c]);if(d<score){score=d;best=c;}
   }
   if(best<0)throw Error('balanced clustering failed');
   updated[i]=best;capacity[best]--;
  }
  const counts=Array(k).fill(0);updated.forEach(c=>counts[c]++);
  const sums=Array.from({length:k},()=>new Array(dims).fill(0));
  updated.forEach((c,i)=>features[i].forEach((x,j)=>sums[c][j]+=x));
  centroids=sums.map((row,i)=>row.map(x=>x/counts[i]));
  if(updated.every((c,i)=>c===labels[i]))break;
  labels=updated;
 }
 return labels;
}
export function allGroupings(h,profiles,signals,{k=K,seed=41}={}){
 return {
  'Functional':functionalGroups(profiles,k),
  'Material bins':rankGroups(h,k),
  'Flow bins':rankGroups(signals.flows,k),
  'Rectangles':GRAPH.edges.map(e=>e.region),
  'Shuffled':randomGroups(EDGES,k,seed)
 };
}
/** Same training edge effects for ALL groupings, no held-out measurements. */
export function predictFromGroups(h,profiles,labels){
 if(labels.length!==EDGES)throw Error('group length mismatch');
 const k=Math.max(...labels)+1, totalH=new Float64Array(k),totalEffect=new Float64Array(k);
 for(let e=0;e<EDGES;e++){
  const group=labels[e];totalH[group]+=h[e];totalEffect[group]+=avg(profiles[e]);
 }
 return Array.from(h,(v,e)=>v*(totalEffect[labels[e]]/Math.max(totalH[labels[e]],1e-12)));
}
export function trainingPredictions(h,profiles,signals,groupings){
 const truth=profiles.map(row=>avg(row)), pred={};
 const unscaled={
  'Material only':Array.from(h),
  'Mean train flow':Array.from(signals.flows,(v,i)=>v*(h[i]>.0000000001?1:0)),
  'Train derivative':Array.from(signals.gradient)
 };
 for(const [label,features] of Object.entries(unscaled)){
  const gain=fit(features,truth);pred[label]=features.map(v=>v*gain);
 }
 for(const [label,labels] of Object.entries(groupings)){
  pred[label+' groups']=predictFromGroups(h,profiles,labels);
 }
 return pred;
}
/** Probe subset is picked from trained weights plus deterministic edge IDs only.
 * Identical edge IDs for every explanation on a given frozen medium.
 * This is not unbiased sampling over edges: report both active and dormant cuts.
 */
export function probeEdges(h,{active=30,uniform=24}={}){
 const picked=new Set(Array.from({length:EDGES},(_,e)=>e)
  .sort((a,b)=>h[b]-h[a]||a-b).slice(0,active));
 const spacing=EDGES/uniform;
 for(let j=0;j<uniform;j++)picked.add(Math.min(EDGES-1,Math.floor((j+.5)*spacing)));
 return [...picked].sort((a,b)=>a-b);
}
export function stateHash(h){
 let hash=2166136261>>>0;
 for(const v of h)for(const c of Number(v).toPrecision(16))hash=Math.imul((hash^c.charCodeAt(0))>>>0,16777619)>>>0;
 return hash.toString(16).padStart(8,'0');
}
export function commitInterpretation(h,task,{fraction=FRACTION}={}){
 if(h.length!==EDGES)throw Error('incorrect conductance length');
 if(fraction!==FRACTION)throw Error('Gate 3 fixed half-strength intervention');
 const training=needSplits(task).train;
 const profiles=trainEdgeInterventions(h,training,{fraction});
 const signals=trainSignals(h,training);
 const groupings=allGroupings(h,profiles,signals);
 const predictions=trainingPredictions(h,profiles,signals,groupings);
 const probeIds=probeEdges(h);
 const shuffledNullPredictions=Array.from({length:20},(_,i)=>
  predictFromGroups(h,profiles,randomGroups(EDGES,K,41+i*13)));
 const clusters=Array.from({length:K},(_,k)=>{
  const edges=groupings.Functional.flatMap((g,e)=>g===k?[e]:[]);
  const signature=training.map((_,j)=>avg(edges.map(e=>profiles[e][j])));
  return {id:k,edges,signature,material:edges.reduce((s,e)=>s+h[e],0)};
 });
 return {protocol:'growth-microscope-gate3-v1',task,fraction,hash:stateHash(h),
  edges:EDGES,groups:K,trainPairCount:training.length,probeIds,
  groupings,predictions,shuffledNullPredictions,clusters,
  trainingEffects:profiles.map(row=>avg(row)),
  trainingProfiles:profiles,
  // All model outputs are committed before any test ablation calls.
  interpretation:'Unsupervised groups of causal training-edge signatures; no semantic or spatial labels in feature vectors.'};
}
export function revealInterpretation(h,commit,{testCount=12}={}){
 if(stateHash(h)!==commit.hash)throw Error('state changed after commitment');
 if(testCount!==12&&testCount!==24)throw Error('Only frozen 12/24 unseen-pair test sets');
 const held=sweepPairs(commit.task,24).slice(0,testCount);
 const baseline=objective(h,held).loss;
 const truth=commit.probeIds.map(e=>Math.max(0,objective(cutEdge(h,e,commit.fraction),held).loss-baseline));
 const outcomes={};for(const [name,all] of Object.entries(commit.predictions)){
  const predicted=commit.probeIds.map(e=>all[e]);
  const t=Math.min(5,truth.length), winner=topK(truth,t),guess=topK(predicted,t);
  outcomes[name]={pearson:pearson(predicted,truth),rmse:rmse(predicted,truth),top5:guess.filter(i=>winner.includes(i)).length};
 }
 const shuffledNull=commit.shuffledNullPredictions.map(all=>{
  const pred=commit.probeIds.map(e=>all[e]);
  return {pearson:pearson(pred,truth),rmse:rmse(pred,truth)};
 });
 const sortedNull=shuffledNull.map(x=>x.rmse).sort((a,b)=>a-b);
 const randomSummary={n:sortedNull.length,medianRmse:(sortedNull[9]+sortedNull[10])/2,
  minRmse:sortedNull[0],maxRmse:sortedNull.at(-1),
  randomBeatingFunctional:shuffledNull.filter(x=>x.rmse<outcomes['Functional groups'].rmse).length};
 // Descriptive coherence: within-group variance of actual intervention loss,
 // same standardized outcome and probe subset for every partition.
 const coherence={};
 for(const [name,labels] of Object.entries(commit.groupings)){
  const groupVals=new Map();commit.probeIds.forEach((e,i)=>{const k=labels[e];if(!groupVals.has(k))groupVals.set(k,[]);groupVals.get(k).push(truth[i]);});
  let total=0,within=0;for(const vals of groupVals.values()){
   const mean=avg(vals);for(const x of vals){total++;within+=(x-mean)**2;}
  }
  coherence[name]=Math.sqrt(within/Math.max(total,1));
 }
 return {testCount,actual:truth,baseline,probeIds:commit.probeIds.slice(),scores:outcomes,randomSummary,coherence,
  units:'Delta effective resistance for halving ONE recruited edge, fixed scaffold intact.',
  limitations:['Shared procedural 12/24 held-out source/sink pairs, no new task family',
   'Probe set contains high-material and fixed-spaced edges, not independent random samples',
   'Rectangular group partition allowed only as a control; functional features exclude positions',
   'Train derivative is given exact known physics and can be stronger than grouped explanations']};
}
export function runGate3(task='bridge',{steps=160,arm='diverse'}={}){
 const t=trajectory(task,arm,{steps,checkpoints:[0,10,20,40,80,160,320]});
 const f=selectCheckpoint(t),h=Float64Array.from(f.h),commit=commitInterpretation(h,task),reveal=revealInterpretation(h,commit);
 return {task,arm,steps,selectedStep:f.step,material:f.material,train:f.train,validation:f.validation,
  groupStats:commit.clusters.map(({id,edges,signature,material})=>({id,count:edges.length,signature,material})),
  score:reveal.scores,randomSummary:reveal.randomSummary,coherence:reveal.coherence,probeCount:reveal.probeIds.length,
  actual:reveal.actual,indices:reveal.probeIds};
}
