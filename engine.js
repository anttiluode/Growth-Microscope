/** Growth Microscope: deterministic passive conductance growth + causal intervention lab.
 * All dynamics are mathematical toy models; there is no neuron birth or biological growth.
 */
export const W = 11, H = 9, N = W * H;
export const xy = i => [i % W, Math.floor(i / W)];
export const id = (x, y) => y * W + x;
export const TASKS = ['bridge', 'fork', 'cross'];
const PAIRS = {
  bridge: {
    train: [[[0,4],[10,4]], [[0,3],[10,4]], [[0,4],[10,5]]],
    held: [[[0,5],[10,3]], [[1,4],[9,4]], [[0,2],[10,5]]]
  },
  fork: {
    train: [[[0,4],[10,1]], [[0,4],[10,7]], [[0,4],[10,4]]],
    held: [[[1,4],[9,2]], [[0,3],[10,6]], [[0,5],[9,4]]]
  },
  cross: {
    train: [[[0,1],[10,7]], [[0,7],[10,1]], [[0,4],[10,4]]],
    held: [[[0,2],[10,8]], [[1,7],[9,0]], [[0,5],[9,3]]]
  }
};
export const pairSets = task => {
  if (!PAIRS[task]) throw Error('unknown need');
  const conv = p => p.map(([x,y])=>id(x,y));
  return {train: PAIRS[task].train.map(conv), held: PAIRS[task].held.map(conv)};
};
export function makeGraph() {
  const edges = [];
  for (let y=0;y<H;y++) for(let x=0;x<W;x++) {
    if (x+1<W) edges.push(edge(id(x,y),id(x+1,y),1));
    if (y+1<H) edges.push(edge(id(x,y),id(x,y+1),1));
  }
  return {edges, n:N};
}
function edge(a,b,len) {
  const [xa,ya]=xy(a), [xb,yb]=xy(b), mx=(xa+xb)*.5,my=(ya+yb)*.5;
  const bx=Math.min(3,Math.floor(mx*4/W)),by=Math.min(2,Math.floor(my*3/H));
  return {a,b,len,region:by*4+bx,mx,my};
}
export const GRAPH=makeGraph();
export const REGIONS=Array.from({length:12},(_,i)=>({id:i, name:`${['Top','Middle','Bottom'][Math.floor(i/4)]} ${i%4+1}`}));
export function fresh(material=.065) {return Float64Array.from({length:GRAPH.edges.length},()=>material);}
export function conductances(h, base=.06) { return Float64Array.from(h,v=>base+v); }
function dot(a,b) {let v=0;for(let i=0;i<a.length;i++)v+=a[i]*b[i];return v;}
/** Anchored graph Laplacian, diagonally preconditioned CG. */
export function solve(g, source, sink, {tol=1e-10, maxIter=1200}={}) {
  if(source===sink)throw Error('source and sink must differ');
  if (g.length!==GRAPH.edges.length)throw Error('bad conductance count');
  const b=new Float64Array(N); b[source]=1;b[sink]=-1;
  const anchor=N-1; b[anchor]=0; // anchor is coordinate choice, not a current sink
  // If injection or extraction is at anchor, the reduced RHS is still correct.
  const diag=new Float64Array(N);
  for(let e=0;e<g.length;e++) {const {a,b}=GRAPH.edges[e],w=g[e];if(!(w>0))throw Error('non-positive conductance');diag[a]+=w;diag[b]+=w;}
  const product=(v,out)=>{
    out.fill(0);
    for(let e=0;e<g.length;e++) {const {a,b}=GRAPH.edges[e],c=g[e]*(v[a]-v[b]);out[a]+=c;out[b]-=c;}
    out[anchor]=0;
  };
  let v=new Float64Array(N), r=b.slice(), z=new Float64Array(N),p=new Float64Array(N),ap=new Float64Array(N);
  const norm=Math.sqrt(dot(r,r));
  for(let i=0;i<N;i++)z[i]=i===anchor?0:r[i]/diag[i];
  p.set(z);let rz=dot(r,z),iters=0, residual=norm;
  for(;iters<maxIter && residual>tol*norm;iters++){
    product(p,ap);const denom=dot(p,ap);
    if(!(denom>0))throw Error('Laplacian CG breakdown');
    const alpha=rz/denom;
    for(let i=0;i<N;i++){v[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
    r[anchor]=0;
    residual=Math.sqrt(dot(r,r));if(residual<=tol*norm)break;
    for(let i=0;i<N;i++)z[i]=i===anchor?0:r[i]/diag[i];
    const next=dot(r,z), beta=next/rz;rz=next;
    for(let i=0;i<N;i++)p[i]=z[i]+beta*p[i];
  }
  if(residual>1e-6*norm)throw Error(`CG convergence failure: ${residual}`);
  return {v, resistance:v[source]-v[sink], iterations:iters+1,residual};
}
/** Mean effective resistance and exact conductance derivative. */
export function objective(h,pairs) {
  const g=conductances(h), grad=new Float64Array(h.length), flows=new Float64Array(h.length);
  let loss=0;
  for(const [source,sink] of pairs){
    const {v,resistance}=solve(g,source,sink);loss+=resistance/pairs.length;
    for(let e=0;e<h.length;e++){
      const {a,b}=GRAPH.edges[e], dv=v[a]-v[b];
      grad[e]-=dv*dv/pairs.length;
      flows[e]+=Math.abs(g[e]*dv)/pairs.length;
    }
  }
  return {loss,grad,flows};
}
/** Euclidean project onto 0<=h<=cap and sum len*h<=budget; bisection finds dual. */
export function project(candidate,budget=24,cap=2) {
  if (!(budget>0 && cap>0))throw Error('positive budget and cap required');
  const out=Float64Array.from(candidate,v=>Math.max(0,Math.min(cap,v)));
  const mass=arr=>arr.reduce((s,v,i)=>s+v*GRAPH.edges[i].len,0);
  if(mass(out)<=budget)return out;
  let low=0,high=1;
  const at=l=>Float64Array.from(candidate,(v,i)=>Math.max(0,Math.min(cap,v-l*GRAPH.edges[i].len)));
  while(mass(at(high))>budget)high*=2;
  for(let k=0;k<55;k++){const mid=(low+high)/2;if(mass(at(mid))>budget)low=mid;else high=mid;}
  return at(high);
}
export const material = h => h.reduce((s,v,i)=>s+v*GRAPH.edges[i].len,0);
export function growStep(h,pairs,{budget=24,rate=.14}={}) {
  const data=objective(h,pairs), g=data.grad;
  // normalized descent, capped step; zeros can regrow if there is causal benefit
  let scale=0; for(let i=0;i<g.length;i++)scale=Math.max(scale,Math.abs(g[i]));
  const next=Float64Array.from(h,(v,i)=>v-rate*g[i]/Math.max(scale,1e-12));
  return {h:project(next,budget),data};
}
export function grow(task,{steps=320,budget=24,rate=.14,snapshots=[0,20,60,140,320]}={}) {
  const {train,held}=pairSets(task);
  let h=fresh(); const history=[],frames=[]; const snaps=new Set([...snapshots,steps]);
  for(let t=0;t<=steps;t++){
    if(t%5===0 || snaps.has(t)){
      const tr=objective(h,train).loss,te=objective(h,held).loss;
      history.push({step:t,train:tr,held:te,material:material(h)});
    }
    if(snaps.has(t))frames.push({step:t,h:Array.from(h)});
    if(t<steps)h=growStep(h,train,{budget,rate}).h;
  }
  return {task,h,history,frames,train,held,budget,steps};
}
/** Finite perturbation: weaken every grown edge in this region; fixed scaffold remains. */
export function ablate(h,region,fraction=.5) {
  if(!(fraction>=0 && fraction<=1))throw Error('invalid fraction');
  return Float64Array.from(h,(v,i)=>GRAPH.edges[i].region===region?v*(1-fraction):v);
}
export function causalMap(h,pairs,fraction=.5) {
  const base=objective(h,pairs).loss;
  return Float64Array.from(REGIONS,region=>objective(ablate(h,region.id,fraction),pairs).loss-base);
}
export function groupSum(values) {
  const r=new Float64Array(REGIONS.length);
  for(let i=0;i<values.length;i++)r[GRAPH.edges[i].region]+=values[i];
  return r;
}
export function fitGain(predictor,truth){ // scalar fit on TRAIN regions only
  const denom=dot(predictor,predictor);
  return denom>1e-20?dot(predictor,truth)/denom:0;
}
export function pearson(x,y){
  const mx=x.reduce((a,b)=>a+b,0)/x.length,my=y.reduce((a,b)=>a+b,0)/y.length;
  let xx=0,yy=0,xy=0;for(let i=0;i<x.length;i++){const a=x[i]-mx,b=y[i]-my;xx+=a*a;yy+=b*b;xy+=a*b;}
  return xx>1e-20 && yy>1e-20?xy/Math.sqrt(xx*yy):0;
}
export function rmse(x,y){return Math.sqrt(x.reduce((s,v,i)=>s+(v-y[i])**2,0)/x.length);}
export function topK(x,k=3){return Array.from(x,(_,i)=>i).sort((a,b)=>x[b]-x[a]).slice(0,k);}
export function compare(h,pairsTrain,pairsHeld,{fraction=.5}={}){
  const train=objective(h,pairsTrain),held=objective(h,pairsHeld);
  const actualTrain=causalMap(h,pairsTrain,fraction), actual=causalMap(h,pairsHeld,fraction);
  const predictions={
    // Training functional signature transferred directly without looking at test sensitivities.
    'Train causal fingerprint': actualTrain.slice(),
    // Simple structural baseline, calibrated on training interventions.
    'Weight only':groupSum(h),
    // Probe-dependent controls use ONE held-out intact-state field solve (no ablation).
    'Held-out activity': groupSum(held.flows),
    'Held-out adjoint sensitivity': groupSum(Float64Array.from(held.grad,(v,i)=>-fraction*h[i]*v))
  };
  const trainPredict={
    'Weight only':groupSum(h),
    'Held-out activity':groupSum(train.flows),
    'Held-out adjoint sensitivity':groupSum(Float64Array.from(train.grad,(v,i)=>-fraction*h[i]*v))
  };
  const scores={};
  for(const [label,p] of Object.entries(predictions)){
    const gain=label==='Train causal fingerprint'?1:fitGain(trainPredict[label],actualTrain);
    for(let i=0;i<p.length;i++)p[i]*=gain;
    scores[label]={r:pearson(p,actual),rmse:rmse(p,actual),top3_overlap:topK(p).filter(x=>topK(actual).includes(x)).length,gain};
  }
  return {actual:Array.from(actual),actualTrain:Array.from(actualTrain), predictions:Object.fromEntries(Object.entries(predictions).map(([k,v])=>[k,Array.from(v)])),scores,trainLoss:train.loss,heldLoss:held.loss,fraction};
}
export function fingerprint(h, pairClasses) {
  const matrix=pairClasses.map(pairs=>Array.from(causalMap(h,pairs)));
  return REGIONS.map((region,i)=>{
    const profile=matrix.map(row=>row[i]); const max=Math.max(...profile),sum=profile.reduce((a,b)=>a+b,0);
    return {region:region.id,profile,role:max<=1e-9?'Dormant':Math.min(...profile)>max*.6?'Shared':`Need ${profile.indexOf(max)+1}`,total:sum};
  });
}
export function runAll({steps=320}={}){
  return TASKS.map(task=>{
    const g=grow(task,{steps});return {task,steps,train:g.history.at(-1).train,held:g.history.at(-1).held,comparison:compare(g.h,g.train,g.held),material:material(g.h),edges:g.h.length};
  });
}
