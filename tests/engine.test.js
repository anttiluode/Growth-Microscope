import test from 'node:test';
import assert from 'node:assert/strict';
import {W,H,N,GRAPH,REGIONS,TASKS,pairSets,fresh,conductances,solve,objective,project,material,growStep,grow,ablate,causalMap,fitGain,compare,fingerprint,topK} from '../engine.js';
const near=(a,b,tol=1e-6)=>assert.ok(Math.abs(a-b)<tol,`${a} differs from ${b}`);
test('undirected connected graph and consistent regions',()=>{
  assert.equal(N, W*H); assert.equal(GRAPH.edges.length,(W-1)*H+(H-1)*W);
  for(const e of GRAPH.edges){assert.ok(e.a>=0 && e.b<N && e.region>=0 && e.region<12);assert.equal(e.len,1);}
});
test('distinct train/test terminal configurations',()=>{
  for(const task of TASKS){const p=pairSets(task);assert.equal(p.train.length,3);assert.equal(p.held.length,3);
    const names=arr=>new Set(arr.map(pair=>pair.join(':')));
    for(const row of names(p.held))assert.ok(!names(p.train).has(row),row);
  }
});
test('Kirchhoff energy = resistance and reciprocity',()=>{
  const g=conductances(fresh()), a=12,b=68;
  const r=solve(g,a,b); near(r.resistance,solve(g,b,a).resistance,1e-7);
  const E=GRAPH.edges.reduce((s,e,i)=>s+g[i]*(r.v[e.a]-r.v[e.b])**2,0);
  near(E,r.resistance,2e-7);assert.ok(r.residual<1e-6);
});
test('exact local gradient matches finite difference',()=>{
  const h=fresh(),p=pairSets('bridge').train,base=objective(h,p),d=1e-5;
  for(const idx of [0,6,22,99,GRAPH.edges.length-1]){
    const other=h.slice();other[idx]+=d;near((objective(other,p).loss-base.loss)/d,base.grad[idx],1.5e-3);
  }
});
test('material projection respects cap and weighted budget',()=>{
  const x=new Float64Array(GRAPH.edges.length).fill(5);
  const p=project(x,22,2);near(material(p),22,1e-8);assert.ok(p.every(v=>v>=0 && v<=2));
  const inside=project(fresh(),50);near(inside[0],.065,1e-12);
});
test('growth uses fixed budget and improves its training task',()=>{
  const p=pairSets('fork').train;let h=fresh(),before=objective(h,p).loss;
  for(let k=0;k<30;k++)h=growStep(h,p).h;
  assert.ok(material(h)<=24+1e-6);
  assert.ok(objective(h,p).loss<before);
});
test('group causal effect is nonnegative for passive edge weakening',()=>{
  const g=grow('bridge',{steps:40});const c=causalMap(g.h,g.train,.5);
  assert.ok(c.every(x=>x>=-1e-6));
  const cut=ablate(g.h,1,.5);
  GRAPH.edges.forEach((e,i)=>near(cut[i],e.region===1?.5*g.h[i]:g.h[i],1e-12));
});
test('infinitesimal gradient correctly predicts a small intervention',()=>{
  const g=grow('cross',{steps:15}); const data=objective(g.h,g.held),region=4,fraction=0.001;
  const pred=GRAPH.edges.reduce((s,e,i)=>s+(e.region===region?-data.grad[i]*g.h[i]*fraction:0),0);
  const actual=objective(ablate(g.h,region,fraction),g.held).loss-data.loss;
  near(pred,actual,1e-4);
});
test('causal prediction uses only train ablation, held-out probe baselines are marked',()=>{
  const run=grow('fork',{steps:60});const result=compare(run.h,run.train,run.held);
  assert.equal(result.actual.length,12);assert.equal(Object.keys(result.scores).length,4);
  for(let i=0;i<12;i++)near(result.predictions['Train causal fingerprint'][i],result.actualTrain[i],1e-12);
  for(const score of Object.values(result.scores)){assert.ok(Number.isFinite(score.r));assert.ok(score.top3_overlap>=0&&score.top3_overlap<=3);}
});
test('fingerprints, deterministic checkpoints and export data',()=>{
  const a=grow('cross',{steps:40,snapshots:[0,20,40]});
  const b=grow('cross',{steps:40,snapshots:[0,20,40]});
  assert.deepEqual(Array.from(a.h),Array.from(b.h));assert.deepEqual(a.frames.map(f=>f.step),[0,20,40]);
  const fp=fingerprint(a.h,a.train.map(p=>[p]));assert.equal(fp.length,REGIONS.length);
  assert.ok(fp.every(x=>x.profile.length===3 && typeof x.role==='string'));
});
test('no-input controls and errors',()=>{
  assert.equal(fitGain(new Float64Array(12),new Float64Array(12)),0);
  assert.deepEqual(topK([0,4,3],2),[1,2]);
  assert.throws(()=>pairSets('unknown'));assert.throws(()=>solve(conductances(fresh()),4,4));
  assert.throws(()=>ablate(fresh(),1,2));assert.throws(()=>project(fresh(),-1));
});
