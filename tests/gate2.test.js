import test from 'node:test';
import assert from 'node:assert/strict';
import {GRAPH,pairSets,material,objective,REGIONS,ablate} from '../engine.js';
import {CHECKPOINTS,needSplits,sweepPairs,objectivePairs,trajectory,selectCheckpoint,
 evaluateFinal,commitDamagePrediction,revealDamagePrediction,equalMaterialUniform,audit} from '../gate2.js';
for(const need of ['bridge','fork','cross']){
 test(`${need}: all four prescribed splits plus broad sweep are disjoint`,()=>{
  const p=needSplits(need),sweep=sweepPairs(need);
  const sets=[p.train,p.augmentation,p.validation,p.test,sweep];
  for(let i=0;i<sets.length;i++){
    const keys=new Set(sets[i].map(x=>x.join(',')));
    assert.equal(keys.size,sets[i].length);
    for(let j=0;j<i;j++)for(const item of sets[j])assert.ok(!keys.has(item.join(',')));
  }
  assert.equal(sweep.length,24);
 });
 test(`${need}: specialist and diverse use exactly three field solves per growth step`,()=>{
  for(let step=0;step<35;step++){
    assert.deepEqual(objectivePairs(need,'specialist',step),needSplits(need).train);
    assert.equal(objectivePairs(need,'diverse',step).length,3);
  }
  const unique=new Set(Array.from({length:35},(_,t)=>objectivePairs(need,'diverse',t)).flat().map(x=>x.join(',')));
  assert.equal(unique.size,7);
 });
}
test('checkpoint is selected strictly on validation and not final test',()=>{
 const t=trajectory('bridge','specialist',{steps:40,checkpoints:[0,10,20,40]});
 const chosen=selectCheckpoint(t),v=Math.min(...t.frames.map(f=>f.validation));
 assert.ok(Math.abs(chosen.validation-v)<1e-12);
 assert.ok(t.frames.every(f=>!('sweep'in f)&&!('test'in f)));
 const evals=evaluateFinal(t);
 assert.equal(evals.selected.step,chosen.step);
});
test('predictions are locked to the exact state before any held-out ablations',()=>{
 const t=trajectory('fork','diverse',{steps:40,checkpoints:[10,40]});
 const h=Float64Array.from(selectCheckpoint(t).h);
 const commit=commitDamagePrediction(h,'fork');
 assert.equal(Object.keys(commit.predictions).length,4);
 assert.equal(commit.trainingHarm.length,REGIONS.length);
 assert.ok(!('actual'in commit)&&!('testResistance'in commit));
 const result=revealDamagePrediction(h,commit);
 assert.equal(result.pairs,24);assert.equal(result.actual.length,12);
 for(const row of Object.values(result.scores))assert.ok(Number.isFinite(row.pearson));
 const modified=h.slice();modified[5]+=.001;
 assert.throws(()=>revealDamagePrediction(modified,commit),/changed state/);
});
test('uniform control has identical grown material and positive conductance',()=>{
 const t=trajectory('cross','specialist',{steps:40});
 const h=Float64Array.from(selectCheckpoint(t).h),uniform=equalMaterialUniform(h);
 assert.ok(Math.abs(material(uniform)-material(h))<1e-9);
 assert.ok(uniform.every(x=>x>=0));
});
test('audit is exactly reproducible with fixed demands',()=>{
 const one=audit('bridge',{steps:40,checkpoints:[0,10,20,40]});
 const two=audit('bridge',{steps:40,checkpoints:[0,10,20,40]});
 assert.deepEqual(one,two);
});
