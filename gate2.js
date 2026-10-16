/** V2 prospective-need gate: independent validation and test sets, equal per-step
 * field-solve budgets, frozen checkpoint selection and blinded ablation forecasts.
 * All sets are synthetic and hand-designed. This is NOT neural interpretability.
 */
import {GRAPH,REGIONS,xy,id,TASKS,pairSets,fresh,growStep,objective,material,causalMap,groupSum,fitGain,pearson,rmse,topK} from './engine.js';
export const CHECKPOINTS=[0,10,20,40,80,160,320];
const pairs=p=>p.map(([[sx,sy],[tx,ty]])=>[id(sx,sy),id(tx,ty)]);
// The old 'held' splits become explicit model-selection VALIDATION only.
const TEST={
 bridge: [ [[0,6],[10,2]], [[0,1],[10,6]], [[1,3],[9,5]] ],
 fork:   [ [[0,2],[10,0]], [[1,5],[9,7]], [[0,6],[10,3]] ],
 cross:  [ [[0,0],[10,8]], [[1,8],[9,1]], [[0,6],[10,2]] ]
};
const EXTRA={
 bridge: [ [[0,4],[10,3]],[[0,5],[10,4]],[[1,3],[10,4]],[[0,4],[9,5]] ],
 fork:   [ [[0,4],[10,2]],[[0,4],[10,6]],[[0,3],[10,1]],[[0,5],[10,7]] ],
 cross:  [ [[0,2],[10,7]],[[0,6],[10,1]],[[0,1],[10,6]],[[0,7],[10,2]] ]
};
export function needSplits(task){
 const original=pairSets(task);
 const splits={train:original.train, augmentation:pairs(EXTRA[task]||[]), validation:original.held, test:pairs(TEST[task]||[])};
 if(!TASKS.includes(task))throw Error('unknown task');
 const sets=Object.entries(splits).map(([key,items])=>[key,new Set(items.map(x=>x.join(':')))]);
 for(let i=0;i<sets.length;i++)for(let j=0;j<i;j++)for(const k of sets[i][1]){
   if(sets[j][1].has(k))throw Error(`data leak ${sets[i][0]} vs ${sets[j][0]}: ${k}`);
 }
 return splits;
}
/** A reproducible, non-optimized and disjoint broad endpoint distribution.
 * Fixed pseudo-random draws are frozen in code, not tuned to the result.
 */
export function sweepPairs(task,count=24,seed=20261016){
 if(!TASKS.includes(task)||!Number.isInteger(count)||count<1)throw Error('invalid sweep');
 const p=needSplits(task);
 const excluded=new Set([...p.train,...p.augmentation,...p.validation,...p.test].map(q=>q.join(':')));
 let state=(seed + TASKS.indexOf(task)*104729)>>>0;
 const draw=n=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)%n;};
 const output=[];
 let attempts=0;
 while(output.length<count){
   if(++attempts>count*1000)throw Error('could not sample distinct test pairs');
   const left=draw(2),right=10-draw(2);
   let sy,ty;
   if(task==='bridge'){sy=1+draw(7);ty=1+draw(7);}
   else if(task==='fork'){sy=2+draw(5);ty=draw(9);}
   else {sy=draw(2)===0?draw(3):6+draw(3);ty=sy<4?6+draw(3):draw(3);}
   const pair=[id(left,sy),id(right,ty)],key=pair.join(':');
   if(!excluded.has(key)){excluded.add(key);output.push(pair);}
 }
 return output;
}
export function objectivePairs(task,arm,step){
 const p=needSplits(task);
 if(arm==='specialist')return p.train;
 if(arm!=='diverse')throw Error('arm must be specialist or diverse');
 const pool=[...p.train,...p.augmentation];
 // Exactly THREE source/sink solves per gradient update in both arms.
 // No validation or test pair enters optimization.
 return [0,1,2].map(j=>pool[(3*step+j)%pool.length]);
}
export function trajectory(task,arm,{steps=320,budget=24,rate=.14,checkpoints=CHECKPOINTS}={}){
 if(!Number.isInteger(steps)||steps<0)throw Error('steps must be a nonnegative integer');
 const p=needSplits(task), keep=new Set([...checkpoints.filter(x=>x<=steps),steps]);
 let h=fresh();const frames=[];
 for(let t=0;t<=steps;t++){
   if(keep.has(t))frames.push({step:t,h:Array.from(h),
     train:objective(h,p.train).loss,validation:objective(h,p.validation).loss,
     material:material(h)});
   if(t!==steps)h=growStep(h,objectivePairs(task,arm,t),{budget,rate}).h;
 }
 return {task,arm,frames,steps,budget,rate,pairsPerUpdate:3};
}
export function selectCheckpoint(traj){
 return traj.frames.reduce((best,cur)=>cur.validation<best.validation-1e-12?cur:best);
}
export function evaluateFinal(traj){
 const p=needSplits(traj.task), sweep=sweepPairs(traj.task), chosen=selectCheckpoint(traj);
 const final=traj.frames.find(f=>f.step===traj.steps);
 const evalState=f=>({step:f.step,train:f.train,validation:f.validation,test:objective(f.h,p.test).loss,sweep:objective(f.h,sweep).loss,material:f.material});
 return {selected:evalState(chosen),final:evalState(final)};
}
/** A model's prediction is committed using only h and TRAINING demands.
 * The 'test' split is used only by the reveal function (or by benchmark scoring).
 */
function stateDigest(h){
 // Integrity check for commit/reveal, not a cryptographic commitment.
 let hash=2166136261>>>0;
 for(const v of h)for(const c of v.toPrecision(15)){
   hash=Math.imul((hash^c.charCodeAt(0))>>>0,16777619)>>>0;
 }
 return hash.toString(16).padStart(8,'0');
}
export function commitDamagePrediction(h,task,{fraction=.5}={}){
 const p=needSplits(task), training=objective(h,p.train), observed=causalMap(h,p.train,fraction);
 const proposed={
  'Train causal':Array.from(observed),
  'Material':Array.from(groupSum(h)),
  'Train flow':Array.from(groupSum(training.flows)),
  'Train gradient':Array.from(groupSum(Float64Array.from(training.grad,(v,i)=>-fraction*h[i]*v)))
 };
 const predictors={
  'Material':proposed.Material,
  'Train flow':proposed['Train flow'],
  'Train gradient':proposed['Train gradient']
 };
 for(const [name,x] of Object.entries(predictors)){
   const gain=fitGain(x,observed);for(let i=0;i<x.length;i++)x[i]*=gain;
 }
 return {task,fraction,steps:null,stateDigest:stateDigest(h),predictions:proposed,trainingHarm:Array.from(observed)};
}
export function revealDamagePrediction(h,committed,{split='sweep'}={}){
 if(committed.stateDigest!==stateDigest(h))throw Error('changed state since prediction commitment');
 if(!['sweep','stress'].includes(split))throw Error('invalid test split');
 const p=needSplits(committed.task), cases=split==='sweep'?sweepPairs(committed.task):p.test;
 const actual=Array.from(causalMap(h,cases,committed.fraction));
 const scores={};
 for(const [name,pred] of Object.entries(committed.predictions)){
   scores[name]={pearson:pearson(pred,actual),rmse:rmse(pred,actual),top3:topK(pred).filter(x=>topK(actual).includes(x)).length};
 }
 return {actual,scores,testResistance:objective(h,cases).loss,split,pairs:cases.length};
}
export function equalMaterialUniform(h){
 const m=material(h), num=GRAPH.edges.length;
 return Float64Array.from({length:num},()=>m/num);
}
export function audit(task,options={}){
 const specialist=trajectory(task,'specialist',options),diverse=trajectory(task,'diverse',options);
 const ps=needSplits(task),sweep=sweepPairs(task);
 const a=evaluateFinal(specialist),b=evaluateFinal(diverse);
 // Selection is frozen by validation only. Test readout happens below.
 const uniform={};for(const [name,traj] of [['specialist',specialist],['diverse',diverse]]){
   const h=Float64Array.from(selectCheckpoint(traj).h);
   uniform[name]={stress:objective(equalMaterialUniform(h),ps.test).loss,sweep:objective(equalMaterialUniform(h),sweep).loss};
 }
 const testDamage={};
 for(const [name,traj] of [['specialist',specialist],['diverse',diverse]]){
   const h=Float64Array.from(selectCheckpoint(traj).h),commit=commitDamagePrediction(h,task);
   testDamage[name]=revealDamagePrediction(h,commit).scores;
 }
 return {task,protocol:'growth-microscope-gate2-v1',steps:options.steps??320,
   gateRules:['Train: exact original pairs; diverse: cycling 3 of 7 train/augmentation pairs, identical 3 field solves per update',
    'Checkpoint picked by validation only, BEFORE test scoring','Independent hidden test pairs never used for growth or selection',
    'Causal harm predictors and gains use TRAIN pairs only; test ablations occur after commit'],
   specialist:a,diverse:b,equalMaterialUniformTest:uniform,damagePrediction:testDamage,
   frames:{specialist:specialist.frames.map(({h,...f})=>f),diverse:diverse.frames.map(({h,...f})=>f)}};
}
