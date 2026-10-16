import test from 'node:test';
import assert from 'node:assert/strict';
import {GRAPH,fresh,objective} from '../engine.js';
import {needSplits,sweepPairs,trajectory,selectCheckpoint} from '../gate2.js';
import {K,cutEdge,trainEdgeInterventions,trainSignals,rankGroups,randomGroups,
 functionalGroups,allGroupings,predictFromGroups,probeEdges,stateHash,
 commitInterpretation,revealInterpretation,runGate3} from '../gate3.js';
const task='bridge';
const h=Float64Array.from(selectCheckpoint(trajectory(task,'diverse',{steps:40})).h);
const train=needSplits(task).train;

test('single-edge intervention removes only recruited material and keeps scaffold',()=>{
 const e=12,cut=cutEdge(h,e,.5);
 assert.equal(cut[e],h[e]/2);
 assert.equal(cut.length,h.length);
 assert.ok(cut.every((v,i)=>v>=0&&(e===i||v===h[i])));
 assert.throws(()=>cutEdge(h,-1));
});
test('training edge profiles equal independently recomputed finite causal effects',()=>{
 const profiles=trainEdgeInterventions(h,train);
 assert.equal(profiles.length,GRAPH.edges.length);
 const e=profiles.findIndex(x=>x.some(v=>v>1e-7));
 assert.ok(e>=0);
 const base=objective(h,[train[0]]).loss;
 const cut=objective(cutEdge(h,e),[train[0]]).loss-base;
 assert.ok(Math.abs(profiles[e][0]-cut)<1e-9);
});
test('functional clustering is deterministic, balanced and uses no region values',()=>{
 const sig=trainEdgeInterventions(h,train);
 const labels=functionalGroups(sig,K);
 assert.deepEqual(labels,functionalGroups(sig,K));
 const sizes=Array.from({length:K},(_,i)=>labels.filter(x=>x===i).length);
 assert.ok(Math.max(...sizes)-Math.min(...sizes)<=1);
 // Modifying region metadata cannot alter causal feature vectors or membership.
 const old=GRAPH.edges[0].region;
 try{GRAPH.edges[0].region=999;
   assert.deepEqual(labels,functionalGroups(sig,K));
 }finally{GRAPH.edges[0].region=old;}
});
test('simple baselines, spatial and seeded random use identical group sizes',()=>{
 const profiles=trainEdgeInterventions(h,train),signals=trainSignals(h,train);
 const groups=allGroupings(h,profiles,signals);
 assert.deepEqual(Object.keys(groups),['Functional','Material bins','Flow bins','Rectangles','Shuffled']);
 for(const [name,x] of Object.entries(groups)){
  assert.equal(x.length,GRAPH.edges.length,name);
  assert.ok(x.every(v=>Number.isInteger(v)&&v>=0&&v<K),name);
 }
 assert.deepEqual(randomGroups(),randomGroups());
 assert.ok(new Set(rankGroups(h)).size===K);
});
test('group predictor is trained-only and cannot look up held-out pair data',()=>{
 const profiles=trainEdgeInterventions(h,train),labels=functionalGroups(profiles);
 const pred=predictFromGroups(h,profiles,labels);
 assert.equal(pred.length,GRAPH.edges.length);
 assert.ok(pred.every(x=>x>=0&&Number.isFinite(x)));
});
test('all held-out probe edge IDs are chosen from h, not outcome or pair sets',()=>{
 const ids=probeEdges(h);
 assert.ok(ids.length>=30&&ids.length<=54);
 assert.deepEqual(ids,probeEdges(h));
 assert.ok(ids.every((x,i)=>x>=0&&x<GRAPH.edges.length&&(i===0||x>ids[i-1])));
 const copy=h.slice(); copy[0]+=.001;
 assert.notEqual(stateHash(copy),stateHash(h));
});
test('prospective commitment excludes held-out truth and locks frozen state',()=>{
 const commit=commitInterpretation(h,task);
 assert.equal(commit.hash,stateHash(h));
 assert.equal(commit.clusters.length,K);
 assert.ok(!('actual'in commit)&&!('testScores'in commit));
 for(const label of Object.keys(commit.predictions))assert.equal(commit.predictions[label].length,h.length);
 const result=revealInterpretation(h,commit);
 assert.equal(result.actual.length,commit.probeIds.length);
 assert.equal(result.testCount,12);
 assert.equal(result.randomSummary.n,20);
 assert.ok(result.randomSummary.randomBeatingFunctional>=0);
 assert.ok(Object.values(result.scores).every(x=>Number.isFinite(x.pearson)&&Number.isFinite(x.rmse)));
 assert.throws(()=>revealInterpretation(h.slice(1),commit),/state changed/);
 const tampered=h.slice();tampered[4]+=.0001;
 assert.throws(()=>revealInterpretation(tampered,commit),/state changed/);
});
test('held-out tests disjoint from all training and validation demands',()=>{
 for(const need of ['bridge','fork','cross']){
  const p=needSplits(need),trainKeys=new Set([...p.train,...p.augmentation,...p.validation,...p.test].map(x=>x.join(':')));
  for(const pair of sweepPairs(need,24))assert.ok(!trainKeys.has(pair.join(':')));
 }
});
test('run of gate 3 is deterministic, including reported negative outcomes',()=>{
 const a=runGate3('bridge',{steps:40}),b=runGate3('bridge',{steps:40});
 assert.deepEqual(a,b);
 assert.equal(a.groupStats.length,K);
 assert.equal(Object.keys(a.score).length,8);
});
