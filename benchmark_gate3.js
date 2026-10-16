/** Frozen Gate 3 reference protocol; no tuning on test observations. */
import {writeFileSync,mkdirSync} from 'node:fs';
import {runGate3} from './gate3.js';
const steps=Number(process.argv[2]??320);
if(!Number.isInteger(steps)||steps<10||steps>640)throw Error('steps must be integer 10..640');
const cases=[];
for(const task of ['bridge','fork','cross'])for(const arm of ['specialist','diverse']){
 const result=runGate3(task,{steps,arm});
 cases.push({task,arm,selectedStep:result.selectedStep,probeCount:result.probeCount,
  scores:result.score,coherence:result.coherence,randomSummary:result.randomSummary,
  groupSizes:result.groupStats.map(x=>x.count)});
 console.log(`\n${task} / ${arm}: selected ${result.selectedStep}, ${result.probeCount} single-edge interventions`);
 for(const [name,v] of Object.entries(result.score))console.log(` ${name.padEnd(23)} r=${v.pearson.toFixed(3)}  RMSE=${v.rmse.toFixed(5)}  top5=${v.top5}/5`);
}
const report={protocol:'growth-microscope-gate3-v1',steps,armPairing:['specialist','diverse'],
 test:'12 of the frozen 24 disjoint sweepPairs; never used for clustering or fitting',
 training:'Three original training stimulus configurations only, exact half-edge finite cuts',
 groups:'Balanced 12-way functional kmeans on standardized 3D causal edge signatures, geometry excluded',
 diagnostics:'Held-out SINGLE edge cuts chosen by trained material ranking plus deterministic evenly spaced IDs',
 caveats:['Same synthetic Kirchhoff task family and fixed graph; no semantic neuron claims',
   'High-material-biased intervention sample, not random independent draws',
   'All grouping controls share the same training causal measurements',
   'Simple gradient and physical measurements have privileged known solver access',
   '12 frozen procedural test pairs are public and selected before scoring',
   'Only one fixed network and deterministic clustering per task/arm; no uncertainty intervals'],
 cases};
mkdirSync('results',{recursive:true});
const path='results/gate3-reference.json';writeFileSync(path,JSON.stringify(report,null,2)+'\n');
console.log(`\nSaved ${path}`);
