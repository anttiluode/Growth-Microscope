import {runAll} from './engine.js';
import {writeFileSync} from 'node:fs';
const steps=Number(process.argv[2]||320);
if(!Number.isInteger(steps)||steps<1||steps>5000)throw Error('steps must be integer 1..5000');
const result={protocol:'Growth Microscope v1: train-region causal fingerprints vs shifted held-out demands',steps,
 intervention:'thinning 50% of grown conductance in one 3x4 spatial region; scaffold remains',
 limitations:['Train and test pairs belong to the same need family, differing by endpoint shifts','All trained structures come from a deterministic passive Kirchhoff lattice','Adjoint uses held-out unablated potentials; it is not a blind task inference','No novel neuron birth, learned language concepts or autonomous interpretability'],
 runs:runAll({steps})};
for(const row of result.runs){console.log(`\n${row.task}: train ${row.train.toFixed(5)}, held ${row.held.toFixed(5)}, material ${row.material.toFixed(2)}`);
for(const [name,s] of Object.entries(row.comparison.scores))console.log(`  ${name.padEnd(30)} r=${s.r.toFixed(3)} RMSE=${s.rmse.toFixed(6)} top3=${s.top3_overlap}/3`);}
if(process.argv.includes('--save')){writeFileSync('benchmark-results.json',JSON.stringify(result,null,2)+'\n');console.log('Wrote benchmark-results.json');}
