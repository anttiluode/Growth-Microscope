import {audit} from './gate2.js';
import {writeFileSync} from 'node:fs';
const steps=Number(process.argv[2]||320);
if(!Number.isInteger(steps)||steps<10||steps>1000)throw Error('step count must be integer 10..1000');
const report={protocol:'Growth Microscope Gate 2',version:2,steps,
 design:'Specialist versus rotating seven-pair diverse training, exactly three Kirchhoff solves per gradient step; independent three-case validation chooses from fixed checkpoints; unseen three-stress and 24-pair synthetic sweep evaluate once.',
 warnings:['Synthetic, hand-designed endpoints; deterministic, not randomized statistical replication','The 12 spatial region groups are predefined','Public source implies procedural, not cryptographic blinding','Uniform baseline is simple but material matched; matched-purpose topology optimization not yet compared'],
 tasks:[]};
for(const task of ['bridge','fork','cross']){
 const r=audit(task,{steps});report.tasks.push(r);
 const s=r.specialist,d=r.diverse;
 console.log(`${task.padEnd(7)}  specialist picked ${String(s.selected.step).padStart(3)} / ${steps}  test-sweep ${s.selected.sweep.toFixed(4)} (final ${s.final.sweep.toFixed(4)})`);
 console.log(`         diverse    picked ${String(d.selected.step).padStart(3)} / ${steps}  test-sweep ${d.selected.sweep.toFixed(4)} (final ${d.final.sweep.toFixed(4)})  uniform ${r.equalMaterialUniformTest.specialist.sweep.toFixed(4)}`);
 for(const [arm,scores] of Object.entries(r.damagePrediction)){
  console.log(`         ${arm} damage: `+Object.entries(scores).map(([k,v])=>`${k} r=${v.pearson.toFixed(3)}`).join(' | '));
 }
}
if(process.argv.includes('--save')){
 writeFileSync('results/gate2-reference.json',JSON.stringify(report,null,2)+'\n');
 console.log('Saved results/gate2-reference.json');
}
