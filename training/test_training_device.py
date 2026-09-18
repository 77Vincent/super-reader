"""Numerical parity, portable optimizer checkpoints and complete MPS resume/export."""
from __future__ import annotations

import copy
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import run_surface_retraining as coordinator
from text_policy import DATA_POLICY, training_pairs
from train_smoke import (BoundaryChooser, batch_to_device, clone_state, evaluate,
    configure_mps, iterate_batches, mps_memory_restart_needed, predict_record, save_training_state)

ROOT = Path(__file__).resolve().parent.parent


class RuntimeSnapshotTests(unittest.TestCase):
    def test_migration_preserves_preparation_and_checkpoint_and_pins_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';snapshot=run/'source'
            for name in coordinator.FILES:
                for directory, text in [(root,'working'),(snapshot,'frozen')]:
                    path=directory/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(text+name)
            checkpoint=run/'candidate/training-state.pt'
            checkpoint.parent.mkdir();checkpoint.write_bytes(b'complete CPU checkpoint')
            with patch.object(coordinator,'ROOT',root):
                runtime,args=coordinator.training_runtime(run,snapshot,refresh=True,device='mps')
                self.assertEqual(args[:2],['--device','mps'])
                for name in coordinator.FILES:
                    origin=root if name in coordinator.TRAINING_FILES else snapshot
                    self.assertEqual((runtime/name).read_bytes(),(origin/name).read_bytes())
                    self.assertEqual((snapshot/name).read_text(),'frozen'+name)
                metadata=coordinator.read(run/'training-runtime.json')
                self.assertEqual(Path(metadata['checkpoint_before_migration']).read_bytes(),checkpoint.read_bytes())
                (root/'training/train_sharded.py').write_text('later working edit')
                self.assertEqual(coordinator.training_runtime(run,snapshot),(runtime,args))
                with self.assertRaisesRegex(ValueError,'only MPS'):
                    coordinator.training_runtime(run,snapshot,device='cpu')
                (runtime/'training/train_smoke.py').write_text('untracked edit')
                with self.assertRaisesRegex(ValueError,'Frozen training runtime changed'):
                    coordinator.training_runtime(run,snapshot)

    def test_old_run_needs_explicit_migration_and_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);snapshot=run/'source'
            (run/'run.json').write_text(json.dumps({'training':{'threads':5}}))
            with self.assertRaisesRegex(ValueError,'migrate'):
                coordinator.training_runtime(run,snapshot)
            with self.assertRaisesRegex(ValueError,'migrate'):
                coordinator.training_runtime(run,snapshot,device='mps')
            with self.assertRaisesRegex(ValueError,'saved training checkpoint'):
                coordinator.training_runtime(run,snapshot,refresh=True,device='mps')

    def test_fresh_run_is_mps_without_a_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);snapshot=run/'source'
            (run/'run.json').write_text(json.dumps({'training':{'device':'mps'}}))
            self.assertEqual(coordinator.training_runtime(run,snapshot),
                (snapshot,['--device','mps','--mps-memory-fraction','0.4']))


class TrainingEntryTests(unittest.TestCase):
    def test_training_defaults_to_mps_and_rejects_cpu(self):
        import train_smoke, train_sharded
        for module in (train_smoke,train_sharded):
            with self.subTest(module=module.__name__):
                with patch.object(sys,'argv',['trainer']):
                    args=module.parse_arguments()
                self.assertEqual(args.device,'mps')
                self.assertEqual(args.threads,1)
                with patch.object(sys,'argv',['trainer','--device','cpu']),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:module.parse_arguments()
                self.assertEqual(error.exception.code,2)

    def test_missing_metal_fails_instead_of_starting_cpu_training(self):
        with patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'requires Apple Metal'):
                configure_mps()

    def test_memory_guard_reclaims_free_cache_before_requesting_restart(self):
        with patch('torch.mps.recommended_max_memory',return_value=1000),patch('torch.mps.empty_cache') as clear:
            with patch('torch.mps.driver_allocated_memory',return_value=100):
                self.assertFalse(mps_memory_restart_needed(.4));clear.assert_not_called()
            with patch('torch.mps.driver_allocated_memory',side_effect=[300,200]):
                self.assertFalse(mps_memory_restart_needed(.4))
            with patch('torch.mps.driver_allocated_memory',side_effect=[300,280]):
                self.assertTrue(mps_memory_restart_needed(.4))
            self.assertEqual(clear.call_count,2)


@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS hardware required')
class TrainingDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.mps.set_per_process_memory_fraction(.25)

    def setUp(self):
        torch.manual_seed(418)
        texts=['甲乙','今天下雨我们留在家里','普通 空格和8:30应当保留在这里',
               '我们希望读者能够更加轻松地找到句子中的重点并理解不同段落之间的联系']
        self.records=[{'id':str(i),'tokens':list(text),'token_ids':list(range(2,len(text)+2)),
            'target_index':len(text)//2-1,'training_weight':1.,'domain':'news'}
            for i,text in enumerate(texts)]
        self.batch=next(iterate_batches(self.records,8,512,shuffle=False,seed=0,
            batch_indices=[list(range(len(self.records)))]))

    def step(self,model,optimizer):
        batch=batch_to_device(self.batch,next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(batch['token_ids'],batch['token_mask'],batch['gap_mask']),batch['targets'])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        self.assertTrue(torch.isfinite(loss).item())

    def test_masked_forward_gradients_and_evaluation_agree(self):
        cpu=BoundaryChooser(128,32,8);gpu=copy.deepcopy(cpu).to('mps')
        outputs=[];gradients=[]
        for model in (cpu,gpu):
            batch=batch_to_device(self.batch,next(model.parameters()).device)
            logits=model(batch['token_ids'],batch['token_mask'],batch['gap_mask'])
            F.cross_entropy(logits,batch['targets']).backward()
            outputs.append(logits.detach().cpu())
            gradients.append(torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()]))
        torch.testing.assert_close(outputs[0],outputs[1],rtol=2e-4,atol=2e-5)
        self.assertLess(((gradients[0]-gradients[1]).norm()/gradients[0].norm()).item(),1e-4)
        expected=evaluate(cpu,self.records,8,512);actual=evaluate(gpu,self.records,8,512)
        for key in expected:
            if key=='loss':self.assertAlmostEqual(expected[key],actual[key],places=5)
            else:self.assertEqual(expected[key],actual[key])
        self.assertEqual(predict_record(cpu,self.records[-1],512)['predicted_index'],
            predict_record(gpu,self.records[-1],512)['predicted_index'])

    def test_portable_cpu_checkpoint_resumes_mps_optimizer_at_same_next_step(self):
        gpu=BoundaryChooser(128,32,2).to('mps')
        gpu_optimizer=torch.optim.AdamW(gpu.parameters(),lr=.0003,foreach=True)
        self.step(gpu,gpu_optimizer)
        self.step(gpu,gpu_optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt'
            save_training_state(path,{'model_state':gpu.state_dict(),'optimizer_state':gpu_optimizer.state_dict(),
                'best_state':clone_state(gpu),'progress':{'epoch':1,'next_batch':2}})
            state=torch.load(path,weights_only=True)
            self.assertFalse(path.with_name('state.pt.part').exists())
            self.assertTrue(all(v.device.type=='cpu' for v in state['model_state'].values()))
            for moments in state['optimizer_state']['state'].values():
                self.assertTrue(all(v.device.type=='cpu' for v in moments.values() if torch.is_tensor(v)))
                self.assertEqual(moments['step'].item(),2)
            self.assertEqual(state['progress'],{'epoch':1,'next_batch':2})
            restored=BoundaryChooser(128,32,2).to('mps');restored.load_state_dict(state['model_state'])
            restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.0003,foreach=True)
            restored_optimizer.load_state_dict(state['optimizer_state'])
            self.step(gpu,gpu_optimizer);self.step(restored,restored_optimizer)
            for key,expected in clone_state(gpu).items():
                torch.testing.assert_close(expected,clone_state(restored)[key],rtol=1e-6,atol=1e-7)
            reference=BoundaryChooser(128,32,2);reference.load_state_dict(state['model_state'])
            self.assertTrue(all(torch.equal(v,state['model_state'][key]) for key,v in reference.state_dict().items()))

    def test_sharded_default_mps_stop_resume_full_evaluation_and_browser_export(self):
        # Complete, isolated fixture splits; never cap or touch production data.
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);artifact=directory/'candidate'
            pairs=list(training_pairs('引句。今天下雨，我们留在家里。大家一起商量，明天放晴之后出去散步。'))
            self.assertGreater(len(pairs),0)
            characters=set('0123456789:、ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')
            characters.update(''.join(text for text,_,_ in pairs))
            vocabulary={'<pad>':0,'<unk>':1,**{c:i+2 for i,c in enumerate(sorted(characters))}}
            (directory/'vocabulary.json').write_text(json.dumps(vocabulary))
            rows=[[text,target,0,0,0] for text,target,_ in pairs]*128
            shard=directory/'train.jsonl';shard.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
            evaluation=[{'id':str(i),'document_id':'fixture:'+str(i),'domain':'news','tokens':list(text),'target_index':target,'punctuation':punct}
                for i,(text,target,punct) in enumerate(pairs)]
            for split in ('validation','test'):
                (directory/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evaluation))
            (directory/'summary.json').write_text(json.dumps(DATA_POLICY))
            manifest={**DATA_POLICY,'format':'super-reader-sharded-training-v1','vocabulary_path':str(directory/'vocabulary.json'),
                'domains':['news'],'shards':[{'path':str(shard)}],'statistics':{
                    'cells':[[len(rows)]],'domain_samples':{'news':len(rows)},'samples':len(rows),
                    'tokens':sum(len(row[0]) for row in rows),'maximum_sequence_length':max(len(row[0]) for row in rows),
                    'random_baseline_sum':sum(1/(len(row[0])-1) for row in rows),'center_correct':0}}
            (directory/'manifest.json').write_text(json.dumps(manifest))
            command=[sys.executable,str(ROOT/'training/run_sharded.py'),'--manifest',str(directory/'manifest.json'),
                '--data-dir',str(directory),'--artifact-dir',str(artifact),'--epochs','1','--channels','192',
                '--residual-blocks','8','--batch-size','32','--max-tokens-per-batch','512','--threads','1',
                '--learning-rate','0.0003','--gradient-clip','1','--domain-weight-power','0.65','--selection-macro-weight','0.5']
            env=dict(os.environ,DEBUG='0',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0',
                PYTORCH_MPS_FAST_MATH='0',PYTORCH_MPS_PREFER_METAL='0')
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            output=[]
            try:
                for line in process.stdout:
                    output.append(line)
                    if 'batch=1/' in line:
                        process.send_signal(signal.SIGTERM);break
                tail,_=process.communicate(timeout=60);output.append(tail)
                self.assertEqual(process.returncode,0,''.join(output))
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stdout.close()
            checkpoint=artifact/'training-state.pt'
            before=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertEqual(before['execution']['device'],'mps')
            self.assertGreater(before['progress']['next_batch'],0)
            self.assertLess(before['progress']['seen_examples'],len(rows))
            # Simulate the memory guard at the next safe step, then resume from
            # its exit-75 checkpoint exactly as the coordinator does.
            probe="from run_smoke import ensure_dependencies; ensure_dependencies(); import train_sharded as t; t.mps_memory_restart_needed=lambda _: True; t.main()"
            refresh=subprocess.run([sys.executable,'-c',probe,*command[2:],'--resume'],cwd=ROOT,env=env,
                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(refresh.returncode,75,refresh.stdout)
            refreshed=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertEqual(refreshed['progress']['next_batch'],before['progress']['next_batch']+1)
            self.assertGreater(refreshed['progress']['seen_examples'],before['progress']['seen_examples'])
            self.assertEqual(refreshed['training_weight_mean'],before['training_weight_mean'])
            before=refreshed
            after_run=subprocess.run([*command,'--resume'],cwd=ROOT,env=env,
                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            self.assertEqual(after_run.returncode,0,after_run.stdout)
            self.assertIn(f"resumed epoch=1 shard=0 next_batch={before['progress']['next_batch']}",after_run.stdout)
            after=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertEqual(after['progress']['epoch'],2)
            self.assertEqual(after['configuration'],before['configuration'])
            self.assertEqual(after['data_identity'],before['data_identity'])
            self.assertEqual(after['execution']['device'],'mps')
            self.assertGreater(next(iter(after['optimizer_state']['state'].values()))['step'].item(),
                next(iter(before['optimizer_state']['state'].values()))['step'].item())
            metrics=json.loads((artifact/'smoke-metrics.json').read_text())
            self.assertEqual(metrics['data_sizes'],{'train':len(rows),'validation':len(pairs),'test':len(pairs)})
            self.assertEqual(metrics['training_backend']['convolution'],'native_conv1d')
            exported=artifact/'model.js'
            subprocess.run([sys.executable,str(ROOT/'training/export_browser_model.py'),'--artifact-dir',str(artifact),
                '--output',str(exported)],env=env,check=True,stdout=subprocess.PIPE,timeout=30)
            model=BoundaryChooser(len(vocabulary),192,8).to('mps');model.load_state_dict(after['best_state']);model.eval()
            text=pairs[0][0];ids=torch.tensor([[vocabulary[c] for c in text]],device='mps')
            with torch.inference_mode():
                scores=model(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,len(text)-1),device='mps',dtype=torch.bool))[0].cpu().tolist()
            reference=directory/'reference.json';reference.write_text(json.dumps({'tokens':list(text),'scores':scores}))
            js="""const fs=require('node:fs'),vm=require('node:vm');const c=vm.createContext({atob,Intl});
for(const p of process.argv.slice(1,3))vm.runInContext(fs.readFileSync(p,'utf8'),c);
const r=JSON.parse(fs.readFileSync(process.argv[3]));const a=c.SuperReaderModelBackend.scoreTokens(r.tokens);
if(a.length!==r.scores.length||a.indexOf(Math.max(...a))!==r.scores.indexOf(Math.max(...r.scores)))throw Error('predictions differ');
if(a.some((x,i)=>!Number.isFinite(x)||Math.abs(x-r.scores[i])>0.001))throw Error('logits differ');"""
            subprocess.run(['node','-e',js,str(exported),str(ROOT/'src/backend/inference.js'),str(reference)],check=True,timeout=30)

            # The independent non-sharded entry point must also train on MPS.
            smoke_data=directory/'smoke-data';smoke_data.mkdir()
            (smoke_data/'summary.json').write_text(json.dumps(DATA_POLICY))
            for split in ('train','validation','test'):
                (smoke_data/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evaluation))
            smoke_artifact=directory/'smoke-candidate'
            smoke_command=[sys.executable,str(ROOT/'training/run_smoke.py'),'--data-dir',str(smoke_data),
                '--artifact-dir',str(smoke_artifact),'--epochs','1','--channels','16','--residual-blocks','2']
            smoke_run=subprocess.run(smoke_command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(smoke_run.returncode,0,smoke_run.stdout)
            smoke_state=torch.load(smoke_artifact/'training-state.pt',weights_only=True)
            self.assertEqual(smoke_state['execution']['device'],'mps')
            self.assertEqual(smoke_state['progress']['epoch'],2)
            self.assertTrue(smoke_state['optimizer_state']['state'])
            smoke_resume=subprocess.run([*smoke_command,'--resume'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(smoke_resume.returncode,0,smoke_resume.stdout)

    def test_both_comparison_models_train_on_mps(self):
        import argparse
        import compare_boundary_encoders as comparison
        args=argparse.Namespace(seed=418,learning_rate=.0003,epochs=1,batch_size=8,max_tokens_per_batch=512)
        for model_type in (comparison.SharedPairBoundaryChooser,comparison.CandidateWindowBoundaryChooser):
            model=model_type(128,8,1)
            with self.subTest(model=model_type.__name__),contextlib.redirect_stdout(io.StringIO()):
                report=comparison.train_model('fixture',lambda:model,
                    {split:self.records for split in ('train','validation','test')},args)
            self.assertEqual(next(model.parameters()).device.type,'mps')
            self.assertEqual(report['best_epoch'],1)
            self.assertTrue(all(torch.isfinite(p).all().item() for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
