"""Coordinator recovery must retain progress without restarting failed jobs blindly."""
import os
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from run_web_continuation import run_with_recovery, validate_training_options, extend_completed_run


class WebContinuationTests(unittest.TestCase):
    def completed_fixture(self, run):
        import torch
        candidate = run / 'candidate'
        candidate.mkdir()
        torch.save({'progress': {'epoch': 2, 'shard': 0, 'next_batch': 0},
                    'model_state': {'weight': torch.tensor([3.])},
                    'optimizer_state': {'state': {0: {'step': torch.tensor(17.)}}}},
                   candidate / 'training-state.pt')
        (candidate / 'smoke-metrics.json').write_text('{"best_epoch":1}')
        plan = {'epochs': 1, 'source_hashes': {'train.py': 'frozen'}, 'data': '/same/data'}
        for name, value in [('run.json', plan), ('status.json', {'stage': 'complete'}),
                            ('result.json', {'history': [{'epoch': 1}]})]:
            (run / name).write_text(json.dumps(value))
        return plan

    def test_extension_archives_results_and_preserves_exact_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            plan = self.completed_fixture(run)
            checkpoint = run / 'candidate/training-state.pt'
            before = checkpoint.read_bytes()
            extend_completed_run(run, plan, 2, resume=True)
            self.assertEqual(checkpoint.read_bytes(), before)
            archive = run / 'completed-epochs/epoch-1'
            self.assertEqual((archive / 'candidate/training-state.pt').read_bytes(), before)
            self.assertEqual(json.loads((archive / 'run.json').read_text())['epochs'], 1)
            self.assertEqual(json.loads((archive / 'result.json').read_text())['history'], [{'epoch': 1}])
            self.assertEqual(plan['epochs'], 2)
            self.assertEqual(plan['source_hashes'], {'train.py': 'frozen'})
            self.assertFalse((run / 'result.json').exists())
            self.assertEqual(json.loads((run / 'status.json').read_text())['stage'], 'ready-to-resume')
            # Recover both possible interruptions after publishing the plan.
            for restore_result in (True, False):
                (run / 'status.json').write_text('{"stage":"complete"}')
                if restore_result:
                    (run / 'result.json').write_text((archive / 'result.json').read_text())
                extend_completed_run(run, plan, 2, resume=True)
                self.assertFalse((run / 'result.json').exists())
                self.assertEqual(json.loads((run / 'status.json').read_text())['stage'], 'ready-to-resume')
            self.assertEqual(len(plan['epoch_extensions']), 1)

    def test_extension_rejects_missing_resume_unfinished_or_optimizerless_checkpoints(self):
        import torch
        for problem in ('resume', 'reduce', 'incomplete', 'optimizer'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as folder:
                run = Path(folder)
                plan = self.completed_fixture(run)
                if problem == 'incomplete':
                    (run / 'status.json').write_text('{"stage":"stopped"}')
                if problem == 'optimizer':
                    path = run / 'candidate/training-state.pt'
                    state = torch.load(path, weights_only=True)
                    state['optimizer_state']['state'] = {}
                    torch.save(state, path)
                before = (run / 'run.json').read_bytes()
                with self.assertRaises(ValueError):
                    extend_completed_run(run, plan, 0 if problem == 'reduce' else 2, resume=problem != 'resume')
                self.assertEqual((run / 'run.json').read_bytes(), before)
                self.assertFalse((run / 'completed-epochs').exists())

    def test_changed_learning_rate_or_selection_cannot_silently_resume(self):
        plan = {'training': {'learning_rate': .00003, 'selection_macro_weight': 0}, 'checkpoint_shards': 2}
        options = dict(learning_rate=.00003, selection_macro_weight=0, checkpoint_shards=2)
        validate_training_options(plan, Namespace(**options))
        for key, value in [('learning_rate', .0003), ('selection_macro_weight', .5), ('checkpoint_shards', 4)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_options(plan, Namespace(**{**options, key: value}))

    def test_memory_refresh_resumes_saved_state_instead_of_initializing_again(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            calls, refreshes = [], []

            def run_once(command):
                calls.append(command)
                if len(calls) < 3:
                    checkpoint.write_text(f"next batch {len(calls)}")
                    os.utime(checkpoint, ns=(len(calls) * 1_000_000_000,) * 2)
                    return 75
                return 0

            run_with_recovery("training", ["train", "--epochs", "1", "--initialize-from", "best.pt"],
                              run_once=run_once, checkpoint=checkpoint, should_stop=lambda: False,
                              on_refresh=lambda: refreshes.append(True))
            self.assertEqual(len(refreshes), 2)
            self.assertEqual(calls[1:], [["train", "--epochs", "1", "--resume"]] * 2)

    def test_refresh_without_new_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            for exists in [False, True]:
                if exists:
                    checkpoint.write_text("old checkpoint")
                with self.subTest(exists=exists), self.assertRaisesRegex(RuntimeError, "did not advance"):
                    run_with_recovery("training", ["train"], run_once=lambda _: 75,
                                      checkpoint=checkpoint, should_stop=lambda: False,
                                      on_refresh=lambda: self.fail("Must not refresh"))

    def test_stop_after_checkpoint_exits_without_relaunching(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            stopped = []

            def run_once(_):
                checkpoint.write_text("saved")
                stopped.append(True)
                return 75

            with self.assertRaises(InterruptedError):
                run_with_recovery("training", ["train"], run_once=run_once,
                                  checkpoint=checkpoint, should_stop=lambda: bool(stopped),
                                  on_refresh=lambda: self.fail("Must honor stop request"))
            self.assertEqual(checkpoint.read_text(), "saved")

    def test_other_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            for stage, code in [("prepare", 75), ("training", 1)]:
                with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, f"exited {code}"):
                    run_with_recovery(stage, ["command"], run_once=lambda _: code,
                                      checkpoint=Path(directory) / "state.pt", should_stop=lambda: False,
                                      on_refresh=lambda: self.fail("Must not retry an unrelated failure"))


if __name__ == "__main__":
    unittest.main()
