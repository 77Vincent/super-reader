"""Coordinator recovery must retain progress without restarting failed jobs blindly."""
import os
from pathlib import Path
import tempfile
import unittest

from run_web_continuation import run_with_recovery


class WebContinuationTests(unittest.TestCase):
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
