"""Behavioral regression checks for E7 host memory gates and restart accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pirl_runtime_guard import (GIB, memory_sample, evaluation_allowed, should_defer,
                                FatalLogScanner, lineage_records, metric_segments)
import pirl_five_task_controller as controller

POLICY = {'eval_reserve_gib':28, 'minimum_headroom_gib':12,
          'eval_ram_reserve_gib':20, 'minimum_ram_headroom_gib':24,
          'overlap_after_training_iterations':2, 'defer_ram_headroom_gib':18}


class RuntimeGuardTests(unittest.TestCase):
    def test_gpu_capacity_cannot_override_low_host_memory(self):
        gpu = {'used_gib':20, 'total_gib':96}
        ram = {'available_gib':30,'used_gib':95,'total_gib':125,'pressure_fraction':.76}
        self.assertFalse(evaluation_allowed(gpu,ram,POLICY,training=True,iterations=3,gpu_peak=45,ram_peak=65))
        ram.update(available_gib=60,used_gib=65)
        self.assertTrue(evaluation_allowed(gpu,ram,POLICY,training=True,iterations=3,gpu_peak=45,ram_peak=65))
        self.assertFalse(evaluation_allowed(gpu,ram,POLICY,training=True,iterations=1,gpu_peak=45,ram_peak=65))

    def test_loading_evaluator_reservation_blocks_double_launch(self):
        gpu={'used_gib':10,'total_gib':96};ram={'available_gib':50,'total_gib':125,'used_gib':75}
        self.assertTrue(evaluation_allowed(gpu,ram,POLICY))
        self.assertFalse(evaluation_allowed(gpu,ram,POLICY,pending=1))
        self.assertFalse(evaluation_allowed({'used_gib':60,'total_gib':96},ram,POLICY))

    def test_host_pressure_defers_evaluation_before_ray_oom(self):
        gpu={'used_gib':40,'total_gib':96}
        self.assertTrue(should_defer(gpu,{'available_gib':16,'pressure_fraction':.87},POLICY))
        self.assertFalse(should_defer(gpu,{'available_gib':50,'pressure_fraction':.60},POLICY))

    def test_memory_limit_uses_tighter_host_or_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);proc=p/'proc';cg=p/'cgroup';proc.mkdir();cg.mkdir()
            (proc/'meminfo').write_text(f'MemTotal: {128*GIB//1024} kB\nMemAvailable: {100*GIB//1024} kB\n')
            (cg/'memory.max').write_text(str(64*GIB));(cg/'memory.current').write_text(str(60*GIB))
            (cg/'memory.stat').write_text(f'inactive_file {8*GIB}\nactive_file {2*GIB}\n')
            r=memory_sample(proc,cg);self.assertEqual(r['total_gib'],64);self.assertEqual(r['available_gib'],14)
            (cg/'memory.max').write_text('max');r=memory_sample(proc,cg);self.assertEqual(r['available_gib'],78)

    def test_fatal_worker_error_detected_across_log_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'log';scan=FatalLogScanner();p.write_text('normal training\nray.exceptions.OutOf')
            self.assertIsNone(scan.scan(p))
            with p.open('a') as f:f.write('MemoryError: killed worker\n')
            self.assertIn('OutOfMemoryError',scan.scan(p))

    def test_resume_keeps_model_lineage_and_counts_discarded_cost(self):
        rows=[{'time_unix':i,'requested_actions':32} for i in range(1,60)]
        rows += [{'time_unix':100+i,'requested_actions':32} for i in range(1,4)]
        active=lineage_records(rows,[{'retained_until_unix':50,'started_unix':100}])
        self.assertEqual(len(active),53)
        self.assertEqual((len(rows)-len(active))*32,9*32)
        again=lineage_records(rows,[{'retained_until_unix':50,'started_unix':100},
                                   {'retained_until_unix':50,'started_unix':200}])
        self.assertEqual(len(again),50)

    def test_metrics_rollback_does_not_claim_previous_iteration59(self):
        old=b'old iterations through 59\n';new=b'resumed iteration 51\n'
        segments=list(metric_segments(old+new,[{'metrics_start_bytes':len(old),'resume_step':50}]))
        self.assertEqual(segments,[(None,old.decode()),(50,new.decode())])

    def test_missing_optimizer_metadata_is_not_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'RUN',Path(tmp)):
            job={'name':'test','tasks':['a','b']};p=controller.checkpoint(job,50)
            (p/'model_state_dict').mkdir(parents=True);(p/'dcp_checkpoint').mkdir()
            (p/'model_state_dict/full_weights.pt').write_bytes(b'weights')
            (p/'diagnostic-state.json').write_text(json.dumps({'global_step':50,'optimizer_steps':100}))
            (p/'dcp_checkpoint/a.distcp').write_bytes(b'state')
            self.assertFalse(controller.checkpoint_ready(job,50))
            (p/'dcp_checkpoint/.metadata').write_bytes(b'metadata')
            self.assertTrue(controller.checkpoint_ready(job,50))
            (p/'diagnostic-state.json').write_text(json.dumps({'global_step':50,'optimizer_steps':99}))
            self.assertFalse(controller.checkpoint_ready(job,50))


if __name__=='__main__': unittest.main()
