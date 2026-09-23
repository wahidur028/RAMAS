import json,tempfile,unittest
from pathlib import Path
from run_suite import artifact_map,verify_completed,stable_record
from stage63lib.journal import atomic_json
from suite64.metrics import atomic_csv
import pandas as pd
from unittest.mock import patch

class SuiteIntegrityTests(unittest.TestCase):
    def test_root_manifest_protects_nested_arm_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);arm=root/'arms'/'a';arm.mkdir(parents=True)
            atomic_json(arm/'RUN_COMPLETE.json',{'spec':'original'})
            mapping=artifact_map(root)
            self.assertIn('arms/a/RUN_COMPLETE.json',mapping)
            atomic_json(root/'RUN_COMPLETE.json',{'artifact_sha256':mapping})
            verify_completed(root)
            atomic_json(arm/'RUN_COMPLETE.json',{'spec':'changed'})
            with self.assertRaises(RuntimeError):verify_completed(root)
    def test_resume_cannot_overwrite_changed_source_audit(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'audit.json';stable_record(p,{'sha':'original'})
            stable_record(p,{'sha':'original'})
            with self.assertRaises(RuntimeError):stable_record(p,{'sha':'other'})
            self.assertEqual(json.loads(p.read_text()),{'sha':'original'})
    def test_failed_csv_publication_preserves_prior_complete_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ledger.csv'
            old=pd.DataFrame({'value':[1,2,3]});new=pd.DataFrame({'value':range(11256)})
            atomic_csv(old,path);before=path.read_bytes()
            with patch('suite64.metrics.os.replace',side_effect=OSError('interrupted publication')):
                with self.assertRaises(OSError):atomic_csv(new,path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(td).glob('*.tmp')),[])
            atomic_csv(new,path)
            self.assertEqual(len(pd.read_csv(path)),11256)
if __name__=='__main__':unittest.main()

