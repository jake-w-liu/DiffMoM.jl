"""Counter completion and genuine errors must survive without a selected deadline."""
import pathlib,subprocess,sys,tempfile,unittest
from unittest import mock
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))
from slopfix_lib import counting

class CounterCompletionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=self.directory.name;self.file=pathlib.Path(self.root)/'example.jl';self.file.write_text('x=1\n')
        for name in ('julia_path','scc_path'):
            patch=mock.patch.object(counting,name,return_value=name.split('_')[0]);patch.start();self.addCleanup(patch.stop)

    def test_identity_and_classification_wait_for_completion(self):
        cases=[(counting.scc_identity,(),'scc version 1.0\n'),(counting.run_scc,(self.root,[],True),'[]'),
            (counting.julia_identity,(),'julia/1.12.7\n'),(counting.run_julia,(self.root,['example.jl']),str(self.file)+'\t1\t0\t0\n')]
        for function,args,output in cases:
            with self.subTest(operation=function.__name__),mock.patch.object(counting.subprocess,'run',return_value=subprocess.CompletedProcess([],0,output,'')) as run:
                result=function(*args)
                self.assertIsNone(run.call_args.kwargs.get('timeout'))
                self.assertEqual(run.call_count,1)
                if function is counting.run_julia:self.assertEqual(result['example.jl'].code,1)
                elif function is counting.run_scc:self.assertEqual(result,[])
                elif function is counting.scc_identity:self.assertEqual(result,'scc/1.0')
                else:self.assertEqual(result,output.strip())

    def test_scc_nonzero_and_reported_errors_are_rejected(self):
        for code,stderr in ((1,'failed'),(0,'ERROR: unreadable file')):
            with self.subTest(code=code),mock.patch.object(counting.subprocess,'run',return_value=subprocess.CompletedProcess([],code,'[]',stderr)):
                with self.assertRaises(counting.SccOutputError):counting.run_scc(self.root,[],True)

    def test_julia_nonzero_and_missing_files_are_rejected(self):
        for code,stdout in ((1,''),(0,''),(0,str(self.file)+'\tERR\tinvalid syntax\n')):
            with self.subTest(code=code,stdout=stdout),mock.patch.object(counting.subprocess,'run',return_value=subprocess.CompletedProcess([],code,stdout,'failed')):
                with self.assertRaises(counting.JuliaOutputError):counting.run_julia(self.root,['example.jl'])

    def test_spawn_errors_stay_visible(self):
        for function,args,error in ((counting.run_scc,(self.root,[],True),counting.SccUnavailable),(counting.run_julia,(self.root,['example.jl']),counting.JuliaUnavailable)):
            with self.subTest(operation=function.__name__),mock.patch.object(counting.subprocess,'run',side_effect=OSError('cannot launch')):
                with self.assertRaises(error):function(*args)

if __name__=='__main__':unittest.main()
