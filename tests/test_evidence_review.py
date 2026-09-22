import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from zulf_tools.evidence_review import require_compatible, review_decay_evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.base=dict(source_arrays_sha256='abc',discovery_groups=[0,2],validation_groups=[1,3],
            preprocessing=dict(start_sample=0,stop_sample=1000,sg_window=0,sg_order=2))

    def test_source_split_and_processing_must_match(self):
        require_compatible(self.base,copy.deepcopy(self.base),True)
        for key,value in [('source_arrays_sha256','def'),('discovery_groups',[1,3]),('validation_groups',[0,2])]:
            other=copy.deepcopy(self.base);other[key]=value
            with self.assertRaises(ValueError): require_compatible(self.base,other)
        for key,value in [('stop_sample',500),('sg_window',301),('remove_mean',False)]:
            other=copy.deepcopy(self.base);other['preprocessing'][key]=value
            with self.assertRaises(ValueError): require_compatible(self.base,other,True)
            require_compatible(self.base,other,False)

    def test_sampling_precision_cannot_hide_method_dependence(self):
        candidate=dict(frequencies_hz=[40.],t2star_s=[1.],range_hz=[35,45],
            numerical_diagnostics=dict(numerical_warning_flags=[]),optimizer_converged=True)
        parent=dict(self.base,operation='fit_frequency_decay',parent_run_id='groups',candidates=[candidate])
        signal=dict(self.base,operation='inspect_repeat_signals',candidates=[dict(frequency_hz=40.,classification='reproducible_signal_candidate')])
        stability=dict(operation='inspect_decay_stability',parent_run_id='fit',candidate_index=0,requested_groups=[1,3],
            group_fits=[dict(matched_t2star_s=[1.],frequency_shift_hz=[0.],numerical_diagnostics=dict(requires_review=False))]*2)
        sample=dict(operation='resample_decay_groups',parent_run_id='fit',candidate_index=0,block_length=1,
            summary=dict(status='conditional_descriptive_percentiles',t2star_percentiles_s=[[.99],[1.],[1.01]]))
        other=dict(parent,candidates=[dict(candidate,t2star_s=[1.07])])
        records=dict(fit=parent,signal=signal,stable=stability,sample=sample,other=other)
        residual=dict(available=True,requires_review=False,status='no_large_residual_flag')
        with tempfile.TemporaryDirectory() as tmp, patch('zulf_tools.evidence_review.storage.get_result',side_effect=records.__getitem__), patch('zulf_tools.evidence_review.load_group_averages',return_value=(None,None,dict(arrays_sha256='abc',sampling_rate_hz=1000))), patch('zulf_tools.evidence_review.candidate_residual_evidence',return_value=residual):
            args=dict(fit_run_id='fit',signal_run_ids=['signal'],stability_run_id='stable',resampling_run_ids=['sample'],comparison_refs=[dict(run_id='other',candidate_index=0)],record={},directory=Path(tmp),cancel=lambda:False,progress=lambda *_:None)
            row=review_decay_evidence(**args)['modes'][0]
            self.assertEqual(row['interpretation_status'],'model_sensitive_candidate')
            self.assertEqual(row['sensitivity_comparisons'][0]['outside_sampling_ranges'],['sample'])
            self.assertFalse(row['physical_component_accepted'])
            residual['requires_review']=True
            self.assertEqual(review_decay_evidence(**args)['modes'][0]['interpretation_status'],'band_model_residual_unresolved')
            residual['requires_review']=False
            sample['summary']['status']='insufficient_or_unstable_resampling'
            self.assertEqual(review_decay_evidence(**args)['modes'][0]['interpretation_status'],'unstable_decay')
            stability['candidate_index']=1
            with self.assertRaisesRegex(ValueError,'another candidate'): review_decay_evidence(**args)


if __name__=='__main__': unittest.main()
