import unittest
import numpy as np
import torch
from vguard.discriminant_recovery import (fit_discriminant, EnrollmentDiscriminant,
    class_margin, quality_caps, acceptable, make_projector)


class DiscriminantTests(unittest.TestCase):
    def enrollment(self):
        rng=np.random.default_rng(47)
        means=np.array([[0.,0.,0.],[4.,0.,0.],[0.,4.,0.]])
        y=np.repeat(np.arange(3),20)
        return means[y]+rng.normal(0,.2,(60,3)),y

    def test_enrollment_generalizes_on_clear_class_centers(self):
        x,y=self.enrollment(); p=fit_discriminant(x,y,3)
        classifier=EnrollmentDiscriminant(p)
        self.assertEqual(classifier(torch.tensor([[0.,0.,0.],[4.,0.,0.],[0.,4.,0.]])).argmax(-1).tolist(),[0,1,2])
        self.assertEqual(p['enrollment_accuracy'],1.)

    def test_common_embedding_offset_does_not_change_class_scores(self):
        x,y=self.enrollment(); shift=np.array([17.,-40.,100.])
        a,b=[EnrollmentDiscriminant(fit_discriminant(v,y,3)) for v in (x,x+shift)]
        torch.testing.assert_close(a(torch.tensor(x,dtype=torch.float32)),
            b(torch.tensor(x+shift,dtype=torch.float32)),atol=1e-5,rtol=1e-5)

    def test_singular_covariance_has_finite_gradients_into_embedding_only(self):
        x,y=self.enrollment(); x[:,2]=1.
        classifier=EnrollmentDiscriminant(fit_discriminant(x,y,3))
        value=torch.tensor(x[:1],dtype=torch.float32,requires_grad=True)
        class_margin(classifier(value),0).mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(value.grad.norm()),0.)
        self.assertEqual(len(list(classifier.parameters())),0)

    def test_margin_positive_means_owner_wins_and_negative_means_lost(self):
        scores=torch.tensor([[3.,2.,1.],[1.,2.,3.]])
        self.assertEqual(class_margin(scores,0).tolist(),[1.,-2.])
        with self.assertRaises(ValueError): class_margin(scores,3)

    def test_relaxed_budget_increases_actual_spectral_amplitude_cap(self):
        pp={'n_fft':64,'hop_length':16,'win_length':64}
        theta=np.zeros((10,33))
        strict,relaxed=[make_projector(theta,.1,pp,d) for d in (0.,3.)]
        torch.testing.assert_close(relaxed.limit, strict.limit * 10**(3/20))
        with self.assertRaises(ValueError): make_projector(theta,.1,pp,-3.)

    def test_missing_enrollment_or_invalid_shrinkage_rejected(self):
        x,y=self.enrollment()
        for labels,shrink in [(y,0.),(y,1.1),(np.zeros_like(y),.2)]:
            with self.assertRaises(ValueError): fit_discriminant(x,labels,3,shrink)

    def test_export_acceptance_checks_all_quality_proxies_and_finiteness(self):
        v={'ctc':1.,'own_similarity':.98,'spectrum':.1,'defense':-.1}
        caps=quality_caps(v)
        self.assertTrue(acceptable(v,caps))
        for key,value in [('ctc',1.02),('own_similarity',.97),('spectrum',.14),('defense',float('nan'))]:
            self.assertFalse(acceptable({**v,key:value},caps))


if __name__=='__main__': unittest.main()
