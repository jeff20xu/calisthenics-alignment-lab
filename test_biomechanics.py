import unittest
import numpy as np
from biomechanics import measurements, in_posture, RepCounter


def pose(**changes):
    p = dict(wrist=(100,600),elbow=(100,500),shoulder=(100,400),hip=(100,250),knee=(100,150),ankle=(100,50))
    p.update(changes)
    return {k:np.array(v,dtype=float) for k,v in p.items()}


class GeometryTests(unittest.TestCase):
    def test_stacked_handstand(self):
        p = pose()
        m = measurements(p,150)
        self.assertTrue(in_posture(p,'Handstand'))
        self.assertEqual(m['elbow_deg'],180)
        self.assertEqual(m['shoulder_deg'],180)
        self.assertEqual(m['body_line_offset'],0)
        self.assertEqual(m['stack_offset'],0)

    def test_scale_and_translation_invariance(self):
        p = pose(hip=(130,250))
        transformed = {k:v*2+50 for k,v in p.items()}
        a,b = measurements(p,150),measurements(transformed,300)
        for key in ('stack_offset','body_line_offset','hip_deg','leg_lever'):
            self.assertAlmostEqual(a[key],b[key])

    def test_planche_geometry_and_gravity_model(self):
        p = pose(wrist=(80,250),elbow=(90,175),shoulder=(100,100),hip=(250,100),knee=(350,100),ankle=(450,100))
        m = measurements(p,150,70,.5)
        self.assertTrue(in_posture(p,'Planche'))
        self.assertFalse(in_posture(p,'Handstand'))
        self.assertEqual(m['torso_horizontal_deg'],0)
        self.assertEqual(m['hip_deg'],180)
        expected = 70*9.81*(.5*75+.2*200+.09*300)*(.5/150)
        self.assertAlmostEqual(m['partial_gravity_moment_nm'],expected)

    def test_degenerate_joint_is_not_measured(self):
        self.assertIsNone(measurements(pose(elbow=(100,600)),150))

    def test_full_rep_requires_sustained_cycle(self):
        c = RepCounter()
        for t,e in [(0,170),(.15,170),(.25,90),(.40,90),(.50,170),(.65,170)]:
            c.update(t,e)
        self.assertEqual(c.count,1)
        c.update(.75,90)
        c.update(.80,170)
        self.assertEqual(c.count,1)

    def test_missing_tracking_cannot_complete_rep(self):
        c = RepCounter()
        for t,e in [(0,170),(.15,170),(.25,90),(.40,90),(.45,None),(.5,170),(.65,170)]:
            c.update(t,e)
        self.assertEqual(c.count,0)


if __name__=='__main__':
    unittest.main()
