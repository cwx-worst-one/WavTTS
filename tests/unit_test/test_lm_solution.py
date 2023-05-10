"""Test lm solution"""
# pylint: skip-file
import core.runner
from core.models.lm.hotword_fst import HotwordFst
from core.solutions.inference.hypothesis import Hypothesis
from core.solutions.lm.lm_solution import LmSolution
from core.utils import ConfigDict
import openfst_python as fst
import unittest

INT_MAX = 2147483647

lm_cfg = ConfigDict()
lm_cfg.hotword_fst = ConfigDict()
lm_cfg.hotword_fst.lm_token_beam_size = 10
lm_cfg.hotword_fst.max_active_lm_token_num = 5000
lm_cfg.hotword_fst.enable_multi_step_search = True
lm_cfg.hotword_fst.hotword_weight = ''


class TestLmSolution(unittest.TestCase):
    """Unit test of lm solution"""

    # hotword: ab
    def _test_simple_case(self):
        """test a simple case"""
        f = fst.Fst()
        s0 = f.add_state()
        s1 = f.add_state()
        s2 = f.add_state()
        f.add_arc(s0, fst.Arc(10, 10, -10, s1))
        f.add_arc(s0, fst.Arc(INT_MAX, INT_MAX, 0.0, s0))
        f.add_arc(s1, fst.Arc(20, 20, -20, s0))
        f.add_arc(s1, fst.Arc(INT_MAX, INT_MAX, 10, s0))
        f.set_start(s0)
        f.set_final(s0, 0.0)
        f.write("tmp.fst")
        hotword_fst = HotwordFst(lm_cfg.hotword_fst)
        hotword_fst.load("tmp.fst")

        lm_cfg.hotword_fst.hotword_weight = '0.5'
        lm_solution = LmSolution(lm_cfg)
        lm_solution.hotword_fst_list.append(hotword_fst)

        hyp = Hypothesis()
        # input ab
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(20, hyp, False)
        self.assertEqual(15.0, hyp.score)

        # input ad
        lm_solution.step_hotword(10, hyp, False)
        self.assertEqual(20.0, hyp.score)
        lm_solution.step_hotword(30, hyp, False)
        self.assertEqual(15.0, hyp.score)

        # input a
        lm_solution.step_hotword(10, hyp, True)
        self.assertEqual(15.0, hyp.score)

    # hotword:ab bc cd, input abcd
    def _test_overlap_case(self):
        """test overlaped cases"""
        f = fst.Fst()
        s0 = f.add_state()
        s1 = f.add_state()
        s2 = f.add_state()
        s3 = f.add_state()
        f.add_arc(s0, fst.Arc(10, 10, -10, s1))
        f.add_arc(s0, fst.Arc(INT_MAX, INT_MAX, 0.0, s0))
        f.add_arc(s1, fst.Arc(INT_MAX, INT_MAX, 10, s0))
        f.add_arc(s1, fst.Arc(20, 20, -5, s0))  # ab
        f.add_arc(s0, fst.Arc(20, 20, -20, s2))
        f.add_arc(s2, fst.Arc(30, 30, -30, s0))
        f.add_arc(s2, fst.Arc(INT_MAX, INT_MAX, 20, s0))  # bc
        f.add_arc(s0, fst.Arc(30, 30, -30, s3))
        f.add_arc(s3, fst.Arc(40, 40, -40, s0))
        f.add_arc(s3, fst.Arc(INT_MAX, INT_MAX, 30, s0))  # cd

        f.set_start(s0)
        f.set_final(s0, 0.0)
        f.write("tmp.fst")
        hotword_fst = HotwordFst(lm_cfg.hotword_fst)
        hotword_fst.load("tmp.fst")

        lm_cfg.hotword_fst.hotword_weight = '1.0'
        lm_solution = LmSolution(lm_cfg)
        lm_solution.hotword_fst_list.append(hotword_fst)

        hyp = Hypothesis()
        # input abcd
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(20, hyp, False)
        self.assertEqual(20, hyp.score)
        lm_solution.step_hotword(30, hyp, False)
        lm_solution.step_hotword(40, hyp, False)
        self.assertEqual(85, hyp.score)  # ab + cd

        # input ab
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(20, hyp, True)
        self.assertEqual(hyp.score, 100)

        # input abd
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(20, hyp, False)
        lm_solution.step_hotword(40, hyp, False)
        self.assertEqual(hyp.score, 115)

        # input aab
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(20, hyp, False)
        self.assertEqual(hyp.score, 135)

    def _test_duplicated_words(self):
        """test duplicated hotwords"""
        f = fst.Fst()
        s0 = f.add_state()
        s1 = f.add_state()
        f.add_arc(s0, fst.Arc(INT_MAX, INT_MAX, 0.0, s0))
        f.add_arc(s0, fst.Arc(10, 10, -10, s1))
        f.add_arc(s1, fst.Arc(INT_MAX, INT_MAX, 10, s0))
        f.add_arc(s1, fst.Arc(10, 10, -10, s0))
        f.set_start(s0)
        f.set_final(s0, 0.0)
        f.write("tmp.fst")
        hotword_fst = HotwordFst(lm_cfg.hotword_fst)
        hotword_fst.load("tmp.fst")

        lm_cfg.hotword_fst.hotword_weight = '1.0'
        lm_solution = LmSolution(lm_cfg)
        lm_solution.hotword_fst_list.append(hotword_fst)

        hyp = Hypothesis()

        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(10, hyp, False)
        lm_solution.step_hotword(10, hyp, False)
        self.assertEqual(hyp.score, 30)

        lm_solution.step_hotword(20, hyp, False)
        self.assertEqual(20, hyp.score)


if __name__ == '__main__':
    unittest.main()
