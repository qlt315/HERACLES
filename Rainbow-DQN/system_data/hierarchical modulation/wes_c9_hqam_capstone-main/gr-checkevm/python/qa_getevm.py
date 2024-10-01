#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Ashwini Bhagat.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
import checkevm_swig as checkevm

class qa_getevm(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_t(self):
        # set up fg
        src_data = (0.0432468460658841-0.0109829160911651j,0.0253696906390455+0.00069136426725247j,0.0524572702806811+0.0505473235294273j,-0.0556690863898125+0.0567273053195921j,-0.00709915341168951-0.0363518068711361j,-0.0813816203316941-9.99752351932692e-05j,-0.0281885441030595-0.0228338447495401j,-0.0259665412262302+0.00942060444704033j,-0.0198598596288435+0.0307794313363699j,-0.0149895883809178+0.0504361556887989j,-0.0575913592322312-0.040801557956935j,0.00453285713215826-0.00770695281990838j,-0.0392837311981496-0.0391699639004847j,0.0339490479983139-0.0427146670740614j,0.0779692830982996-0.00134745118744738j);
        src = blocks.vector_source_c(src_data)
        sqr = checkevm.getevm(0.4)
        dst_hp = blocks.vector_sink_i()
        dst_lp = blocks.vector_sink_i()
        self.tb.connect(src, sqr)
        self.tb.connect((sqr,0), dst_hp)
        self.tb.connect((sqr,1), dst_lp)
        self.tb.run()
        # check data
        result_data1 = dst_hp.data()
        result_data2 = dst_lp.data()
        print(src_data)
        #print(expected_result)
        print(result_data1)
        print(result_data2)
        
        #self.assertFloatTuplesAlmostEqual(expected_result, result_data, 5)


if __name__ == '__main__':
    gr_unittest.run(qa_getevm)
