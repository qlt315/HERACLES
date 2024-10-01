#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Hierarchical QAM Mod / Demod Demonstration
# Author: Jeff Cuenco and Ashwini Bhagat
# Description: Explore Soft Decoding of constellations. Selec the constellation from the available objects.
# GNU Radio version: 3.8.5.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from PyQt5 import Qt
from gnuradio import qtgui
import display
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from hier_pack_byte_to_float import hier_pack_byte_to_float  # grc-generated hier_block
import checkevm
import dct
import dd_pll
import fll_est
import numpy as np

from gnuradio import qtgui

class hqam_rx_only(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Hierarchical QAM Mod / Demod Demonstration")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Hierarchical QAM Mod / Demod Demonstration")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "hqam_rx_only")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 4
        self.rolloff = rolloff = 0.35
        self.qpsk_preamble = qpsk_preamble = [(-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (-0.7071067690849304+0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j), (0.7071067690849304+0.7071067690849304j), (0.7071067690849304-0.7071067690849304j), (-0.7071067690849304-0.7071067690849304j)]
        self.num_tag_key = num_tag_key = "packet_num"
        self.nfilts = nfilts = 32
        self.len_tag_key = len_tag_key = "packet_length"
        self.header_len = header_len = 32
        self.scrambler_seed = scrambler_seed = 0x7FFF
        self.scrambler_mask = scrambler_mask = 106513
        self.scrambler_len = scrambler_len = 15
        self.samp_rate = samp_rate = 500000
        self.rx_gain_dB = rx_gain_dB = 20
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts,1.0,1.0/(nfilts*sps), rolloff, int(11*sps*nfilts))
        self.qpsk_preamble_zp = qpsk_preamble_zp = np.concatenate((qpsk_preamble, [0,0,0,0,0,0,0,0]))
        self.qpsk_const = qpsk_const = digital.constellation_calcdist(digital.psk_4()[0], digital.psk_4()[1],
        4, 1).base()
        self.qpsk_const.gen_soft_dec_lut(8)
        self.header_formatter = header_formatter = digital.packet_header_default(header_len, len_tag_key,num_tag_key,1)
        self.carrier_freq = carrier_freq = 2000e6
        self.agc_gain = agc_gain = 0.82

        ##################################################
        # Blocks
        ##################################################
        self.notebook = Qt.QTabWidget()
        self.notebook_widget_0 = Qt.QWidget()
        self.notebook_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.notebook_widget_0)
        self.notebook_grid_layout_0 = Qt.QGridLayout()
        self.notebook_layout_0.addLayout(self.notebook_grid_layout_0)
        self.notebook.addTab(self.notebook_widget_0, 'Received Signal')
        self.notebook_widget_1 = Qt.QWidget()
        self.notebook_layout_1 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.notebook_widget_1)
        self.notebook_grid_layout_1 = Qt.QGridLayout()
        self.notebook_layout_1.addLayout(self.notebook_grid_layout_1)
        self.notebook.addTab(self.notebook_widget_1, 'Received Image')
        self.top_grid_layout.addWidget(self.notebook, 1, 1, 8, 1)
        for r in range(1, 9):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=[1],
            ),
        )
        self.uhd_usrp_source_0.set_center_freq(carrier_freq, 0)
        self.uhd_usrp_source_0.set_gain(rx_gain_dB, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        self.show_image_0 = display.show_image( 512, 512  )
        self.show_image_0.displayBottomUp(False)
        self._show_image_0_win = sip.wrapinstance(self.show_image_0.pyqwidget(), Qt.QWidget)
        self.notebook_grid_layout_1.addWidget(self._show_image_0_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.notebook_grid_layout_1.setRowStretch(r, 1)
        for c in range(0, 1):
            self.notebook_grid_layout_1.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_1 = qtgui.const_sink_c(
            1000, #size
            'RX Constellation', #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_1.set_update_time(0.10)
        self.qtgui_const_sink_x_1.set_y_axis(-1.2, 1.2)
        self.qtgui_const_sink_x_1.set_x_axis(-1.2, 1.2)
        self.qtgui_const_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_1.enable_autoscale(False)
        self.qtgui_const_sink_x_1.enable_grid(True)
        self.qtgui_const_sink_x_1.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_1_win = sip.wrapinstance(self.qtgui_const_sink_x_1.pyqwidget(), Qt.QWidget)
        self.notebook_grid_layout_0.addWidget(self._qtgui_const_sink_x_1_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.notebook_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.notebook_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0_0_0 = qtgui.const_sink_c(
            1024, #size
            "After Header/Payload Demux Constellation", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0_0.set_y_axis(-1.2, 1.2)
        self.qtgui_const_sink_x_0_0_0.set_x_axis(-1.2, 1.2)
        self.qtgui_const_sink_x_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0_0.enable_grid(True)
        self.qtgui_const_sink_x_0_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 1, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, -1, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 0.5, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_0.pyqwidget(), Qt.QWidget)
        self.notebook_grid_layout_0.addWidget(self._qtgui_const_sink_x_0_0_0_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.notebook_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.notebook_grid_layout_0.setColumnStretch(c, 1)
        self.hier_pack_byte_to_float_0_0 = hier_pack_byte_to_float(
            scaling_const=1000,
        )
        self.hier_pack_byte_to_float_0 = hier_pack_byte_to_float(
            scaling_const=1000,
        )
        self.fll_est_my_fll_0 = fll_est.my_fll(qpsk_preamble_zp, 1, 129, 0.7, digital.THRESHOLD_ABSOLUTE)
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(sps, 6.28/100.0, rrc_taps, nfilts, nfilts/2, 1.5, 1)
        self.digital_packet_headerparser_b_0_0 = digital.packet_headerparser_b(header_formatter.formatter())
        self.digital_map_bb_0 = digital.map_bb(qpsk_const.pre_diff_code())
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            int(header_len/2),
            1,
            0,
            len_tag_key,
            "corr_est",
            False,
            gr.sizeof_gr_complex,
            "rx_time",
            samp_rate,
            (),
            0)
        self.digital_descrambler_bb_0_0 = digital.descrambler_bb(scrambler_mask, scrambler_seed, scrambler_len)
        self.digital_descrambler_bb_0 = digital.descrambler_bb(scrambler_mask, scrambler_seed, scrambler_len)
        self.digital_constellation_decoder_cb_1 = digital.constellation_decoder_cb(qpsk_const)
        self.dd_pll_qam_pll_0 = dd_pll.qam_pll(0.7071,1000,samp_rate)
        self.dct_dct_ff_0_0_0_0 = dct.dct_ff(512, 512, 1)
        self.dct_dct_ff_0_0_0 = dct.dct_ff(512, 512, 1)
        self.checkevm_getevm_0 = checkevm.getevm(0.7)
        self.blocks_tag_gate_0 = blocks.tag_gate(gr.sizeof_gr_complex * 1, False)
        self.blocks_tag_gate_0.set_single_key("")
        self.blocks_skiphead_0_0 = blocks.skiphead(gr.sizeof_char*1, 2)
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_char*1, 2)
        self.blocks_repack_bits_bb_0_1_0 = blocks.repack_bits_bb(1, 8, "", False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_0_1 = blocks.repack_bits_bb(1, 8, "", False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_0_0_0 = blocks.repack_bits_bb(2, 1, "", False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_0_0 = blocks.repack_bits_bb(2, 1, "", False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(2, 1, "", False, gr.GR_MSB_FIRST)
        self.blocks_multiply_const_vxx_1_0 = blocks.multiply_const_ff(1)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_ff(1)
        self.blocks_float_to_char_0 = blocks.float_to_char(1, 1)
        self.blocks_add_xx_1 = blocks.add_vff(1)
        self.analog_sig_source_x_0 = analog.sig_source_b(samp_rate, analog.GR_CONST_WAVE, 60, 1, 0, 0)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(-40, 1e-4, 0, True)
        self.analog_agc_xx_0 = analog.agc_cc(1e-4, agc_gain, 1)
        self.analog_agc_xx_0.set_max_gain(65536)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.digital_packet_headerparser_b_0_0, 'header_data'), (self.digital_header_payload_demux_0, 'header_data'))
        self.connect((self.analog_agc_xx_0, 0), (self.digital_pfb_clock_sync_xxx_0, 0))
        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.analog_agc_xx_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.show_image_0, 1))
        self.connect((self.blocks_add_xx_1, 0), (self.blocks_float_to_char_0, 0))
        self.connect((self.blocks_float_to_char_0, 0), (self.show_image_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.dct_dct_ff_0_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1_0, 0), (self.dct_dct_ff_0_0_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_descrambler_bb_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0, 0), (self.digital_descrambler_bb_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0_0, 0), (self.digital_packet_headerparser_b_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_1, 0), (self.blocks_skiphead_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_1_0, 0), (self.blocks_skiphead_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.hier_pack_byte_to_float_0_0, 0))
        self.connect((self.blocks_skiphead_0_0, 0), (self.hier_pack_byte_to_float_0, 0))
        self.connect((self.blocks_tag_gate_0, 0), (self.checkevm_getevm_0, 0))
        self.connect((self.checkevm_getevm_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.checkevm_getevm_0, 1), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.dct_dct_ff_0_0_0, 0), (self.blocks_add_xx_1, 0))
        self.connect((self.dct_dct_ff_0_0_0_0, 0), (self.blocks_add_xx_1, 1))
        self.connect((self.dd_pll_qam_pll_0, 0), (self.fll_est_my_fll_0, 0))
        self.connect((self.digital_constellation_decoder_cb_1, 0), (self.digital_map_bb_0, 0))
        self.connect((self.digital_descrambler_bb_0, 0), (self.blocks_repack_bits_bb_0_1, 0))
        self.connect((self.digital_descrambler_bb_0_0, 0), (self.blocks_repack_bits_bb_0_1_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.blocks_tag_gate_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.digital_constellation_decoder_cb_1, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.qtgui_const_sink_x_0_0_0, 0))
        self.connect((self.digital_map_bb_0, 0), (self.blocks_repack_bits_bb_0_0_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.dd_pll_qam_pll_0, 0))
        self.connect((self.fll_est_my_fll_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.fll_est_my_fll_0, 0), (self.qtgui_const_sink_x_1, 0))
        self.connect((self.hier_pack_byte_to_float_0, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.hier_pack_byte_to_float_0_0, 0), (self.blocks_multiply_const_vxx_1_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.analog_pwr_squelch_xx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "hqam_rx_only")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts,1.0,1.0/(self.nfilts*self.sps), self.rolloff, int(11*self.sps*self.nfilts)))

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts,1.0,1.0/(self.nfilts*self.sps), self.rolloff, int(11*self.sps*self.nfilts)))

    def get_qpsk_preamble(self):
        return self.qpsk_preamble

    def set_qpsk_preamble(self, qpsk_preamble):
        self.qpsk_preamble = qpsk_preamble
        self.set_qpsk_preamble_zp(np.concatenate((self.qpsk_preamble, [0,0,0,0,0,0,0,0])))

    def get_num_tag_key(self):
        return self.num_tag_key

    def set_num_tag_key(self, num_tag_key):
        self.num_tag_key = num_tag_key
        self.set_header_formatter(digital.packet_header_default(self.header_len, self.len_tag_key,self.num_tag_key,1))

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts,1.0,1.0/(self.nfilts*self.sps), self.rolloff, int(11*self.sps*self.nfilts)))

    def get_len_tag_key(self):
        return self.len_tag_key

    def set_len_tag_key(self, len_tag_key):
        self.len_tag_key = len_tag_key
        self.set_header_formatter(digital.packet_header_default(self.header_len, self.len_tag_key,self.num_tag_key,1))

    def get_header_len(self):
        return self.header_len

    def set_header_len(self, header_len):
        self.header_len = header_len
        self.set_header_formatter(digital.packet_header_default(self.header_len, self.len_tag_key,self.num_tag_key,1))

    def get_scrambler_seed(self):
        return self.scrambler_seed

    def set_scrambler_seed(self, scrambler_seed):
        self.scrambler_seed = scrambler_seed

    def get_scrambler_mask(self):
        return self.scrambler_mask

    def set_scrambler_mask(self, scrambler_mask):
        self.scrambler_mask = scrambler_mask

    def get_scrambler_len(self):
        return self.scrambler_len

    def set_scrambler_len(self, scrambler_len):
        self.scrambler_len = scrambler_len

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rx_gain_dB(self):
        return self.rx_gain_dB

    def set_rx_gain_dB(self, rx_gain_dB):
        self.rx_gain_dB = rx_gain_dB
        self.uhd_usrp_source_0.set_gain(self.rx_gain_dB, 0)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.digital_pfb_clock_sync_xxx_0.update_taps(self.rrc_taps)

    def get_qpsk_preamble_zp(self):
        return self.qpsk_preamble_zp

    def set_qpsk_preamble_zp(self, qpsk_preamble_zp):
        self.qpsk_preamble_zp = qpsk_preamble_zp

    def get_qpsk_const(self):
        return self.qpsk_const

    def set_qpsk_const(self, qpsk_const):
        self.qpsk_const = qpsk_const

    def get_header_formatter(self):
        return self.header_formatter

    def set_header_formatter(self, header_formatter):
        self.header_formatter = header_formatter

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.uhd_usrp_source_0.set_center_freq(self.carrier_freq, 0)

    def get_agc_gain(self):
        return self.agc_gain

    def set_agc_gain(self, agc_gain):
        self.agc_gain = agc_gain
        self.analog_agc_xx_0.set_reference(self.agc_gain)





def main(top_block_cls=hqam_rx_only, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()

if __name__ == '__main__':
    main()
