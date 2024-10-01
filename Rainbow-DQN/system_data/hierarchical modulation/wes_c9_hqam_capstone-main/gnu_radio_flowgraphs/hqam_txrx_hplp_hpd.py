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
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio.filter import pfb
from hier_hqam_build_hdr import hier_hqam_build_hdr  # grc-generated hier_block
from hqam_enc_hp_stream import hqam_enc_hp_stream  # grc-generated hier_block
from hqam_enc_lp_stream import hqam_enc_lp_stream  # grc-generated hier_block
import numpy as np

from gnuradio import qtgui

class hqam_txrx_hplp_hpd(gr.top_block, Qt.QWidget):

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

        self.settings = Qt.QSettings("GNU Radio", "hqam_txrx_hplp_hpd")

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
        self.num_tag_key = num_tag_key = "packet_num"
        self.nfilts = nfilts = 32
        self.len_tag_key = len_tag_key = "packet_length"
        self.header_len = header_len = 32
        self.tx_gain_dB = tx_gain_dB = 10
        self.scrambler_seed = scrambler_seed = 0x7FFF
        self.scrambler_mask = scrambler_mask = 106513
        self.scrambler_len = scrambler_len = 15
        self.samp_rate = samp_rate = 500000
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts,1.0,1.0/(nfilts*sps), rolloff, int(11*sps*nfilts))
        self.qpsk_const = qpsk_const = digital.constellation_calcdist(digital.psk_4()[0], digital.psk_4()[1],
        4, 1).base()
        self.qpsk_const.gen_soft_dec_lut(8)
        self.preamble = preamble = [0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,1,1,0,1,0,0]
        self.header_formatter = header_formatter = digital.packet_header_default(header_len, len_tag_key,num_tag_key,1)
        self.carrier_freq = carrier_freq = 2000e6

        ##################################################
        # Blocks
        ##################################################
        self.notebook = Qt.QTabWidget()
        self.notebook_widget_0 = Qt.QWidget()
        self.notebook_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.notebook_widget_0)
        self.notebook_grid_layout_0 = Qt.QGridLayout()
        self.notebook_layout_0.addLayout(self.notebook_grid_layout_0)
        self.notebook.addTab(self.notebook_widget_0, 'Transmitted Signal')
        self.notebook_widget_1 = Qt.QWidget()
        self.notebook_layout_1 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.notebook_widget_1)
        self.notebook_grid_layout_1 = Qt.QGridLayout()
        self.notebook_layout_1.addLayout(self.notebook_grid_layout_1)
        self.notebook.addTab(self.notebook_widget_1, 'Transmitted Constellation')
        self.top_grid_layout.addWidget(self.notebook, 1, 1, 8, 1)
        for r in range(1, 9):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            '',
        )
        self.uhd_usrp_sink_0.set_center_freq(carrier_freq, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain_dB, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec())
        self.qtgui_time_sink_x_1 = qtgui.time_sink_c(
            200, #size
            samp_rate, #samp_rate
            "TX Time Series", #name
            1 #number of inputs
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-1.2, 1.2)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(True)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.pyqwidget(), Qt.QWidget)
        self.notebook_layout_0.addWidget(self._qtgui_time_sink_x_1_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "TX Spectrum", #name
            1
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-100, 0)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(True)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.pyqwidget(), Qt.QWidget)
        self.notebook_layout_0.addWidget(self._qtgui_freq_sink_x_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "TX Constellation", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-1.2, 1.2)
        self.qtgui_const_sink_x_0.set_x_axis(-1.2, 1.2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


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
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.pyqwidget(), Qt.QWidget)
        self.notebook_layout_1.addWidget(self._qtgui_const_sink_x_0_win)
        self.pfb_arb_resampler_xxx_0 = pfb.arb_resampler_ccf(
            sps,
            taps=firdes.root_raised_cosine(nfilts, 1.0, 1.0/nfilts, rolloff, int(11*sps*nfilts)),
            flt_size=32)
        self.pfb_arb_resampler_xxx_0.declare_sample_delay(0)
        self.hqam_enc_lp_stream_0 = hqam_enc_lp_stream(
            scrambler_len=scrambler_len,
            scrambler_mask=scrambler_mask,
            scrambler_seed=scrambler_seed,
        )
        self.hqam_enc_hp_stream_0 = hqam_enc_hp_stream(
            scrambler_len=scrambler_len,
            scrambler_mask=scrambler_mask,
            scrambler_seed=scrambler_seed,
        )
        self.hier_hqam_build_hdr_0 = hier_hqam_build_hdr(
            const_sym_table=qpsk_const.points(),
            payload_len=512,
            preamble=preamble,
        )
        self.digital_pfb_clock_sync_xxx_0_0 = digital.pfb_clock_sync_ccf(sps, 6.28/100.0, rrc_taps, nfilts, nfilts/2, 1.5, 1)
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, 'packet_length', 0)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, int((len(preamble) + header_len)/2), "packet_length")
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, 512, "packet_length")
        self.blocks_multiply_const_vxx_0_2 = blocks.multiply_const_cc(4.2)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.17)
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_char*1, '/home/wes/projects/capstone/hqam_modulation/GNU Radio /archive/dct_lp_stream.dat', True, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/wes/projects/capstone/hqam_modulation/GNU Radio /archive/dct_hp_stream.dat', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_add_xx_0 = blocks.add_vcc(1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.hqam_enc_hp_stream_0, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.hqam_enc_lp_stream_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.digital_pfb_clock_sync_xxx_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_2, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.blocks_multiply_const_vxx_0_2, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.pfb_arb_resampler_xxx_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.hier_hqam_build_hdr_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.hqam_enc_hp_stream_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.hqam_enc_lp_stream_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.pfb_arb_resampler_xxx_0, 0), (self.blocks_multiply_const_vxx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "hqam_txrx_hplp_hpd")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts,1.0,1.0/(self.nfilts*self.sps), self.rolloff, int(11*self.sps*self.nfilts)))
        self.pfb_arb_resampler_xxx_0.set_taps(firdes.root_raised_cosine(self.nfilts, 1.0, 1.0/self.nfilts, self.rolloff, int(11*self.sps*self.nfilts)))
        self.pfb_arb_resampler_xxx_0.set_rate(self.sps)

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts,1.0,1.0/(self.nfilts*self.sps), self.rolloff, int(11*self.sps*self.nfilts)))
        self.pfb_arb_resampler_xxx_0.set_taps(firdes.root_raised_cosine(self.nfilts, 1.0, 1.0/self.nfilts, self.rolloff, int(11*self.sps*self.nfilts)))

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
        self.pfb_arb_resampler_xxx_0.set_taps(firdes.root_raised_cosine(self.nfilts, 1.0, 1.0/self.nfilts, self.rolloff, int(11*self.sps*self.nfilts)))

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
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len(int((len(self.preamble) + self.header_len)/2))
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len_pmt(int((len(self.preamble) + self.header_len)/2))

    def get_tx_gain_dB(self):
        return self.tx_gain_dB

    def set_tx_gain_dB(self, tx_gain_dB):
        self.tx_gain_dB = tx_gain_dB
        self.uhd_usrp_sink_0.set_gain(self.tx_gain_dB, 0)

    def get_scrambler_seed(self):
        return self.scrambler_seed

    def set_scrambler_seed(self, scrambler_seed):
        self.scrambler_seed = scrambler_seed
        self.hqam_enc_hp_stream_0.set_scrambler_seed(self.scrambler_seed)
        self.hqam_enc_lp_stream_0.set_scrambler_seed(self.scrambler_seed)

    def get_scrambler_mask(self):
        return self.scrambler_mask

    def set_scrambler_mask(self, scrambler_mask):
        self.scrambler_mask = scrambler_mask
        self.hqam_enc_hp_stream_0.set_scrambler_mask(self.scrambler_mask)
        self.hqam_enc_lp_stream_0.set_scrambler_mask(self.scrambler_mask)

    def get_scrambler_len(self):
        return self.scrambler_len

    def set_scrambler_len(self, scrambler_len):
        self.scrambler_len = scrambler_len
        self.hqam_enc_hp_stream_0.set_scrambler_len(self.scrambler_len)
        self.hqam_enc_lp_stream_0.set_scrambler_len(self.scrambler_len)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.digital_pfb_clock_sync_xxx_0_0.update_taps(self.rrc_taps)

    def get_qpsk_const(self):
        return self.qpsk_const

    def set_qpsk_const(self, qpsk_const):
        self.qpsk_const = qpsk_const

    def get_preamble(self):
        return self.preamble

    def set_preamble(self, preamble):
        self.preamble = preamble
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len(int((len(self.preamble) + self.header_len)/2))
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len_pmt(int((len(self.preamble) + self.header_len)/2))
        self.hier_hqam_build_hdr_0.set_preamble(self.preamble)

    def get_header_formatter(self):
        return self.header_formatter

    def set_header_formatter(self, header_formatter):
        self.header_formatter = header_formatter

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.uhd_usrp_sink_0.set_center_freq(self.carrier_freq, 0)





def main(top_block_cls=hqam_txrx_hplp_hpd, options=None):

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
