#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Soft Decoder Example
# Author: Tom Rondeau
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

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio.filter import pfb
from gnuradio.qtgui import Range, RangeWidget
import coarsefreq
import helpers  # embedded python module
import numpy
import numpy as np

from gnuradio import qtgui

class hqam_txrx_hplp(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Soft Decoder Example")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Soft Decoder Example")
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

        self.settings = Qt.QSettings("GNU Radio", "hqam_txrx_hplp")

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
        self.modulated_sync_word = modulated_sync_word = [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1]
        self.sps = sps = 4
        self.rolloff = rolloff = 0.35
        self.nfilts = nfilts = 32
        self.corr_thresh = corr_thresh = 100000
        self.corr_max = corr_max = numpy.abs(numpy.dot(modulated_sync_word,numpy.conj(modulated_sync_word)))
        self.tx_gain_dB = tx_gain_dB = 10
        self.samp_rate = samp_rate = 1000000
        self.rx_gain_dB = rx_gain_dB = 10
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts,1.0,1.0/(nfilts*sps), rolloff, int(11*sps*nfilts))
        self.qpsk_preamble = qpsk_preamble = [0.7071 + 0.7071j, 0.7071 + 0.7071j, 0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, -0.7071 + 0.7071j, 0.7071 - 0.7071j,-0.7071 + 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, -0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j,-0.7071 - 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j, -0.7071 + 0.7071j,-0.7071 - 0.7071j, 0.7071 - 0.7071j,-0.7071 - 0.7071j, 0.7071 + 0.7071j, 0.7071 + 0.7071j, 0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j,-0.7071 + 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j,-0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j, -0.7071 - 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j,-0.7071 + 0.7071j, 0.7071 - 0.7071j,-0.7071 + 0.7071j,-0.7071 - 0.7071j, 0.7071 - 0.7071j, -0.7071 - 0.7071j]
        self.preamble = preamble = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        self.pi = pi = np.pi
        self.p_norm = p_norm = 4.70
        self.num_taps = num_taps = 64
        self.gain = gain = .001
        self.freq_offset = freq_offset = 3
        self.eb = eb = 0.35
        self.delay = delay = 58
        self.corr_calc = corr_calc = corr_thresh/(corr_max*corr_max)
        self.constel_lp = constel_lp = digital.constellation_calcdist([-3-3j, -3+3j, 3+3j, 3-3j], [0, 1, 3, 2],
        4, 1).base()
        self.constel_lp.gen_soft_dec_lut(8)
        self.constel_hp = constel_hp = digital.constellation_calcdist([-1-1j, -1+1j, 1+1j, 1-1j], [0, 1, 3, 2],
        4, 1).base()
        self.constel_hp.gen_soft_dec_lut(8)
        self.cons_norm = cons_norm = digital.constellation_calcdist([(-1-1j), (-0.5-1j), (0.5-1j), (1-1j), (-1-0.5j), (-0.5-0.5j), (0.5-0.5j), (1-0.5j), (-1+0.5j), (-0.5+0.5j), (0.5+0.5j), (1+0.5j), (-1+1j), (-0.5+1j), (0.5+1j), (1+1j)], [0, 4, 12, 8, 1, 5, 13, 9, 3, 7, 15, 11, 2, 6, 14, 10],
        4, 1).base()
        self.carrier_freq = carrier_freq = 2000e6
        self.c_norm = c_norm = 0.17
        self.arity = arity = 4
        self.agc_gain = agc_gain = 0.65

        ##################################################
        # Blocks
        ##################################################
        self._p_norm_range = Range(0.1, 10, 0.01, 4.70, 200)
        self._p_norm_win = RangeWidget(self._p_norm_range, self.set_p_norm, 'HQAM Preamble Amplitude', "counter_slider", float)
        self.top_grid_layout.addWidget(self._p_norm_win, 2, 2, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
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
        self.notebook.addTab(self.notebook_widget_1, 'Received Signal')
        self.top_grid_layout.addWidget(self.notebook, 1, 1, 8, 1)
        for r in range(1, 9):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._c_norm_range = Range(0.1, 2, 0.01, 0.17, 200)
        self._c_norm_win = RangeWidget(self._c_norm_range, self.set_c_norm, 'HQAM Envelope Amplitude', "counter_slider", float)
        self.top_grid_layout.addWidget(self._c_norm_win, 1, 2, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._agc_gain_range = Range(0.01, 10, 0.01, 0.65, 200)
        self._agc_gain_win = RangeWidget(self._agc_gain_range, self.set_agc_gain, 'AGC Gain', "counter_slider", float)
        self.top_grid_layout.addWidget(self._agc_gain_win, 3, 2, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
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
        self.qtgui_time_sink_x_2 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "Correlator Output", #name
            1 #number of inputs
        )
        self.qtgui_time_sink_x_2.set_update_time(1)
        self.qtgui_time_sink_x_2.set_y_axis(-2, 2)

        self.qtgui_time_sink_x_2.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_2.enable_tags(True)
        self.qtgui_time_sink_x_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_2.enable_autoscale(True)
        self.qtgui_time_sink_x_2.enable_grid(True)
        self.qtgui_time_sink_x_2.enable_axis_labels(True)
        self.qtgui_time_sink_x_2.enable_control_panel(False)
        self.qtgui_time_sink_x_2.enable_stem_plot(False)


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
                    self.qtgui_time_sink_x_2.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_2.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_2.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_2.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_2.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_2.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_2.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_2.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_2_win = sip.wrapinstance(self.qtgui_time_sink_x_2.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_2_win)
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
        self.qtgui_freq_sink_x_0.set_y_axis(-100, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
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
        self.qtgui_const_sink_x_2 = qtgui.const_sink_c(
            1000, #size
            "directly from poly", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_2.set_update_time(0.10)
        self.qtgui_const_sink_x_2.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_2.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_2.enable_autoscale(False)
        self.qtgui_const_sink_x_2.enable_grid(False)
        self.qtgui_const_sink_x_2.enable_axis_labels(True)


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
                self.qtgui_const_sink_x_2.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_2.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_2.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_2.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_2.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_2.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_2.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_2_win = sip.wrapinstance(self.qtgui_const_sink_x_2.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_2_win)
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
        self.qtgui_const_sink_x_1.enable_grid(False)
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
        self.notebook_layout_1.addWidget(self._qtgui_const_sink_x_1_win)
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
        self.notebook_layout_0.addWidget(self._qtgui_const_sink_x_0_win)
        self.pfb_arb_resampler_xxx_0 = pfb.arb_resampler_ccf(
            sps,
            taps=firdes.root_raised_cosine(nfilts, 1.0, 1.0/nfilts, rolloff, int(11*sps*nfilts)),
            flt_size=32)
        self.pfb_arb_resampler_xxx_0.declare_sample_delay(0)
        self._freq_offset_range = Range(0, 10, 0.1, 3, 10)
        self._freq_offset_win = RangeWidget(self._freq_offset_range, self.set_freq_offset, 'freq_offset', "counter_slider", float)
        self.top_grid_layout.addWidget(self._freq_offset_win, 4, 2, 1, 1)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.digital_scrambler_bb_0_0 = digital.scrambler_bb(0x8A, 0x7F, 8)
        self.digital_scrambler_bb_0 = digital.scrambler_bb(0x8A, 0x7F, 8)
        self.digital_pfb_clock_sync_xxx_0_0 = digital.pfb_clock_sync_ccf(sps, 6.28/100.0, rrc_taps, nfilts, nfilts/2, 1.5, 1)
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(sps, 6.28/100.0, rrc_taps, nfilts, nfilts/2, 1.5, 1)
        self.digital_chunks_to_symbols_xx_0_0_0 = digital.chunks_to_symbols_bc(constel_lp.points(), 1)
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc(constel_hp.points(), 1)
        self._corr_thresh_range = Range(0.1, 5e6, 1.0, 100000, 200)
        self._corr_thresh_win = RangeWidget(self._corr_thresh_range, self.set_corr_thresh, 'Absolute Corr Thresh (Mag Sq)', "counter_slider", float)
        self.top_grid_layout.addWidget(self._corr_thresh_win, 5, 2, 1, 1)
        for r in range(5, 6):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.coarsefreq_c_freq_offset_0 = coarsefreq.c_freq_offset(640, 15, samp_rate)
        self.blocks_vector_source_x_0_0_0 = blocks.vector_source_c(modulated_sync_word, True, 1, )
        self.blocks_unpack_k_bits_bb_1_0 = blocks.unpack_k_bits_bb(8)
        self.blocks_unpack_k_bits_bb_1 = blocks.unpack_k_bits_bb(8)
        self.blocks_stream_mux_0_0 = blocks.stream_mux(gr.sizeof_gr_complex*1, [128, 512])
        self.blocks_pack_k_bits_bb_0_1 = blocks.pack_k_bits_bb(2)
        self.blocks_pack_k_bits_bb_0 = blocks.pack_k_bits_bb(2)
        self.blocks_multiply_const_vxx_0_2 = blocks.multiply_const_cc(p_norm)
        self.blocks_multiply_const_vxx_0_1 = blocks.multiply_const_cc(1.4142)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_cc(2.8284)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(c_norm)
        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_cc(100, 1/3, 10, 1)
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_char*1, '/home/bhagat/txrx_demo_new/dct_lp_stream.dat', True, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/bhagat/txrx_demo_new/dct_hp_stream.dat', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, 64)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_agc_xx_0 = analog.agc_cc(1e-4, agc_gain, 1)
        self.analog_agc_xx_0.set_max_gain(65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.digital_pfb_clock_sync_xxx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_stream_mux_0_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_unpack_k_bits_bb_1, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_unpack_k_bits_bb_1_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.coarsefreq_c_freq_offset_0, 1))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.qtgui_time_sink_x_2, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.digital_pfb_clock_sync_xxx_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0_1, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_2, 0), (self.blocks_stream_mux_0_0, 0))
        self.connect((self.blocks_pack_k_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.blocks_pack_k_bits_bb_0_1, 0), (self.digital_chunks_to_symbols_xx_0_0_0, 0))
        self.connect((self.blocks_stream_mux_0_0, 0), (self.pfb_arb_resampler_xxx_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_1, 0), (self.digital_scrambler_bb_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_1_0, 0), (self.digital_scrambler_bb_0_0, 0))
        self.connect((self.blocks_vector_source_x_0_0_0, 0), (self.blocks_multiply_const_vxx_0_2, 0))
        self.connect((self.coarsefreq_c_freq_offset_0, 0), (self.qtgui_const_sink_x_1, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.blocks_multiply_const_vxx_0_1, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.coarsefreq_c_freq_offset_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.qtgui_const_sink_x_2, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.digital_scrambler_bb_0, 0), (self.blocks_pack_k_bits_bb_0, 0))
        self.connect((self.digital_scrambler_bb_0_0, 0), (self.blocks_pack_k_bits_bb_0_1, 0))
        self.connect((self.pfb_arb_resampler_xxx_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.analog_agc_xx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "hqam_txrx_hplp")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_modulated_sync_word(self):
        return self.modulated_sync_word

    def set_modulated_sync_word(self, modulated_sync_word):
        self.modulated_sync_word = modulated_sync_word
        self.set_corr_max(numpy.abs(numpy.dot(self.modulated_sync_word,numpy.conj(self.modulated_sync_word))))
        self.blocks_vector_source_x_0_0_0.set_data(self.modulated_sync_word, )

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

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts,1.0,1.0/(self.nfilts*self.sps), self.rolloff, int(11*self.sps*self.nfilts)))
        self.pfb_arb_resampler_xxx_0.set_taps(firdes.root_raised_cosine(self.nfilts, 1.0, 1.0/self.nfilts, self.rolloff, int(11*self.sps*self.nfilts)))

    def get_corr_thresh(self):
        return self.corr_thresh

    def set_corr_thresh(self, corr_thresh):
        self.corr_thresh = corr_thresh
        self.set_corr_calc(self.corr_thresh/(self.corr_max*self.corr_max))

    def get_corr_max(self):
        return self.corr_max

    def set_corr_max(self, corr_max):
        self.corr_max = corr_max
        self.set_corr_calc(self.corr_thresh/(self.corr_max*self.corr_max))

    def get_tx_gain_dB(self):
        return self.tx_gain_dB

    def set_tx_gain_dB(self, tx_gain_dB):
        self.tx_gain_dB = tx_gain_dB
        self.uhd_usrp_sink_0.set_gain(self.tx_gain_dB, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_2.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
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
        self.digital_pfb_clock_sync_xxx_0_0.update_taps(self.rrc_taps)

    def get_qpsk_preamble(self):
        return self.qpsk_preamble

    def set_qpsk_preamble(self, qpsk_preamble):
        self.qpsk_preamble = qpsk_preamble

    def get_preamble(self):
        return self.preamble

    def set_preamble(self, preamble):
        self.preamble = preamble

    def get_pi(self):
        return self.pi

    def set_pi(self, pi):
        self.pi = pi

    def get_p_norm(self):
        return self.p_norm

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm
        self.blocks_multiply_const_vxx_0_2.set_k(self.p_norm)

    def get_num_taps(self):
        return self.num_taps

    def set_num_taps(self, num_taps):
        self.num_taps = num_taps

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset

    def get_eb(self):
        return self.eb

    def set_eb(self, eb):
        self.eb = eb

    def get_delay(self):
        return self.delay

    def set_delay(self, delay):
        self.delay = delay

    def get_corr_calc(self):
        return self.corr_calc

    def set_corr_calc(self, corr_calc):
        self.corr_calc = corr_calc

    def get_constel_lp(self):
        return self.constel_lp

    def set_constel_lp(self, constel_lp):
        self.constel_lp = constel_lp

    def get_constel_hp(self):
        return self.constel_hp

    def set_constel_hp(self, constel_hp):
        self.constel_hp = constel_hp

    def get_cons_norm(self):
        return self.cons_norm

    def set_cons_norm(self, cons_norm):
        self.cons_norm = cons_norm

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.uhd_usrp_sink_0.set_center_freq(self.carrier_freq, 0)
        self.uhd_usrp_source_0.set_center_freq(self.carrier_freq, 0)

    def get_c_norm(self):
        return self.c_norm

    def set_c_norm(self, c_norm):
        self.c_norm = c_norm
        self.blocks_multiply_const_vxx_0.set_k(self.c_norm)

    def get_arity(self):
        return self.arity

    def set_arity(self, arity):
        self.arity = arity

    def get_agc_gain(self):
        return self.agc_gain

    def set_agc_gain(self, agc_gain):
        self.agc_gain = agc_gain
        self.analog_agc_xx_0.set_reference(self.agc_gain)





def main(top_block_cls=hqam_txrx_hplp, options=None):

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
