#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: DCT Test
# Author: Jeff Cuenco
# Copyright: 2022 Zammerstein Enterprises
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
import display
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from hier_gen_hplp_layers import hier_gen_hplp_layers  # grc-generated hier_block
from hier_pack_byte_to_float import hier_pack_byte_to_float  # grc-generated hier_block
import dct

from gnuradio import qtgui

class dct_test(gr.top_block, Qt.QWidget):

    def __init__(self, dct_height=512, dct_width=512):
        gr.top_block.__init__(self, "DCT Test")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("DCT Test")
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

        self.settings = Qt.QSettings("GNU Radio", "dct_test")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Parameters
        ##################################################
        self.dct_height = dct_height
        self.dct_width = dct_width

        ##################################################
        # Variables
        ##################################################
        self.scrambler_mask = scrambler_mask = 0x9
        self.scrambler_len = scrambler_len = 17
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.show_image_0 = display.show_image( 512, 512  )
        self.show_image_0.displayBottomUp(False)
        self._show_image_0_win = sip.wrapinstance(self.show_image_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._show_image_0_win)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1 #number of inputs
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-1, 1)

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


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_win)
        self.hier_pack_byte_to_float_1_0 = hier_pack_byte_to_float(
            scaling_const=1000,
        )
        self.hier_pack_byte_to_float_1 = hier_pack_byte_to_float(
            scaling_const=1000,
        )
        self.hier_gen_hplp_layers_0 = hier_gen_hplp_layers(
            dct_height=512,
            dct_width=512,
            scaling_const=1000,
        )
        self.dct_dct_ff_0_0_0_0 = dct.dct_ff(512, 512, 1)
        self.dct_dct_ff_0_0_0 = dct.dct_ff(512, 512, 1)
        self.blocks_float_to_char_0 = blocks.float_to_char(1, 1)
        self.blocks_add_xx_0 = blocks.add_vff(1)
        self.analog_sig_source_x_0 = analog.sig_source_b(samp_rate, analog.GR_SQR_WAVE, 1000, 1, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.show_image_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_float_to_char_0, 0))
        self.connect((self.blocks_float_to_char_0, 0), (self.show_image_0, 0))
        self.connect((self.dct_dct_ff_0_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.dct_dct_ff_0_0_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.hier_gen_hplp_layers_0, 0), (self.hier_pack_byte_to_float_1, 0))
        self.connect((self.hier_gen_hplp_layers_0, 1), (self.hier_pack_byte_to_float_1_0, 0))
        self.connect((self.hier_pack_byte_to_float_1, 0), (self.dct_dct_ff_0_0_0, 0))
        self.connect((self.hier_pack_byte_to_float_1, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.hier_pack_byte_to_float_1_0, 0), (self.dct_dct_ff_0_0_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "dct_test")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_dct_height(self):
        return self.dct_height

    def set_dct_height(self, dct_height):
        self.dct_height = dct_height

    def get_dct_width(self):
        return self.dct_width

    def set_dct_width(self, dct_width):
        self.dct_width = dct_width

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
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)




def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--dct-height", dest="dct_height", type=intx, default=512,
        help="Set DCT Block Height (Rows) [default=%(default)r]")
    parser.add_argument(
        "--dct-width", dest="dct_width", type=intx, default=512,
        help="Set DCT Block Width (Cols) [default=%(default)r]")
    return parser


def main(top_block_cls=dct_test, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(dct_height=options.dct_height, dct_width=options.dct_width)

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
