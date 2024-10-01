#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: DCT HP/LP Split Test
# Author: Jeff Cuenco
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
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from hier_pack_float_to_byte import hier_pack_float_to_byte  # grc-generated hier_block
from idct_dec_bytes import idct_dec_bytes  # grc-generated hier_block
import dct
import hqam

from gnuradio import qtgui

class dct_hplp_test(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "DCT HP/LP Split Test")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("DCT HP/LP Split Test")
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

        self.settings = Qt.QSettings("GNU Radio", "dct_hplp_test")

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
        self.variable_qtgui_range_0 = variable_qtgui_range_0 = 50
        self.scaling_const = scaling_const = 1000
        self.samp_rate = samp_rate = 32000
        self.dct_width = dct_width = 512
        self.dct_height = dct_height = 512

        ##################################################
        # Blocks
        ##################################################
        self._scaling_const_range = Range(0, 2000, 1, 1000, 200)
        self._scaling_const_win = RangeWidget(self._scaling_const_range, self.set_scaling_const, 'Post-DCT  Float to Fixed Point Scaling Constant', "counter_slider", float)
        self.top_layout.addWidget(self._scaling_const_win)
        self._variable_qtgui_range_0_range = Range(0, 100, 1, 50, 200)
        self._variable_qtgui_range_0_win = RangeWidget(self._variable_qtgui_range_0_range, self.set_variable_qtgui_range_0, 'variable_qtgui_range_0', "counter_slider", float)
        self.top_layout.addWidget(self._variable_qtgui_range_0_win)
        self.show_image_0_0_0 = display.show_image( 512, 512  )
        self.show_image_0_0_0.displayBottomUp(False)
        self._show_image_0_0_0_win = sip.wrapinstance(self.show_image_0_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._show_image_0_0_0_win)
        self.show_image_0_0 = display.show_image( 512, 512  )
        self.show_image_0_0.displayBottomUp(False)
        self._show_image_0_0_win = sip.wrapinstance(self.show_image_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._show_image_0_0_win)
        self.show_image_0 = display.show_image( 512, 512  )
        self.show_image_0.displayBottomUp(False)
        self._show_image_0_win = sip.wrapinstance(self.show_image_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._show_image_0_win)
        self.idct_dec_bytes_0_0 = idct_dec_bytes(
            idct_height=dct_width,
            idct_width=dct_height,
            scaling_const=1000,
        )
        self.idct_dec_bytes_0 = idct_dec_bytes(
            idct_height=dct_width,
            idct_width=dct_height,
            scaling_const=1000,
        )
        self.hqam_split_py_ff_0 = hqam.split_py_ff(dct_width, dct_height)
        self.hier_pack_float_to_byte_0_0 = hier_pack_float_to_byte(
            scaling_const=scaling_const,
        )
        self.hier_pack_float_to_byte_0 = hier_pack_float_to_byte(
            scaling_const=scaling_const,
        )
        self.dct_dct_ff_0_0 = dct.dct_ff(dct_width, dct_height, 0)
        self.blocks_float_to_char_0 = blocks.float_to_char(1, 1)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/wes/projects/capstone/hqam_modulation/GNU Radio /Lenna_raw_gray.dat', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_char_to_float_1_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_1 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0 = blocks.char_to_float(1, 1)
        self.blocks_add_xx_0 = blocks.add_vff(1)
        self.analog_sig_source_x_0_0_0 = analog.sig_source_b(samp_rate, analog.GR_CONST_WAVE, 60, 1, 0, 0)
        self.analog_sig_source_x_0_0 = analog.sig_source_b(samp_rate, analog.GR_CONST_WAVE, 60, 1, 0, 0)
        self.analog_sig_source_x_0 = analog.sig_source_b(samp_rate, analog.GR_CONST_WAVE, 60, 1, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.show_image_0, 1))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.show_image_0_0, 1))
        self.connect((self.analog_sig_source_x_0_0_0, 0), (self.show_image_0_0_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_float_to_char_0, 0))
        self.connect((self.blocks_char_to_float_0, 0), (self.dct_dct_ff_0_0, 0))
        self.connect((self.blocks_char_to_float_1, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_char_to_float_1_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_char_to_float_0, 0))
        self.connect((self.blocks_float_to_char_0, 0), (self.show_image_0_0_0, 0))
        self.connect((self.dct_dct_ff_0_0, 0), (self.hqam_split_py_ff_0, 0))
        self.connect((self.hier_pack_float_to_byte_0, 0), (self.idct_dec_bytes_0, 0))
        self.connect((self.hier_pack_float_to_byte_0_0, 0), (self.idct_dec_bytes_0_0, 0))
        self.connect((self.hqam_split_py_ff_0, 0), (self.hier_pack_float_to_byte_0, 0))
        self.connect((self.hqam_split_py_ff_0, 1), (self.hier_pack_float_to_byte_0_0, 0))
        self.connect((self.idct_dec_bytes_0, 0), (self.blocks_char_to_float_1, 0))
        self.connect((self.idct_dec_bytes_0, 0), (self.show_image_0, 0))
        self.connect((self.idct_dec_bytes_0_0, 0), (self.blocks_char_to_float_1_0, 0))
        self.connect((self.idct_dec_bytes_0_0, 0), (self.show_image_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "dct_hplp_test")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_variable_qtgui_range_0(self):
        return self.variable_qtgui_range_0

    def set_variable_qtgui_range_0(self, variable_qtgui_range_0):
        self.variable_qtgui_range_0 = variable_qtgui_range_0

    def get_scaling_const(self):
        return self.scaling_const

    def set_scaling_const(self, scaling_const):
        self.scaling_const = scaling_const
        self.hier_pack_float_to_byte_0.set_scaling_const(self.scaling_const)
        self.hier_pack_float_to_byte_0_0.set_scaling_const(self.scaling_const)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_0_0.set_sampling_freq(self.samp_rate)

    def get_dct_width(self):
        return self.dct_width

    def set_dct_width(self, dct_width):
        self.dct_width = dct_width
        self.idct_dec_bytes_0.set_idct_height(self.dct_width)
        self.idct_dec_bytes_0_0.set_idct_height(self.dct_width)

    def get_dct_height(self):
        return self.dct_height

    def set_dct_height(self, dct_height):
        self.dct_height = dct_height
        self.idct_dec_bytes_0.set_idct_width(self.dct_height)
        self.idct_dec_bytes_0_0.set_idct_width(self.dct_height)





def main(top_block_cls=dct_hplp_test, options=None):

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
