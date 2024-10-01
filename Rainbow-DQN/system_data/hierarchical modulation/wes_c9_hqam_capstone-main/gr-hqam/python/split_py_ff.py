#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Jeff Cuenco.
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


import numpy
from gnuradio import gr

class split_py_ff(gr.basic_block):
    """
    docstring for block split_py_ff
    """
    def __init__(self, width, height):
        # [JC] Assigning width, height, and imgsize variables
        self.width = width
        self.height = height
        self.imgsize = width * height
                   
        gr.basic_block.__init__(self,
            name="split_py_ff",
            in_sig=[numpy.float32],
            out_sig=[numpy.float32, numpy.float32])

        # [JC] Ensure full image data is loaded prior to doing stream split
        self.set_output_multiple(self.imgsize)
		
		# [JC] We have two outputs to one input, so we have "interpolation" of 2
        self.set_relative_rate(2.0)

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = noutput_items

    def general_work(self, input_items, output_items):
        img = numpy.frombuffer(input_items[0], dtype=numpy.float32)       
		
        if (len(img) > (self.width * self.height)):
            img = img[0:(self.width * self.height)]
            
        img = img.reshape((self.width, self.height))
        #print(self.width)
        #print(self.height)
        #print(img.dtype)
        #print(img.shape)
        #print(img.size)
        
        # [JC] Calculate cutoff for upper triangular / lower triangular split
        cutoff = 0.5*len(img)
        
        # [JC] Perform high priority and low priority triangular split using numpy functions
        low_t = numpy.fliplr(numpy.tril(numpy.fliplr(img), cutoff))
        high_t = img - low_t
        
        # [JC] Flatten high_t and low_t data and send to outputs 0 and 1 respectfully        
        output_items[0][:] = high_t.reshape(1, high_t.size)
        output_items[1][:] = low_t.reshape(1, low_t.size)
        
        #output_items[0][:] = input_items[0]
        self.consume(0, len(input_items[0]))        #self.consume_each(len(input_items[0]))
		
        return len(output_items[0])
