#
# Copyright 2021 Ettus Research, a National Instruments Brand
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

include $(TOOLS_DIR)/make/viv_ip_builder.mak

LIB_IP_AXI_CONV_SRCS = $(IP_BUILD_DIR)/axi_conv/axi_conv.xci

LIB_IP_AXI_CONV_OUTS = $(addprefix $(IP_BUILD_DIR)/axi_conv/, \
axi_conv.xci.out \
synth/axi_conv.vhd \
)

$(LIB_IP_AXI_CONV_SRCS) $(LIB_IP_AXI_CONV_OUTS) : $(OOT_FPGA_DIR)/ip/axi_conv/axi_conv.xci
	$(call BUILD_VIVADO_IP,axi_conv,$(ARCH),$(PART_ID),$(OOT_FPGA_DIR)/ip,$(IP_BUILD_DIR),0)

