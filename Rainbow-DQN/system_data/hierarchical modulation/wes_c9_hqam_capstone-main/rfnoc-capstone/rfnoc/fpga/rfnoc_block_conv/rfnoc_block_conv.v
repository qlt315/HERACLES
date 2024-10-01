//
// Copyright 2022 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Module: rfnoc_block_conv
//
// Description:
//
//   <Add block description here>
//
// Parameters:
//
//   THIS_PORTID : Control crossbar port to which this block is connected
//   CHDR_W      : AXIS-CHDR data bus width
//   MTU         : Maximum transmission unit (i.e., maximum packet size in
//                 CHDR words is 2**MTU).
//

`default_nettype none


module rfnoc_block_conv #(
  parameter [9:0] THIS_PORTID     = 10'd0,
  parameter       CHDR_W          = 64,
  parameter [5:0] MTU             = 10
)(
  // RFNoC Framework Clocks and Resets
  input  wire                   rfnoc_chdr_clk,
  input  wire                   rfnoc_ctrl_clk,
  input  wire                   ce_clk,
  // RFNoC Backend Interface
  input  wire [511:0]           rfnoc_core_config,
  output wire [511:0]           rfnoc_core_status,
  // AXIS-CHDR Input Ports (from framework)
  input  wire [(1)*CHDR_W-1:0] s_rfnoc_chdr_tdata,
  input  wire [(1)-1:0]        s_rfnoc_chdr_tlast,
  input  wire [(1)-1:0]        s_rfnoc_chdr_tvalid,
  output wire [(1)-1:0]        s_rfnoc_chdr_tready,
  // AXIS-CHDR Output Ports (to framework)
  output wire [(1)*CHDR_W-1:0] m_rfnoc_chdr_tdata,
  output wire [(1)-1:0]        m_rfnoc_chdr_tlast,
  output wire [(1)-1:0]        m_rfnoc_chdr_tvalid,
  input  wire [(1)-1:0]        m_rfnoc_chdr_tready,
  // AXIS-Ctrl Input Port (from framework)
  input  wire [31:0]            s_rfnoc_ctrl_tdata,
  input  wire                   s_rfnoc_ctrl_tlast,
  input  wire                   s_rfnoc_ctrl_tvalid,
  output wire                   s_rfnoc_ctrl_tready,
  // AXIS-Ctrl Output Port (to framework)
  output wire [31:0]            m_rfnoc_ctrl_tdata,
  output wire                   m_rfnoc_ctrl_tlast,
  output wire                   m_rfnoc_ctrl_tvalid,
  input  wire                   m_rfnoc_ctrl_tready
);

  // [JC] - Last updated 19May2022
  `include "/home/wes/projects/dependencies/uhd/fpga/usrp3/lib/rfnoc/core/rfnoc_chdr_utils.vh"

  //---------------------------------------------------------------------------
  // Signal Declarations
  //---------------------------------------------------------------------------

  // Clocks and Resets
  wire               ctrlport_clk;
  wire               ctrlport_rst;
  wire               axis_data_clk;
  wire               axis_data_rst;
  // CtrlPort Master
  wire               m_ctrlport_req_wr;
  wire               m_ctrlport_req_rd;
  wire [19:0]        m_ctrlport_req_addr;
  wire [31:0]        m_ctrlport_req_data;
  wire               m_ctrlport_resp_ack;
  wire [31:0]        m_ctrlport_resp_data;
  // Payload Stream to User Logic: in
  wire [8*1-1:0]    m_in_payload_tdata;  // [JC] - 19May2022
  wire [1-1:0]       m_in_payload_tkeep;
  wire [1-1:0]       m_in_payload_tlast;
  wire [1-1:0]       m_in_payload_tvalid;
  wire [1-1:0]       m_in_payload_tready;
  // Context Stream to User Logic: in
  wire [CHDR_W-1:0]  m_in_context_tdata;
  wire [3:0]         m_in_context_tuser;
  wire [1-1:0]       m_in_context_tlast;
  wire [1-1:0]       m_in_context_tvalid;
  reg  [1-1:0]       m_in_context_tready;
  // Payload Stream from User Logic: out
  wire [8*1-1:0]    s_out_payload_tdata;  // [JC] - 19May2022
  wire [0:0]         s_out_payload_tkeep;
  wire [1-1:0]       s_out_payload_tlast;
  wire [1-1:0]       s_out_payload_tvalid;
  wire [1-1:0]       s_out_payload_tready;
  // Context Stream from User Logic: out
  reg  [CHDR_W-1:0]  s_out_context_tdata;
  reg  [3:0]         s_out_context_tuser;
  reg  [1-1:0]       s_out_context_tlast;
  reg  [1-1:0]       s_out_context_tvalid;
  wire [1-1:0]       s_out_context_tready;
	
  wire               ce_rst;     // [JC] - Last updated 19May2022	

  //---------------------------------------------------------------------------
  // NoC Shell
  //---------------------------------------------------------------------------

  noc_shell_conv #(
    .CHDR_W              (CHDR_W),
    .THIS_PORTID         (THIS_PORTID),
    .MTU                 (MTU)
  ) noc_shell_conv_i (
    //---------------------
    // Framework Interface
    //---------------------

    // Clock Inputs
    .rfnoc_chdr_clk      (rfnoc_chdr_clk),
    .rfnoc_ctrl_clk      (rfnoc_ctrl_clk),
    .ce_clk              (ce_clk),
    // Reset Outputs
    .rfnoc_chdr_rst      (),
    .rfnoc_ctrl_rst      (),
    .ce_rst              (ce_rst),  // [JC] - Last updated 19May2022
    // RFNoC Backend Interface
    .rfnoc_core_config   (rfnoc_core_config),
    .rfnoc_core_status   (rfnoc_core_status),
    // CHDR Input Ports  (from framework)
    .s_rfnoc_chdr_tdata  (s_rfnoc_chdr_tdata),
    .s_rfnoc_chdr_tlast  (s_rfnoc_chdr_tlast),
    .s_rfnoc_chdr_tvalid (s_rfnoc_chdr_tvalid),
    .s_rfnoc_chdr_tready (s_rfnoc_chdr_tready),
    // CHDR Output Ports (to framework)
    .m_rfnoc_chdr_tdata  (m_rfnoc_chdr_tdata),
    .m_rfnoc_chdr_tlast  (m_rfnoc_chdr_tlast),
    .m_rfnoc_chdr_tvalid (m_rfnoc_chdr_tvalid),
    .m_rfnoc_chdr_tready (m_rfnoc_chdr_tready),
    // AXIS-Ctrl Input Port (from framework)
    .s_rfnoc_ctrl_tdata  (s_rfnoc_ctrl_tdata),
    .s_rfnoc_ctrl_tlast  (s_rfnoc_ctrl_tlast),
    .s_rfnoc_ctrl_tvalid (s_rfnoc_ctrl_tvalid),
    .s_rfnoc_ctrl_tready (s_rfnoc_ctrl_tready),
    // AXIS-Ctrl Output Port (to framework)
    .m_rfnoc_ctrl_tdata  (m_rfnoc_ctrl_tdata),
    .m_rfnoc_ctrl_tlast  (m_rfnoc_ctrl_tlast),
    .m_rfnoc_ctrl_tvalid (m_rfnoc_ctrl_tvalid),
    .m_rfnoc_ctrl_tready (m_rfnoc_ctrl_tready),

    //---------------------
    // Client Interface
    //---------------------

    // CtrlPort Clock and Reset
    .ctrlport_clk              (ctrlport_clk),
    .ctrlport_rst              (ctrlport_rst),
    // CtrlPort Master
    .m_ctrlport_req_wr         (m_ctrlport_req_wr),
    .m_ctrlport_req_rd         (m_ctrlport_req_rd),
    .m_ctrlport_req_addr       (m_ctrlport_req_addr),
    .m_ctrlport_req_data       (m_ctrlport_req_data),
    .m_ctrlport_resp_ack       (m_ctrlport_resp_ack),
    .m_ctrlport_resp_data      (m_ctrlport_resp_data),

    // AXI-Stream Payload Context Clock and Reset
    .axis_data_clk (axis_data_clk),
    .axis_data_rst (axis_data_rst),
    // Payload Stream to User Logic: in
    .m_in_payload_tdata  (m_in_payload_tdata),
    .m_in_payload_tkeep  (m_in_payload_tkeep),
    .m_in_payload_tlast  (m_in_payload_tlast),
    .m_in_payload_tvalid (m_in_payload_tvalid),
    .m_in_payload_tready (m_in_payload_tready),
    // Context Stream to User Logic: in
    .m_in_context_tdata  (m_in_context_tdata),
    .m_in_context_tuser  (m_in_context_tuser),
    .m_in_context_tlast  (m_in_context_tlast),
    .m_in_context_tvalid (m_in_context_tvalid),
    .m_in_context_tready (m_in_context_tready),
    // Payload Stream from User Logic: out
    .s_out_payload_tdata  (s_out_payload_tdata),
    .s_out_payload_tkeep  (s_out_payload_tkeep),
    .s_out_payload_tlast  (s_out_payload_tlast),
    .s_out_payload_tvalid (s_out_payload_tvalid),
    .s_out_payload_tready (s_out_payload_tready),
    // Context Stream from User Logic: out
    .s_out_context_tdata  (s_out_context_tdata),
    .s_out_context_tuser  (s_out_context_tuser),
    .s_out_context_tlast  (s_out_context_tlast),
    .s_out_context_tvalid (s_out_context_tvalid),
    .s_out_context_tready (s_out_context_tready)
  );

  //---------------------------------------------------------------------------
  // Context Handling - [JC] Last update 19-May-2022
  //---------------------------------------------------------------------------
  //
  // Output packets have 2 times the payload size of input packets, so we need to
  // update the header length field as it passes through.
  //
  //---------------------------------------------------------------------------

  genvar port;

  for (port = 0; port < 1; port = port+1) begin : gen_context_ports

    always @(*) begin : update_packet_length
      reg [CHDR_W-1:0] old_tdata;
      reg [CHDR_W-1:0] new_tdata;

      old_tdata = m_in_context_tdata[CHDR_W*port +: CHDR_W];

      // Check if this context word contains the header
      if (m_in_context_tuser[4*port +: 4] == CONTEXT_FIELD_HDR || 
          m_in_context_tuser[4*port +: 4] == CONTEXT_FIELD_HDR_TS
      ) begin : change_header
        // Update the lower 64-bits (the header word) with the new length
        reg [15:0] pyld_length;
		pyld_length     = chdr_calc_payload_length(CHDR_W, old_tdata) * (2); // output len is 2 * input len
        new_tdata       = old_tdata;
        new_tdata[63:0] = chdr_update_length(CHDR_W, old_tdata, pyld_length);
      end else begin : pass_through_header
        // Not a header word, so pass through unchanged
        new_tdata = old_tdata;
      end

      s_out_context_tdata  [CHDR_W*port +: CHDR_W] = new_tdata;
      s_out_context_tuser  [     4*port +:      4] = m_in_context_tuser   [4*port +: 4];
      s_out_context_tlast  [     1*port +:      1] = m_in_context_tlast   [1*port +: 1];
      s_out_context_tvalid [     1*port +:      1] = m_in_context_tvalid  [1*port +: 1];
      m_in_context_tready  [     1*port +:      1] = s_out_context_tready [1*port +: 1];
    end // update_packet_length

  end // gen_context_ports

  //---------------------------------------------------------------------------
  // User Logic - [JC] Last update 19-May-2022
  //---------------------------------------------------------------------------

  axi_conv inst_axi_conv (
	.aclk(ce_clk), .aresetn(~(ce_rst)),
	  .s_axis_data_tvalid(m_in_payload_tvalid),
	  .s_axis_data_tready(m_in_payload_tready),
//	  .s_axis_data_tlast(m_in_payload_tlast),       // tlast is only valid in axi_conv core for punctured codes
	  .s_axis_data_tdata(m_in_payload_tdata[7:0]),
	  .m_axis_data_tvalid(s_out_payload_tvalid),
	  .m_axis_data_tready(s_out_payload_tready),
//	  .m_axis_data_tlast(s_out_payload_tlast),      // tlast is only valid in axi_conv core for punctured codes
	  .m_axis_data_tdata(s_out_payload_tdata[7:0]));

  // Nothing to do yet, so just drive control signals to default values
  assign m_ctrlport_resp_ack = 1'b0;
  assign m_in_payload_tlast = 1'b0;
  assign s_out_payload_tlast = 1'b0;
	
//  assign m_in_payload_tready = 1'b0;
//  assign m_in_context_tready = 1'b0;
//  assign s_out_payload_tvalid = 1'b0;
//  assign s_out_context_tvalid = 1'b0;

endmodule // rfnoc_block_conv


`default_nettype wire
