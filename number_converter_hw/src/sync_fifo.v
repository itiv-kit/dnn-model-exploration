/*
   updated for this design
*/
/*
 MIT License

 Copyright (c) 2019 Yuya Kudo

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

//-----------------------------------------------------------------------------
// module      : sync_fifo
// description : Synchronous FIFO consist of Dual Port RAM for FPGA implementation
module sync_fifo
  #(parameter
    /*
     You can specify the following parameters.
     1. DATA_WIDTH : input and output data width
     2. FIFO_DEPTH : data capacity
     */
    DATA_WIDTH    = 8,
    FIFO_DEPTH    = 256,
    BORDER_ALMOST_FULL = FIFO_DEPTH-8,
    BORDER_MINIMUM_FILL = 1,

    localparam
    PREFETCH_FIFO_DEPTH    = 4,
    LB_FIFO_DEPTH          = $clog2(FIFO_DEPTH),
    LB_PREFETCH_FIFO_DEPTH = $clog2(PREFETCH_FIFO_DEPTH),
    REAL_FIFO_DEPTH = 2**LB_FIFO_DEPTH)
   (input wire [DATA_WIDTH-1:0]   in_data,
    input wire                    in_valid,
    output reg                   in_ready,

    output reg [DATA_WIDTH-1:0]  out_data,
    output reg                   out_valid,
    input wire                    out_ready,

    input wire                     succesing_fifo_almost_full,
    output reg                    almost_full,

    input wire                    clk,
    input wire                    rstn);

   reg [LB_FIFO_DEPTH-1:0]       waddr_r, raddr_r;
   reg [LB_FIFO_DEPTH:0]         mem_count_r;
   reg [DATA_WIDTH-1:0]          mem_din0;
   wire [DATA_WIDTH-1:0]         mem_dout1;
   reg                           mem_wr_en0;
   // reg [31:0] counter_in, counter_out;

   reg                           in_exec, out_exec;
   reg tmp_out_valid;
   
   Memory_TrueDualPortSingleClk #(.DATA_WIDTH(DATA_WIDTH), .BUFFER_SIZE(REAL_FIFO_DEPTH)) dp_RAM (
                                                      waddr_r,
                                                      mem_wr_en0,
                                                      mem_din0,
                                                      clk,
                                                      , //0
                                                      raddr_r,
                                                      1'b0, //0
                                                      0, //0
                                                      mem_dout1);

   always @(*)  begin
      tmp_out_valid <=  ((mem_count_r > 0)  && !succesing_fifo_almost_full)     ? 1 : 0;
      in_ready   <= (mem_count_r < REAL_FIFO_DEPTH)  ? 1 : 0;
      almost_full <= (mem_count_r+ BORDER_ALMOST_FULL  > REAL_FIFO_DEPTH-1) ? 1 : 0;
      out_valid <= !rstn   ? 1'b0 : tmp_out_valid;
      out_exec  <= (tmp_out_valid | succesing_fifo_almost_full   ) & out_ready;
      in_exec    <= !rstn ? 1'b0 : in_valid  & in_ready ;

      mem_din0   <= in_data;
      mem_wr_en0 <= in_exec;
      out_data <= mem_dout1 ;

   end

   always @(posedge clk) begin
      if(!rstn) begin
         waddr_r                  <= 0;
         raddr_r                  <= 0;
         mem_count_r              <= 0;
         // counter_out <= 0;
         // counter_in <= 0;
      end
      else begin
         case({in_exec, out_exec})
         2'b01: begin
               raddr_r                <= raddr_r + 1;
               mem_count_r            <= mem_count_r - 1;
               // counter_out <= counter_out +1;
         end
         2'b10: begin
               waddr_r                  <= waddr_r + 1;
               mem_count_r              <= mem_count_r +1;
               // counter_in <= counter_in +1;
         end
         2'b11: begin
               waddr_r                <= waddr_r + 1;
               raddr_r                <= raddr_r + 1;
               // counter_in <= counter_in + 1;
               // counter_out <= counter_out + 1;
         end
         endcase         
      end
   end

endmodule
