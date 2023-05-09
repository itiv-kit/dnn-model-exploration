/*

This module contains top_dequant_quant and a fifo.

*/
`include "definitions.v"

module top
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 16,
        BUFFER_SIZE_FIFO_AKA_ACCEL = 2,
        INPUT_BITWIDTH = 32,
        MAX_IMPLEMENTATIONS = 16
) (
    input wire clk, rstn,

    // control data
    input wire AXI_LITE_activate,
    input wire [31:0] AXI_LITE_num_of_input_values, 
    input wire [31:0] AXI_LITE_bitwidth, 
    // input wire [INPUT_BITWIDTH-2:0] validation_bits,
    input wire [31:0] AXI_LITE_scale_fp,
    input wire [31:0] AXI_LITE_inv_scale_fp,
    input wire [31:0] AXI_LITE_num_of_values, 
    // output wire AXI_LITE_done,

    // Performance counter
    output wire [31:0] AXI_LITE_counter_axilite_setup,  
    output wire [31:0] AXI_LITE_counter_dequantization_flow,  
    output wire [31:0] AXI_LITE_counter_quantization_flow,   
    output wire [31:0] AXI_LITE_counter_start_of_quantization_flow, 
    output wire [31:0] AXI_LITE_counter_extractor,  
    output wire [31:0] AXI_LITE_counter_dequantization,  
    output wire [31:0] AXI_LITE_counter_start_of_dequantization,   
    output wire [31:0] AXI_LITE_counter_quantization,  
    output wire [31:0] AXI_LITE_counter_compression,  
    output wire [31:0] AXI_LITE_counter_start_of_compression,   

    //input stream data
    input wire AXIS_RCV_valid, AXIS_RCV_last, 
    input wire [(MAX_IMPLEMENTATIONS*16)-1:0] AXIS_RCV_data,
    output wire AXIS_RCV_ready,  

    //output stream data
    output wire AXIS_TRM_valid, AXIS_TRM_last, //TODO: sending the last bit always along the same value it coming into this module
    output wire [(MAX_IMPLEMENTATIONS*16)-1:0] AXIS_TRM_data,
    input wire AXIS_TRM_ready
);

    wire [(MAX_IMPLEMENTATIONS*16)-1:0]  DEQUANT_DATA, QUANT_DATA ;
    wire [0:MAX_IMPLEMENTATIONS-1] TEMP_AXIS_RCV_ready, TEMP_AXIS_TRM_valid, TEMP_AXIS_TRM_last;
    wire [0:MAX_IMPLEMENTATIONS-1] DEQUANT_VALID,     QUANT_READY ;
    wire DEQUANT_READY, QUANT_VALID;




    genvar implems;

    generate
        for(implems = 0; implems < MAX_IMPLEMENTATIONS; implems = implems +1) begin

            top_dequant_quant #(.MAX_BITWIDTH_QUANTIZED_DATA(16), .BITWIDTH_DMA(16)) top_intern (
                clk, rstn,

                //control data - UPSTREAM
                AXI_LITE_activate, 
                AXI_LITE_num_of_input_values,
                AXI_LITE_bitwidth, 
                AXI_LITE_scale_fp[15:0], 
                AXI_LITE_inv_scale_fp[15:0], 
                AXI_LITE_num_of_values,

                //control data - DOWNSTREAM (same as UPSTREAM)
                AXI_LITE_activate, 
                AXI_LITE_num_of_input_values,
                AXI_LITE_bitwidth, 
                AXI_LITE_scale_fp[15:0], 
                AXI_LITE_inv_scale_fp[15:0], 
                AXI_LITE_num_of_values,

`ifdef USE_PERFORMANCE_COUNTER
                // Performance counter
                ,  
                ,  
                ,  
                , 
                ,  
                ,  
                ,    
                ,  
                ,  
                ,   
`endif // USE_PERFORMANCE_COUNTER 

                //input stream data from dma
                AXIS_RCV_valid, AXIS_RCV_last, AXIS_RCV_data[((implems+1)*16)-1:(implems*16)], TEMP_AXIS_RCV_ready[implems],       
                //output stream data to accel
                DEQUANT_VALID[implems], DEQUANT_DATA[((implems+1)*16)-1:(implems*16)], DEQUANT_READY,
                //input stream data from accel
                QUANT_VALID, QUANT_DATA[((implems+1)*16)-1:(implems*16)], QUANT_READY[implems],
                //output stream data to dma
                TEMP_AXIS_TRM_valid[implems], TEMP_AXIS_TRM_last[implems], AXIS_TRM_data[((implems+1)*16)-1:(implems*16)], AXIS_TRM_ready
                );
            
        end
    endgenerate

    sync_fifo #(.DATA_WIDTH(256), .FIFO_DEPTH(BUFFER_SIZE_FIFO_AKA_ACCEL)) fifo_accel (
                 DEQUANT_DATA, DEQUANT_VALID[0], DEQUANT_READY , 
                 QUANT_DATA,  QUANT_VALID,QUANT_READY[0],
                 1'b0 ,  , 
                 clk, (rstn && AXI_LITE_activate) );

    assign AXIS_RCV_ready = TEMP_AXIS_RCV_ready[0];
    assign AXIS_TRM_last = TEMP_AXIS_TRM_last[0];
    assign AXIS_TRM_valid = TEMP_AXIS_TRM_valid[0];


    assign AXI_LITE_counter_axilite_setup = 0;
    assign AXI_LITE_counter_dequantization_flow = 0;
    assign AXI_LITE_counter_quantization_flow = 0;
    assign AXI_LITE_counter_start_of_quantization_flow = 0;
    assign AXI_LITE_counter_extractor = 0;
    assign AXI_LITE_counter_dequantization = 0;
    assign AXI_LITE_counter_start_of_dequantization = 0;
    assign AXI_LITE_counter_quantization = 0;
    assign AXI_LITE_counter_compression = 0;
    assign AXI_LITE_counter_start_of_compression = 0;


`ifndef USE_LAYER_CNT
    assign AXI_LITE_cnt_layer = 0;
`endif


endmodule
