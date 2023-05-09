/*
    containing the rtl design 
*/
`include "definitions.v"

module top_dequant_quant
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 8,
        BITWIDTH_DMA = 32,
        SINGLE_BITWIDTH = MAX_BITWIDTH_QUANTIZED_DATA,
        SINGLE_NUM_OF_INPUT_VALUES = 16,
        SINGLE_SCALE_FP = 32'b00111100000111111010101011101001,
        SINGLE_INV_SCALE_FP = 32'b01000010110011010011100111110001,
        SINGLE_NUM_OF_OUTPUT_VALUES = SINGLE_NUM_OF_INPUT_VALUES,
        PIPELINE_LENGTH_DEQUANTIZATION = 17,
        PIPELINE_LENGTH_QUANTIZATION = 17,

        BUFFER_SIZE_AXIS2MODEL = 3, // TODO: deletable
        BUFFER_SIZE_QUANT2COMPRESSOR = PIPELINE_LENGTH_QUANTIZATION+1000, // > 29
        BUFFER_SIZE_EXTRACT2DEQUANT = 2, // TODO: deletable
        BUFFER_SIZE_DEQUANT2ACCEL = PIPELINE_LENGTH_DEQUANTIZATION+1000, // > 14
        BUFFER_SIZE_ACCEL2QUANT = 2 // TODO: deletable
       
) (
    input wire clk, rstn,

    // control data - UPSTREAM
    input wire UPSTREAM__AXI_LITE_activate,
    input wire [31:0] UPSTREAM__AXI_LITE_num_of_input_values, 
    input wire [31:0] UPSTREAM__AXI_LITE_bitwidth, 
    input wire [15:0] UPSTREAM__AXI_LITE_scale_fp,
    input wire [15:0] UPSTREAM__AXI_LITE_inv_scale_fp,
    input wire [31:0] UPSTREAM__AXI_LITE_num_of_output_values,  

    // control data - DOWNSTREAM
    input wire DOWNSTREAM__AXI_LITE_activate,
    input wire [31:0] DOWNSTREAM__AXI_LITE_num_of_input_values, 
    input wire [31:0] DOWNSTREAM__AXI_LITE_bitwidth, 
    input wire [15:0] DOWNSTREAM__AXI_LITE_scale_fp,
    input wire [15:0] DOWNSTREAM__AXI_LITE_inv_scale_fp,
    input wire [31:0] DOWNSTREAM__AXI_LITE_num_of_output_values,  


`ifdef USE_PERFORMANCE_COUNTER
    // Performance counter
    //TODO: inser counter for every module
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
`endif


`ifdef USE_DBG_VIVADO
    output wire bit_0,bit_1,bit_2,bit_3,bit_4,bit_5,bit_6,bit_7,
    output wire bit_8,bit_9,bit_10,bit_11,bit_12,bit_13,bit_14,bit_15,
    output wire [31:0] mem_0,mem_1,mem_2,mem_3,mem_4,mem_5,mem_6,mem_7,
    output wire [31:0] mem_8,mem_9,mem_10,mem_11,mem_12,mem_13,mem_14,mem_15,
    output wire [63:0] double_0,double_1,//,double_2,double_3,
`endif 

    //input stream data
    input wire AXIS_RCV_valid, AXIS_RCV_last, 
    input wire [BITWIDTH_DMA-1:0] AXIS_RCV_data,
    output wire AXIS_RCV_ready,  

    //output stream data to accel
    output wire DEQUANT_VALID,
    output wire [15:0] DEQUANT_DATA,
    input wire DEQUANT_READY,

    //input stream data from accel
    input wire QUANT_VALID, 
    input wire [15:0] QUANT_DATA, 
    output wire QUANT_READY,

    //output stream data
    output wire AXIS_TRM_valid, AXIS_TRM_last, //TODO: sending the last bit always along the same value it coming into this module
    output wire [BITWIDTH_DMA-1:0] AXIS_TRM_data,
    input wire AXIS_TRM_ready
);

/*
    |---------|                    |-------------| 
    | memory  |  --- UPSTREAM ---> | Accelerator |
    |---------|                    |-------------|

    |---------|                    |-------------| 
    | memory  |  <-- DOWNSTREAM -- | Accelerator |
    |---------|                    |-------------|

*/




    // REGISTERS - START 
    //TODO: all registers necessary?
    //TODO: adapt 2-D to 1-D registers

    //control - UPSTREAM
    reg [MAX_BITWIDTH_QUANTIZED_DATA-1:0] UPSTREAM__ctrl_mask_valid_bits; 
    reg [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] UPSTREAM__ctrl_bitwidth_d; 
    reg [15:0] UPSTREAM__ctrl_scale_fp;
    reg [15:0] UPSTREAM__ctrl_inv_scale_fp;
    reg [31:0] UPSTREAM__ctrl_num_of_output_values, UPSTREAM__ctrl_num_of_input_values; //TODO: count this one to zero
    reg UPSTREAM__AXI_LITE_activate_delay;

    //control - DOWNSTREAM
    reg [MAX_BITWIDTH_QUANTIZED_DATA-1:0] DOWNSTREAM__ctrl_mask_valid_bits; 
    reg [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] DOWNSTREAM__ctrl_bitwidth_d; 
    reg [15:0] DOWNSTREAM__ctrl_scale_fp;
    reg [15:0] DOWNSTREAM__ctrl_inv_scale_fp;
    reg [31:0] DOWNSTREAM__ctrl_num_of_output_values, DOWNSTREAM__ctrl_num_of_input_values; //TODO: count this one to zero
    reg DOWNSTREAM__AXI_LITE_activate_delay;

    // other
    reg rstn_intern;

    //extractor
    wire extractor_in_ready, extractor_in_valid;
    wire [BITWIDTH_DMA-1:0] extractor_in_data;
    wire extractor_out_ready, extractor_out_valid;
    wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] extractor_out_data ;

    wire dequantization_in_ready;

    // // fifo_extract2dequant
    // wire fifo_extract2dequant_in_valid, fifo_extract2dequant_in_ready, fifo_extract2dequant_out_ready, fifo_extract2dequant_out_valid;
    // wire [MAX_BITWIDTH_QUANTIZED_DATA:0] fifo_extract2dequant_in_data, fifo_extract2dequant_out_data;

    // fifo_dequant2accel
    wire fifo_dequant2accel_in_valid, fifo_dequant2accel_in_ready, fifo_dequant2accel_out_ready, fifo_dequant2accel_out_valid, fifo_dequant2accel_almost_full;
    reg fifo_dequant2accel_almost_full_delay;
    wire [15:0] fifo_dequant2accel_in_data, fifo_dequant2accel_out_data;

    // fifo_accel2quant
    wire fifo_accel2quant_in_valid, fifo_accel2quant_in_ready, fifo_accel2quant_out_ready, fifo_accel2quant_out_valid;
    wire [15:0] fifo_accel2quant_in_data, fifo_accel2quant_out_data;

    // quant2compressor
    wire fifo_quant2compressor_in_valid, fifo_quant2compressor_in_ready, fifo_quant2compressor_out_ready, fifo_quant2compressor_out_valid, fifo_quant2compressor_almost_full;
    wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] fifo_quant2compressor_in_data, fifo_quant2compressor_out_data;


    wire zero_b = 1'b0;
    wire [31:0] zero_w = 32'd0;
    wire [63:0] zero_d = 64'd0;

`ifdef USE_PERFORMANCE_COUNTER
    wire extractor_done;
    wire [31:0] counter_axiLite_setup;
    wire [31:0] counter_dequantization_flow;
    wire [31:0] counter_quantization_flow, counter_start_of_quantization_flow; 
    wire [31:0] counter_extractor; 
    wire [31:0] counter_dequantization, counter_start_of_dequantization; 
    wire [31:0] counter_quantization; 
    wire [31:0] counter_compression, counter_start_of_compression; 
    reg [31:0]  cnt_number_of_input_values_to_accel, cnt_number_of_output_values_from_accel;
    reg dff_3, dff_4;
    wire [31:0] counter_dequant_first, counter_quant_first;
`endif

    //REGISTERS - END
    


    /*****************************************************************************
        SECTION: CONTROL LOGIC
            - START
    *****************************************************************************/

    // control data - UPSTREAM
    always @(posedge clk) begin
        UPSTREAM__AXI_LITE_activate_delay <= UPSTREAM__AXI_LITE_activate;
        if(!(rstn && UPSTREAM__AXI_LITE_activate)) begin
            UPSTREAM__ctrl_mask_valid_bits <= 0;
            UPSTREAM__ctrl_bitwidth_d <= 0;
            UPSTREAM__ctrl_scale_fp <= 0;
            UPSTREAM__ctrl_inv_scale_fp <= 0;
            UPSTREAM__AXI_LITE_activate_delay <= 0;
            rstn_intern <= 1'b0;
        end 
        else begin
            if (UPSTREAM__AXI_LITE_activate && !UPSTREAM__AXI_LITE_activate_delay) begin 
                rstn_intern <= 1'b1;
                UPSTREAM__ctrl_mask_valid_bits <= (2**UPSTREAM__AXI_LITE_bitwidth)-1;
                UPSTREAM__ctrl_bitwidth_d <= UPSTREAM__AXI_LITE_bitwidth;
                UPSTREAM__ctrl_scale_fp <= UPSTREAM__AXI_LITE_scale_fp;
                UPSTREAM__ctrl_inv_scale_fp <= UPSTREAM__AXI_LITE_inv_scale_fp;
                UPSTREAM__ctrl_num_of_input_values <= UPSTREAM__AXI_LITE_num_of_input_values;
                UPSTREAM__ctrl_num_of_output_values <= UPSTREAM__AXI_LITE_num_of_output_values;
            end     
        end
    end

    // control data - DOWNSTREAM
    always @(posedge clk) begin
        DOWNSTREAM__AXI_LITE_activate_delay <= DOWNSTREAM__AXI_LITE_activate;
        if(!(rstn && UPSTREAM__AXI_LITE_activate)) begin
            DOWNSTREAM__ctrl_mask_valid_bits <= 0;
            DOWNSTREAM__ctrl_bitwidth_d <= 0;
            DOWNSTREAM__ctrl_scale_fp <= 0;
            DOWNSTREAM__ctrl_inv_scale_fp <= 0;
            DOWNSTREAM__AXI_LITE_activate_delay <= 0;
            rstn_intern <= 1'b0;
        end 
        else begin
            if (UPSTREAM__AXI_LITE_activate && !DOWNSTREAM__AXI_LITE_activate_delay) begin 
                rstn_intern <= 1'b1;
                DOWNSTREAM__ctrl_mask_valid_bits <= (2**DOWNSTREAM__AXI_LITE_bitwidth)-1;
                DOWNSTREAM__ctrl_bitwidth_d <= DOWNSTREAM__AXI_LITE_bitwidth;
                DOWNSTREAM__ctrl_scale_fp <= DOWNSTREAM__AXI_LITE_scale_fp;
                DOWNSTREAM__ctrl_inv_scale_fp <= DOWNSTREAM__AXI_LITE_inv_scale_fp;
                DOWNSTREAM__ctrl_num_of_input_values <= DOWNSTREAM__AXI_LITE_num_of_input_values;
                DOWNSTREAM__ctrl_num_of_output_values <= DOWNSTREAM__AXI_LITE_num_of_output_values;
            end     
        end
    end





`ifdef USE_PERFORMANCE_COUNTER
    // counter for passed values - UPSTREAM
always @(posedge clk ) begin

    if  (UPSTREAM__AXI_LITE_activate && !UPSTREAM__AXI_LITE_activate_delay) begin
        cnt_number_of_input_values_to_accel <= 0;
    end
    else begin
        if ( !dff_3 && DEQUANT_VALID && DEQUANT_READY) begin
            cnt_number_of_input_values_to_accel <= cnt_number_of_input_values_to_accel + 1;
            // dff_3 <= 1'b1;
        end
    end
    if (!(DEQUANT_VALID && DEQUANT_READY)) begin
        dff_3 <= 1'b0;
    end
end 

    // counter for passed values - DOWNSTREAM
always @(posedge clk ) begin

    if  (UPSTREAM__AXI_LITE_activate && !DOWNSTREAM__AXI_LITE_activate_delay) begin
        cnt_number_of_output_values_from_accel <= 0;
    end
    else begin
        if( !dff_4 && fifo_quant2compressor_in_valid && fifo_quant2compressor_in_ready) begin
            cnt_number_of_output_values_from_accel <= cnt_number_of_output_values_from_accel + 1;
            // dff_4 <= 1'b1;
        end
    end
    if (!(QUANT_VALID && QUANT_READY)) begin
        dff_4 <= 1'b0;
    end
end 

`endif // USE_PERFORMANCE_COUNTER
    

    /*****************************************************************************
        SECTION: CONTROL LOGIC
            - DONE
    *****************************************************************************/

    /*****************************************************************************
        SECTION: DMA TO ACCEL --- UPSTREAM
            - START
    *****************************************************************************/



    // Extractor
    extractor_ring extractor (
                    clk, rstn_intern,
                    UPSTREAM__ctrl_mask_valid_bits, UPSTREAM__ctrl_bitwidth_d, UPSTREAM__ctrl_num_of_input_values,   AXIS_RCV_valid,AXIS_RCV_data, AXIS_RCV_ready,
                    extractor_out_valid, extractor_out_data, extractor_out_ready
`ifdef USE_PERFORMANCE_COUNTER
                    ,extractor_done
`endif 
                    );

    // // Dequantizer
    dequantization #(.MAX_BITWIDTH_QUANTIZED_DATA(MAX_BITWIDTH_QUANTIZED_DATA)) dequants (
        clk, rstn_intern, 
        extractor_out_valid,  extractor_out_data,
        UPSTREAM__ctrl_scale_fp, UPSTREAM__ctrl_bitwidth_d, UPSTREAM__ctrl_mask_valid_bits, 
        fifo_dequant2accel_in_valid, fifo_dequant2accel_in_data); 

    assign extractor_out_ready = ~fifo_dequant2accel_almost_full;
    
    sync_fifo #(.DATA_WIDTH(16), .FIFO_DEPTH(BUFFER_SIZE_DEQUANT2ACCEL), .BORDER_ALMOST_FULL(14)) dequant2accel (
                fifo_dequant2accel_in_data, fifo_dequant2accel_in_valid, ,
                fifo_dequant2accel_out_data, fifo_dequant2accel_out_valid, fifo_dequant2accel_out_ready,
                1'b0,fifo_dequant2accel_almost_full,
                clk, rstn_intern);

    assign DEQUANT_VALID = fifo_dequant2accel_out_valid;
    assign DEQUANT_DATA = fifo_dequant2accel_out_data;
    assign fifo_dequant2accel_out_ready = DEQUANT_READY;




    /*****************************************************************************
        SECTION: DMA TO ACCEL --- UPSTREAM
            - DONE
    *****************************************************************************/


    /*****************************************************************************
        SECTION: ACCEL TO DMA --- DOWNSTREAM
            - START
    *****************************************************************************/


    assign fifo_accel2quant_out_valid = QUANT_VALID & (~fifo_quant2compressor_almost_full);
    assign fifo_accel2quant_out_data = QUANT_DATA;
    assign QUANT_READY = (~fifo_quant2compressor_almost_full)  ;

    // // Quantizer
    quantization #(.MAX_BITWIDTH_QUANTIZED_DATA(MAX_BITWIDTH_QUANTIZED_DATA)) quants (
        clk, rstn_intern, 
        fifo_accel2quant_out_valid , 
        DOWNSTREAM__ctrl_bitwidth_d,
        fifo_accel2quant_out_data,
        DOWNSTREAM__ctrl_inv_scale_fp,  
        fifo_quant2compressor_in_valid, fifo_quant2compressor_in_data); 
   
    sync_fifo #(.DATA_WIDTH(MAX_BITWIDTH_QUANTIZED_DATA), .FIFO_DEPTH(BUFFER_SIZE_QUANT2COMPRESSOR), .BORDER_ALMOST_FULL(29)) quant2compressor (
                fifo_quant2compressor_in_data, fifo_quant2compressor_in_valid, ,
                fifo_quant2compressor_out_data, fifo_quant2compressor_out_valid, fifo_quant2compressor_out_ready,
                1'b0,fifo_quant2compressor_almost_full,
                clk, rstn_intern);

    // compression module
    compressor_ring #(.MAXBITWIDTH(MAX_BITWIDTH_QUANTIZED_DATA), .OUTPUT_BITWIDTH(BITWIDTH_DMA)) compression ( 
        clk, rstn_intern,
            DOWNSTREAM__ctrl_bitwidth_d,
            fifo_quant2compressor_out_valid, fifo_quant2compressor_out_data, DOWNSTREAM__ctrl_num_of_output_values, fifo_quant2compressor_out_ready  , 
            AXIS_TRM_valid, AXIS_TRM_data, AXIS_TRM_last, AXIS_TRM_ready
`ifdef USE_DBG_VIVADO  
            ,DBG_compressor_fsm, DBG_compressor_reg_valid
`endif 
            );


    /*****************************************************************************
        SECTION: ACCEL TO DMA --- DOWNSTREAM
            - DONE
    *****************************************************************************/


    /*****************************************************************************
        SECTION: DBG VALUES
            - START
    *****************************************************************************/


`ifdef USE_DBG_VIVADO    
    assign bit_0 = zero_b;
    assign bit_1 = zero_b;
    assign bit_2 = zero_b;
    assign bit_3 = zero_b  ;
    assign bit_4 = zero_b;
    assign bit_5 = zero_b;
    assign bit_6 = fifo_quant2compressor_out_valid & fifo_quant2compressor_out_ready;
    assign bit_7 = (AXIS_TRM_last & AXIS_TRM_valid & AXIS_TRM_ready);
    assign bit_8 = cnt_number_of_output_values_from_accel == (UPSTREAM__ctrl_num_of_output_values-1);
    assign bit_9 = cnt_number_of_input_values_to_accel == (UPSTREAM__ctrl_num_of_input_values-1);
    assign bit_10 = extractor_out_ready & extractor_out_valid;
    assign bit_11 = UPSTREAM__AXI_LITE_activate && !UPSTREAM__AXI_LITE_activate_delay;
    assign bit_12 = !UPSTREAM__AXI_LITE_activate && UPSTREAM__AXI_LITE_activate_delay;
    assign bit_13 = extractor_done;
    assign bit_14 = DEQUANT_READY & DEQUANT_VALID;
    assign bit_15 = QUANT_VALID & QUANT_READY;
    assign mem_0 = zero_w;
    assign mem_1 = fifo_quant2compressor_out_data;
    assign mem_2 = cnt_number_of_output_values_from_accel;
    assign mem_3 = UPSTREAM__ctrl_num_of_output_values;
    assign mem_4 = cnt_number_of_input_values_to_accel;
    assign mem_5 = UPSTREAM__ctrl_num_of_input_values;
    assign mem_6 = zero_w;
    assign mem_7 = zero_w;
    assign mem_8 = zero_w;
    assign mem_9 = zero_w;
    assign mem_10 = zero_w;
    assign mem_11 = zero_w;
    assign mem_12 = zero_w;
    assign mem_13 = QUANT_DATA;
    assign mem_14 = DEQUANT_DATA;
    assign mem_15 = UPSTREAM__ctrl_num_of_output_values;
    assign double_0 = zero_d;
    assign double_1 = zero_d;
`endif 

    /*****************************************************************************
        SECTION: DBG VALUES
            - DONE
    *****************************************************************************/


    /*****************************************************************************
        SECTION: PERFORMANCE COUNTER
            - START
    *****************************************************************************/


`ifdef USE_PERFORMANCE_COUNTER

    performance_counter perf_axilite (
        clk, rstn_intern,
        {!UPSTREAM__AXI_LITE_activate && UPSTREAM__AXI_LITE_activate_delay},
        {UPSTREAM__AXI_LITE_activate && !UPSTREAM__AXI_LITE_activate_delay},
        ,
        counter_axiLite_setup,
        
    );


    performance_counter perf_dequantization_flow_1 (
        clk, rstn_intern,
        {AXIS_RCV_ready & AXIS_RCV_valid},
        {(cnt_number_of_input_values_to_accel == (UPSTREAM__ctrl_num_of_input_values))},
        ,
        counter_dequantization_flow,
        
    );


    performance_counter perf_extraction (
        clk, rstn_intern,
        {AXIS_RCV_ready & AXIS_RCV_valid},
        {(extractor_done)},
        ,
        counter_extractor,
        
    );


    performance_counter perf_dequantization (
        clk, rstn_intern,
        {extractor_out_ready & extractor_out_valid},
        {(cnt_number_of_input_values_to_accel == (UPSTREAM__ctrl_num_of_input_values))},
        counter_dequantization_flow,
        counter_dequantization,
        counter_start_of_dequantization
    );

    performance_counter perf_quantization_flow (
        clk, rstn_intern,
        {QUANT_VALID & QUANT_READY},
        {(AXIS_TRM_last & AXIS_TRM_valid & AXIS_TRM_ready)},
        counter_dequantization_flow,
        counter_quantization_flow,
        counter_start_of_quantization_flow
    );


    performance_counter perf_quantization (
        clk, rstn_intern,
        {QUANT_VALID & QUANT_READY},
        {(cnt_number_of_output_values_from_accel == (UPSTREAM__ctrl_num_of_output_values))},
        ,
        counter_quantization,
            
    );

    performance_counter perf_compression (
        clk, rstn_intern,
        {fifo_quant2compressor_out_valid & fifo_quant2compressor_out_ready},
        {(AXIS_TRM_last & AXIS_TRM_valid & AXIS_TRM_ready)},
        counter_quantization_flow,
        counter_compression,
        counter_start_of_compression
    );
    
       performance_counter perf_dequant_first (
        clk, rstn_intern,
        {AXIS_RCV_ready & AXIS_RCV_valid},
        {DEQUANT_VALID & DEQUANT_READY},
        ,
        counter_dequant_first,
        
    );
    
    performance_counter perf_quant_first (
        clk, rstn_intern,
        {QUANT_VALID & QUANT_READY},
        {AXIS_TRM_ready & AXIS_TRM_valid},
        ,
        counter_quant_first,
        
    );

    assign AXI_LITE_counter_axilite_setup = counter_axiLite_setup;
    assign AXI_LITE_counter_dequantization_flow = counter_dequantization_flow;
    assign AXI_LITE_counter_quantization_flow = counter_quantization_flow;
    assign AXI_LITE_counter_start_of_quantization_flow = counter_start_of_quantization_flow;
    assign AXI_LITE_counter_extractor = counter_extractor;  
    assign AXI_LITE_counter_dequantization = counter_dequantization;  
    assign AXI_LITE_counter_start_of_dequantization = counter_start_of_dequantization; 
    assign AXI_LITE_counter_quantization = counter_quantization;  
    assign AXI_LITE_counter_compression = counter_compression;  
    assign AXI_LITE_counter_start_of_compression = counter_start_of_compression; 
`endif // USE_PERFORMANCE_COUNTER

    /*****************************************************************************
        SECTION: PERFORMANCE COUNTER
            - END
    *****************************************************************************/




    
endmodule
