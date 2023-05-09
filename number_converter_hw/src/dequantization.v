/*
    dequantizes a 16-bit quantizd value with x-bitwidth into a fp32 value

*/

module dequantization
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 16
) (
    input wire clk, rstn,

    //input handshake
    input wire in_valid,
    input wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] in_data,
    input wire [15:0] scale_fp,
    input wire [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] bitwidth_d,
    input wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] mask_valid_bits, 
    // output wire in_ready,  

    //output handshake
    output wire out_valid,
    output wire [15:0] out_data
);

    // REGISTERS - START

    //converter
    wire  converter__out_valid;
    wire [24:0] converter__out_data;
    wire [24:0] mult_out_data;
    // reg [31:0] counter_in, counter_out;

    // wire tmp__in_valid__in_ready;

    // wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] input_d_abs; 
    // wire sign;

    //REGISTERS - END

    // assign input_d_abs = in_data[bitwidth_d-1]  ? {~((~mask_valid_bits) | in_data) + 1 } : in_data; 
    // assign sign = in_data[bitwidth_d-1] ;

    int2fp_bfloat16 #(.MAX_BITWIDTH_QUANTIZED_DATA(MAX_BITWIDTH_QUANTIZED_DATA)) converter 
        (clk,  rstn,
        in_valid, bitwidth_d, mask_valid_bits, in_data ,
        converter__out_valid,  converter__out_data);

    fp_multiplication_bfloat16  multiplier 
        (clk,  rstn,
        converter__out_valid, converter__out_data, {scale_fp, 9'd0}, 
        out_valid, mult_out_data);

        assign out_data = mult_out_data[24:9];


endmodule


