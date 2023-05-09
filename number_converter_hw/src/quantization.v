/*
quantize fp_value to a signed 16-bit integer with internal x bitwidth


*/

module quantization
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 16
) (
    input wire clk, rstn,
    //input handshake
    input wire values_rdy,
    input wire [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] bitwidth, 
    input wire [15:0] fp_value,
    input wire [15:0] scale_fp,
    //additional input

    //output handshake
    output wire result_rdy,
    output wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] result
    //additional output

);

    // REGISTERS - START
    // output
    wire tmp_module_rdy, tmp_result_rdy;
    wire [15:0] tmp_result;

    //converter
    wire  converter__result_rdy;
    wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] converter__quantized_fp;

    //multiplier
    wire multiplier__values_rdy, multiplier__rdy, multiplier__result_rdy;
    wire [24:0]  multiplier_result;

    //REGISTERS - END

    fp_multiplication_bfloat16  multiplier ( 
        clk,  rstn,
        values_rdy, {fp_value, 9'd0}, {scale_fp, 9'd0}, 
        multiplier__result_rdy, multiplier_result);


    fp2int_bfloat16 #(.MAX_BITWIDTH_QUANTIZED_DATA(MAX_BITWIDTH_QUANTIZED_DATA)) converter (
        clk, rstn, 
        multiplier__result_rdy, bitwidth, multiplier_result,  
        converter__result_rdy, converter__quantized_fp);


    // output
    assign result_rdy = converter__result_rdy;
    assign result = converter__quantized_fp;    

endmodule


