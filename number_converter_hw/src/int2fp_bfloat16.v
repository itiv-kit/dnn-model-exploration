/*

converts given signed integer into a floating point 32-bit value


*/

module int2fp_bfloat16
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 16,
        INIT_EXPONENT = 127+ MAX_BITWIDTH_QUANTIZED_DATA -1, //fix -1
        BITS_EXPONENT = 8,
        BITS_MANTISSA = 16, //INT32: 23, BFLOAT16: 7
        BITS_FLOAT = 1 + BITS_EXPONENT + BITS_MANTISSA
) (
    input wire clk, rstn,
    
    //input handshake
    input wire values_rdy,
    input wire [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] bitwidth_d,
    input wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] mask_valid_bits, 
    input wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] quantized_d,
    
    //output handshake
    output reg result_rdy,
    output reg [24:0] quantized_fp
);
    reg [0:1] tmp__pipe_valid;
    reg [0:1] tmp__pipe_data_sign;
    reg [BITS_EXPONENT-1:0] tmp__pipe_data_exponent [0:1];
    reg [15:0] tmp__pipe_data_mantissa ; //fix wegen unten

    reg [15:0] temp_in = 0;

    wire lzd_z;
    wire [3:0] lzd_out;

    always @(posedge(clk)) begin
        if (!rstn) begin
            tmp__pipe_valid <= 0;
        end
        else begin
            //reading input values 
            tmp__pipe_data_sign[1] <=  tmp__pipe_data_sign[0] ;
            tmp__pipe_data_sign[0] <= quantized_d[bitwidth_d-1]; 

            if(quantized_d[MAX_BITWIDTH_QUANTIZED_DATA-1:0] == 0 ) begin //EXCEPTION: integer is zero
                tmp__pipe_data_exponent[0] <= 0; 
                temp_in <= 1 << MAX_BITWIDTH_QUANTIZED_DATA-1; 
            end
            else begin
                tmp__pipe_data_exponent[0] <= INIT_EXPONENT; 
                temp_in <= quantized_d[bitwidth_d-1]  ? {~((~mask_valid_bits) | quantized_d) + 1 } : quantized_d; 
            end 
            tmp__pipe_valid[0] <= values_rdy; 
            tmp__pipe_valid[1] <= tmp__pipe_valid[0] ; 

            tmp__pipe_data_mantissa <= temp_in << lzd_out;   
            tmp__pipe_data_exponent[1] <= tmp__pipe_data_exponent[0] - lzd_out+16-MAX_BITWIDTH_QUANTIZED_DATA;
    
            quantized_fp <= {tmp__pipe_data_sign[1], tmp__pipe_data_exponent[1], tmp__pipe_data_mantissa[15-1:0], 1'b0}; //TODO:  delete this clock cycle
            result_rdy <= tmp__pipe_valid[1];

            // quantized_fp <= intern_quant_fp; 
            // result_rdy <= intern_quant_fp_valid;
        end
    end



lzd lzd_mod (temp_in, lzd_z, lzd_out);



endmodule

