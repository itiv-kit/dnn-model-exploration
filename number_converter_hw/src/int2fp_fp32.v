/*

converts given signed integer into a floating point 32-bit value


*/

module int2fp_int32
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 16,
        INIT_EXPONENT = 127+ MAX_BITWIDTH_QUANTIZED_DATA -1  //fix -1
) (
    input wire clk, rstn,
    
    //input handshake
    input wire values_rdy,
   input wire sign,
    input wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] quantized_d,
    
    //output handshake
    output reg result_rdy,
    output reg [31:0] quantized_fp
);
    reg [0:1] tmp__pipe_valid;
    reg [0:1] tmp__pipe_data_sign;
    reg [7:0] tmp__pipe_data_exponent [0:1];
    reg [15:0] tmp__pipe_data_mantissa ; //fix wegen unten
    reg [0:31] intern_quant_fp; 
    reg intern_quant_fp_valid;

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
            tmp__pipe_data_sign[0] <= sign; 
            if(quantized_d[MAX_BITWIDTH_QUANTIZED_DATA-1:0] == 0 ) begin //EXCEPTION: integer is zero
                tmp__pipe_data_exponent[0] <= 0; 
                temp_in <= 1 << MAX_BITWIDTH_QUANTIZED_DATA-1; 
            end
            else begin
                tmp__pipe_data_exponent[0] <= INIT_EXPONENT; 
                temp_in <= quantized_d; 
            end 
            tmp__pipe_valid[0] <= values_rdy; 
            tmp__pipe_valid[1] <= tmp__pipe_valid[0] ; 

            tmp__pipe_data_mantissa <= temp_in << lzd_out;   
            tmp__pipe_data_exponent[1] <= tmp__pipe_data_exponent[0] - lzd_out+16-MAX_BITWIDTH_QUANTIZED_DATA;
    
            intern_quant_fp <= {tmp__pipe_data_sign[1], tmp__pipe_data_exponent[1], tmp__pipe_data_mantissa[15-1:0]} ;
            intern_quant_fp_valid <= tmp__pipe_valid[1];

            quantized_fp <= {intern_quant_fp, 8'd0}; //change value from LSB-bounding to MSB-bounding
            result_rdy <= intern_quant_fp_valid;
        end
    end



lzd lzd_mod (temp_in, lzd_z, lzd_out);



endmodule

