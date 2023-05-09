/*
receives integer in floating point formant and converts this into signed integer


*/

module fp2int
#( parameter
        MAX_BITWIDTH_QUANTIZED_DATA = 16
) (
    input wire clk, rstn,

    //input handshake
    input wire values_rdy,
    input wire [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] bitwidth, 
    input wire [31:0] value,
    // output reg out_rdy,  //optional

    //output handshake
    // input wire next_module_rdy,
    output reg result_rdy,
    output reg [MAX_BITWIDTH_QUANTIZED_DATA-1:0] result
);

    // REGISTERS - START
    reg [7:0] tmp__exponent ;
    reg [23:0] tmp__mantissa  ;
    reg [0: 2]  tmp__sign; 
    reg [0: 3]   tmp__valid;
    reg [7:0] shifts ;
    reg [MAX_BITWIDTH_QUANTIZED_DATA-1:0] temp_out [2:0];
    //REGISTERS - END


    // input
    always @(posedge(clk)) begin
        if (!rstn)
        begin
            tmp__valid <= 0;
        end
        else 
        begin
            // shift sign value
            tmp__sign <= tmp__sign >> 1;  
            tmp__sign[0] <= value[31];
            
            // shift valid value
            tmp__valid <= tmp__valid >> 1;  
            tmp__valid[0] <= values_rdy;

            //store mantissa and exponent
            if (value[30:0]  == 0) begin
                tmp__exponent <= 0;
                tmp__mantissa <= 0 ; 
            end
            else begin
                tmp__exponent <= value[30:23] + 1 + 8'b10000001;
                tmp__mantissa <= {1'b1, value[22:0]} ;  
            end

            shifts <= tmp__exponent > bitwidth ? bitwidth : tmp__exponent;

            temp_out[0] <= tmp__mantissa[23:23-MAX_BITWIDTH_QUANTIZED_DATA+1];
            temp_out[1] <= temp_out[0] >> (MAX_BITWIDTH_QUANTIZED_DATA - shifts);

            if (tmp__sign[2]  ) begin
                temp_out[2] <= (~temp_out[1]) + 1; // twos complement
            end 
            else begin
                temp_out[2] <= (temp_out[1]);
            end

        end

        result_rdy <= tmp__valid[3] ;
        result <= temp_out[2] ;
    end

endmodule

