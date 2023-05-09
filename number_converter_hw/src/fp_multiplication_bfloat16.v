/*

multiplies two floating point 32-bit values and returns result


*/

module fp_multiplication_bfloat16
#(
    parameter PIPELINE_LENGTH = 6 //fix TODO: tbd
) (
    input wire clk, rstn,

    //input handshake
    input wire values_rdy,
    input wire [24:0] fp_value_1,
    input wire [24:0] fp_value_2,

    //output handshake
    output reg result_rdy,
    output reg [24:0] result
);

    // REGISTERS - START
    //TODO: reduce sizes of registers
    reg [8:0] tmp__exponent_products [1:PIPELINE_LENGTH-1] ;
    reg [7:0] tmp__exponent_1  ;
    reg [7:0] tmp__exponent_2  ;
    reg [16:0] tmp__mantissa_1  ; //one less for DSP in MENTA
    reg [16:0] tmp__mantissa_2 ; //one less for DSP in MENTA
    reg [33:0] tmp__mantissas_product ;//[0:1] ; //Hint: add second stage for better pipelining in xilinx fpga
    reg [17:0] tmp__mantissa [2:PIPELINE_LENGTH-1] ; // silenced MSB and additional LSB-1


    reg [PIPELINE_LENGTH-1:0] tmp__product_sign, tmp__pipe_valid;


    //REGISTERS - END


    // input
    always @(posedge(clk)) begin
        if (!rstn) begin
            tmp__pipe_valid <= 0;
        end
        else begin
            tmp__pipe_valid <= tmp__pipe_valid << 1;
            tmp__pipe_valid[0] <= values_rdy;
            //sign done
            tmp__product_sign <= tmp__product_sign << 1; 
            tmp__product_sign[0] <=  fp_value_1[24] ^ fp_value_2[24];   

            tmp__exponent_1 <= fp_value_1[23:16] == 0 || fp_value_2[23:16] == 0 ? 9'd0 : {1'b0,  fp_value_1[23:16]} ; 
            tmp__exponent_2 <= fp_value_1[23:16] == 0 || fp_value_2[23:16] == 0 ? 9'd0 : {1'b0,  fp_value_2[23:16]} ; 
            tmp__mantissa_1 <= fp_value_1[23:16] == 0 ? 0 : {1'b1, fp_value_1[15:0]}  ;
            tmp__mantissa_2 <= fp_value_2[23:16]  == 0 ? 0 : {1'b1, fp_value_2[15:0]}  ;
             

            tmp__exponent_products[1] <=  tmp__exponent_1 +  tmp__exponent_2 ;
            tmp__mantissas_product <=  tmp__mantissa_1 * tmp__mantissa_2;
            // tmp__mantissas_product[1] <= tmp__mantissas_product[0]; //2nd stage for vivado

            
            tmp__exponent_products[2]  <=  tmp__exponent_products[1]   + 8'b10000001 + 1; //-127 ;
            tmp__mantissa[2] <= tmp__mantissas_product[33:16] ;//>> 1 ;

            // rounding up (lsb-1 is high)
            if (tmp__mantissa[2][0] ) begin
                tmp__mantissa[3] <= tmp__mantissa[2] + 18'd1; //if one, then rounding up, else nothing happens
                tmp__exponent_products[3] <= tmp__exponent_products[2];
            end
            else begin 
                tmp__mantissa[3] <= tmp__mantissa[2] ;
                tmp__exponent_products[3] <= tmp__exponent_products[2];
            end

            // shift right if msb is not high
            if(!tmp__mantissa[3][17] ) begin
                tmp__mantissa[4] <= tmp__mantissa[3] << 1;
                tmp__exponent_products[4] <= tmp__exponent_products[3] + 8'b11111111;
            end 
            else begin
                tmp__mantissa[4] <= tmp__mantissa[3] ;
                tmp__exponent_products[4] <= tmp__exponent_products[3];
            end
            
        end
    
        result_rdy <= tmp__pipe_valid[PIPELINE_LENGTH-2]; 
        result <= tmp__mantissa[4][17:1]  == 0 ? {tmp__product_sign[4], 23'd0  } :  {tmp__product_sign[4], tmp__exponent_products[4][7:0] ,  tmp__mantissa[4][16:1]  };
    end

    

endmodule
