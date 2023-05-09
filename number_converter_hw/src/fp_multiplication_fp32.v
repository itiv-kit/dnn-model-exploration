/*

multiplies two floating point 32-bit values and returns result


*/

module fp_multiplication_fp_32
#(
    parameter PIPELINE_LENGTH = 6 //fix TODO: tbd
) (
    input wire clk, rstn,

    //input handshake
    input wire values_rdy,
    input wire [31:0] fp_value_1,
    input wire [31:0] fp_value_2,

    //output handshake
    output reg result_rdy,
    output reg [31:0] result
);

    // REGISTERS - START
    //TODO: reduce sizes of registers
    reg [7:0] tmp__exponent_products [1:PIPELINE_LENGTH-1] ;
    reg [7:0] tmp__exponent_1  ;
    reg [7:0] tmp__exponent_2  ;
    reg [22:0] tmp__mantissa_1  ; //one less for DSP in MENTA
    reg [22:0] tmp__mantissa_2 ; //one less for DSP in MENTA
    reg [45:0] tmp__mantissas_product ;//[0:1] ; //Hint: add second stage for better pipelining in xilinx fpga
    reg [24:0] tmp__mantissa [2:PIPELINE_LENGTH-1] ; // silenced MSB and additional LSB-1


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
            tmp__product_sign[0] <=  fp_value_1[31] ^ fp_value_2[31];   

            tmp__exponent_1 <= {1'b0,  fp_value_1[30:23]} ; 
            tmp__exponent_2 <= {1'b0,  fp_value_2[30:23]} ; 
            tmp__mantissa_1 <= fp_value_1[30:23]  == 0 ? 0 : {1'b1, fp_value_1[22:1]}  ;
            tmp__mantissa_2 <= fp_value_2[30:23]  == 0 ? 0 : {1'b1, fp_value_2[22:1]}  ;
             

            tmp__exponent_products[1] <=  tmp__exponent_1 +  tmp__exponent_2 ;
            tmp__mantissas_product <=  tmp__mantissa_1 * tmp__mantissa_2;
            // tmp__mantissas_product[1] <= tmp__mantissas_product[0]; //2nd stage for vivado

            
            tmp__exponent_products[2]  <=  tmp__exponent_products[1]   + 8'b10000001 + 1; //-127 ;
            tmp__mantissa[2] <= tmp__mantissas_product[45:21] ;//>> 1 ;

            // shift if msb in mantissa is zero
            if(!tmp__mantissa[2][24] ) begin
                tmp__mantissa[3] <= tmp__mantissa[2] << 1;
                tmp__exponent_products[3] <= tmp__exponent_products[2] + 8'b11111111;
            end 
            else begin
                tmp__mantissa[3] <= tmp__mantissa[2] ;
                tmp__exponent_products[3]<= tmp__exponent_products[2];
            end

            // shift if msb in mantissa is zero
            if(!tmp__mantissa[3][24] ) begin
                tmp__mantissa[4] <= tmp__mantissa[3] << 1;
                tmp__exponent_products[4] <= tmp__exponent_products[3] + 8'b11111111;
            end 
            else begin
                tmp__mantissa[4] <= tmp__mantissa[3] ;
                tmp__exponent_products[4] <= tmp__exponent_products[3];
            end

            // adaption
            if (tmp__mantissa[4][0] ) begin
                tmp__mantissa[5] <= tmp__mantissa[4] +1;
                tmp__exponent_products[5] <= tmp__exponent_products[4];
            end
            else begin 
                tmp__mantissa[5] <= tmp__mantissa[4] ;
                tmp__exponent_products[5] <= tmp__exponent_products[4];
            end
            
        end
    
        result_rdy <= tmp__pipe_valid[PIPELINE_LENGTH-1]; 
        result <= tmp__mantissa[PIPELINE_LENGTH-1][24:1]  == 0 ? {tmp__product_sign[PIPELINE_LENGTH-1], 8'd0 ,  23'd0  } :  {tmp__product_sign[PIPELINE_LENGTH-1], tmp__exponent_products[PIPELINE_LENGTH-1][7:0] ,  tmp__mantissa[PIPELINE_LENGTH-1][23:1]  };
    end

    

endmodule
