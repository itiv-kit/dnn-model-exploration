/*

receives 32-bit values - extract signed ints with dynamic bitwidth


*/
`include "definitions.v"
// `define USE_PERFORMANCE_COUNTER

//TODO: idea: implement this twice / multiple to fasten up the extraction - necessary? check again when the bottleneck is defined
module extractor
#(  parameter MAX_BITWIDTH_QUANTIZED_DATA = 16,
    parameter MAX_BITWIDTH_VALID_BITS = (2**MAX_BITWIDTH_QUANTIZED_DATA) - 1,
    parameter MEMORY_BITWIDTH = 16, MEMORY_MASK = (2 ** MEMORY_BITWIDTH) -1 
) (
    input wire clk, rstn,
    //input handshake
    input wire values_rdy,
    input wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] mask_valid_bits, //TODO: change this to simple last bit ready and do another last value handling
    input wire [$clog2(MAX_BITWIDTH_QUANTIZED_DATA):0] bitwidth_d, 
    input wire [31:0] num_of_input_vals,
    // input wire [MEMORY_BITWIDTH-2:0] validation_bits,
    input wire [MEMORY_BITWIDTH-1:0] value, 
    // input wire last_value,
    output wire out_rdy,  //optional

    //output handshake
    input wire next_module_rdy,
    output wire result_rdy,
    output wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] result
`ifdef USE_PERFORMANCE_COUNTER
    ,output reg done 
`endif
    //TODO: possible to reduce bitwidth by one bit due to the fact that the value range of absolute quantized value needs one bit less?
    //additional output
);

    // REGISTERS - START
    // output
    reg out__out_rdy,  transmitting_rdy;
    wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] out__quantized_value_maxbitwidth,  out__quantized_value_maxbitwidth_validbits;
    wire [MAX_BITWIDTH_QUANTIZED_DATA-1:0] out__next_quantized_value_maxbitwidth,  out__next_quantized_value_maxbitwidth_validbits;
    wire tmp__next_result_rdy, tmp__valid_bits_higher_byte__last_value, tmp__num_of_values_done;
    reg [MAX_BITWIDTH_QUANTIZED_DATA-1:0] result_delay; 
    // reg sign;
    reg tmp__result_rdy_delay;
    reg [6:0] cnt_shifts__invalid_bits;

    reg [3:0] tmp__fsm;  
    reg [31:0] tmp__num_of_vals;

    reg [(2*MEMORY_BITWIDTH)-1:0] loop_register__data;
    reg [(2*MEMORY_BITWIDTH)-1:0] loop_register__valid;

    



    wire tmp__result_rdy;
    wire tmp__valid_bits_lower_byte__anywhere_high, tmp__valid_bits_higher_byte__anywhere_high;
    

    wire lzd_z;
    wire [3:0] lzd_out;
    

    //REGISTERS - END

    /*
    loop_register:
          vv INPUT DATA vv
    |------------------------|------------------------|
    |      higher byte       |      lower byte        |
    |------------------------|------------------------|
                                  vv OUTPUT DATA vv
    */


    // input
    always @(posedge(clk)) begin
        if (!rstn)
        begin
            loop_register__valid <= 0;
            tmp__fsm <= 0;
            transmitting_rdy <= 1'b0;
            out__out_rdy <= 1'b0;
            tmp__num_of_vals <= 0;

`ifdef USE_PERFORMANCE_COUNTER
            done <= 0;
`endif        
        end
        else 
        begin
            if (out__out_rdy) begin
                out__out_rdy <= 1'b0;
            end
            else if(tmp__fsm == 0  && values_rdy) begin
                loop_register__data[MEMORY_BITWIDTH-1:0] <= value;
                loop_register__valid[MEMORY_BITWIDTH-1:0] <= MEMORY_MASK;
                // tmp__fsm <= 1;
                out__out_rdy <= 1'b1;
            end
            else if(tmp__valid_bits_higher_byte__anywhere_high & values_rdy) begin
                out__out_rdy <= 1'b1;
                loop_register__data[2*MEMORY_BITWIDTH-1:MEMORY_BITWIDTH] <= value;
                loop_register__valid[2*MEMORY_BITWIDTH-1:MEMORY_BITWIDTH] <= MEMORY_MASK;
            end

            // TODO: possible to reduce clock idle cycles and complexity?
            case (tmp__fsm)
                0: begin
                    if ( loop_register__valid[0] == 1) begin
                        tmp__fsm <= 1;
                    end
                end
                1: begin //idle
                    transmitting_rdy <= 1'b0;
// `ifdef USE_PERFORMANCE_COUNTER
                    if ( !tmp__num_of_values_done ) begin
// `endif
                        if (loop_register__valid[(MEMORY_BITWIDTH)] == 1 && loop_register__valid[MEMORY_BITWIDTH-1] == 0 && loop_register__valid[0] == 1 ) begin
                            tmp__fsm <= 3;
                            cnt_shifts__invalid_bits <= 0;
                            // transmitting_rdy <= 1'b0;
                            // tmp__num_of_vals <= tmp__num_of_vals +1;
                        end
                        else if(tmp__result_rdy && loop_register__valid != 0 ) begin
                                tmp__fsm <= 2;
                                transmitting_rdy <= 1'b1;
                        end
                    end
`ifdef USE_PERFORMANCE_COUNTER
                    else begin
                        done <= 1'b1;
                    end 
`endif                       
                end 
                2: begin // transmit value
                        out__out_rdy <= 1'b0;
                    if (loop_register__valid[(2*MEMORY_BITWIDTH)-1] == 1 && loop_register__valid[MEMORY_BITWIDTH-1] == 0 ) begin
                        tmp__fsm <= 3;
                        cnt_shifts__invalid_bits <= lzd_out;
                        transmitting_rdy <= 1'b0;
`ifdef USE_PERFORMANCE_COUNTER
                        tmp__num_of_vals <= tmp__num_of_vals +1;
`endif                        
                    end
                    else if (next_module_rdy) begin
                        transmitting_rdy <= 1'b1;
                        loop_register__data <= loop_register__data >> bitwidth_d;
                        loop_register__valid <= loop_register__valid >> bitwidth_d;

                        if(tmp__valid_bits_higher_byte__last_value & values_rdy) begin
                            tmp__fsm <= 4;
                            transmitting_rdy <= 1'b0;
                        end 
                        else if (!tmp__result_rdy || tmp__num_of_values_done) begin
                            tmp__fsm <= 1;
                            transmitting_rdy <= 1'b0;
                        end
                        else begin
                            // transmitting_rdy <= 1'b1;
                        end
`ifdef USE_PERFORMANCE_COUNTER
                        tmp__num_of_vals <= tmp__num_of_vals +1;
`endif                        
                    end
                end
                3: begin //invalid bits after new block of data arrived --> shiftings required
                 transmitting_rdy <= 1'b0;
                    if(loop_register__valid[(2*MEMORY_BITWIDTH)-1] == 1 && loop_register__valid[MEMORY_BITWIDTH-1] == 0 ) begin
                        loop_register__data[MEMORY_BITWIDTH-1:0] <=  loop_register__data[MEMORY_BITWIDTH-1:0] << cnt_shifts__invalid_bits;
                        loop_register__valid[MEMORY_BITWIDTH-1:0] <=  loop_register__valid[MEMORY_BITWIDTH-1:0] << cnt_shifts__invalid_bits;
                        cnt_shifts__invalid_bits <= cnt_shifts__invalid_bits + bitwidth_d;
                    end
                    else begin
                        loop_register__data[2*MEMORY_BITWIDTH-1:0] <=  loop_register__data[2*MEMORY_BITWIDTH-1:0] >> cnt_shifts__invalid_bits ;
                        loop_register__valid[2*MEMORY_BITWIDTH-1:0] <=  loop_register__valid[2*MEMORY_BITWIDTH-1:0] >> cnt_shifts__invalid_bits;
                        tmp__fsm <= 1;
                    end
                end
                4: begin
                     transmitting_rdy <= 1'b0;
                    // if(tmp__valid_bits_higher_byte__anywhere_high & values_rdy) begin
                        out__out_rdy <= 1'b1;
                        loop_register__data[2*MEMORY_BITWIDTH-1:MEMORY_BITWIDTH] <= value;
                        loop_register__valid[2*MEMORY_BITWIDTH-1:MEMORY_BITWIDTH] <= 16'hFFFF;
                    // end
                        if (tmp__result_rdy) begin
                            tmp__fsm <= 2;
                            transmitting_rdy <= 1'b1;
                        end
                        else begin 
                            tmp__fsm <= 1;
                            // transmitting_rdy <= 1'b0;
                        end
                    // tmp__fsm <= tmp__result_rdy ? 2 : 1;
                    end
                default: begin
                    tmp__fsm <= 0;
                end
            endcase
            
        end

        
    end
    
    assign result_rdy = transmitting_rdy ;
    assign  result = out__quantized_value_maxbitwidth;//sign ? {sign, ~result_delay +1 }  : {sign, result_delay} ;
    assign out_rdy = out__out_rdy;

    assign tmp__num_of_values_done = tmp__num_of_vals > (num_of_input_vals -1) ? 1'b1 : 1'b0;

    // high when new value ready to transmit
    assign tmp__result_rdy = (out__quantized_value_maxbitwidth_validbits == MAX_BITWIDTH_VALID_BITS) ? 1'b1 : 1'b0;

    // high when next value  will be ready to transmit
    assign tmp__next_result_rdy = (out__next_quantized_value_maxbitwidth_validbits == MAX_BITWIDTH_VALID_BITS) ? 1'b1 : 1'b0;

    // high when  higher byte is invalid
    assign tmp__valid_bits_higher_byte__anywhere_high = loop_register__valid[2*MEMORY_BITWIDTH-1:MEMORY_BITWIDTH] == 0 ? 1'b1 : 1'b0;

    // high when last value in  higher byte
    assign tmp__valid_bits_higher_byte__last_value = mask_valid_bits >= loop_register__valid [2*MEMORY_BITWIDTH-1:MEMORY_BITWIDTH] ? 1'b1 : 1'b0;

    //TODO: better solution?
    genvar l;
    generate
        for (l = 0; l < MAX_BITWIDTH_QUANTIZED_DATA; l = l+1) begin //sets the out register
            assign out__quantized_value_maxbitwidth[l] = loop_register__data[l] && mask_valid_bits[l];
            assign out__quantized_value_maxbitwidth_validbits[l] =  !mask_valid_bits[l] || loop_register__valid[l]; //inverting for accepting, when upper bits are not valid
        end
    endgenerate
    // genvar k;
    generate
        for (l = 0; l < MAX_BITWIDTH_QUANTIZED_DATA; l = l+1) begin //sets the out register
            assign out__next_quantized_value_maxbitwidth[l] = loop_register__data[l+bitwidth_d] && mask_valid_bits[l];
            assign out__next_quantized_value_maxbitwidth_validbits[l] =  !mask_valid_bits[l+bitwidth_d] || loop_register__valid[l]; //inverting for accepting, when upper bits are not valid
        end
    endgenerate

    //TODO: does this solution really reduce the design?
    // assign out__quantized_value_maxbitwidth = loop_register__data & mask_valid_bits;
    // assign out__quantized_value_maxbitwidth_validbits = (~mask_valid_bits) | loop_register__valid; 



    lzd lzd_h (loop_register__valid[MEMORY_BITWIDTH-1:MEMORY_BITWIDTH-16], lzd_z, lzd_out);

endmodule

