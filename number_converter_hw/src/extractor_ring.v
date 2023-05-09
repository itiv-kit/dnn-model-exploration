/*

receives 32-bit values - extract signed ints with dynamic bitwidth


*/
`include "definitions.v"
// `define USE_PERFORMANCE_COUNTER

//TODO: idea: implement this twice / multiple to fasten up the extraction - necessary? check again when the bottleneck is defined
module extractor_ring
#(  parameter MAX_BITWIDTH_QUANTIZED_DATA = 16,
    parameter MAX_BITWIDTH_VALID_BITS = (2**MAX_BITWIDTH_QUANTIZED_DATA) - 1
) (
    input wire clk, rstn,
    //input handshake
    input wire [15:0] mask_valid_bits, //TODO: change this to simple last bit ready and do another last value handling
    input wire [4:0] bitwidth_d, 
    input wire [31:0] num_of_input_vals,
    input wire rcv_valid,
    // input wire [MEMORY_BITWIDTH-2:0] validation_bits,
    input wire [15:0] rcv_data, 
    // input wire last_value,
    output wire rcv_ready,  //optional

    //output handshake
    output wire trm_valid,
    output wire [15:0] trm_data,
    input wire trm_ready
`ifdef USE_PERFORMANCE_COUNTER
    ,output wire finished 
`endif
    //TODO: possible to reduce bitwidth by one bit due to the fact that the value range of absolute quantized value needs one bit less?
    //additional output
);

/*

    higher byte    lower byte 
  id:      1           0
  |--------------|-------------|
        ram / bit_is_valid

*/

    reg  [31:0] ram;
    // reg  [31:0] bit_is_valid;
    reg  [4:0] multiplex_regs [15:0];
    wire  [15:0] single_data;

    reg received_byte_id;
    reg [1:0] initialized;
    reg last_multiplex_regs_0_msb;
    wire done;

    wire reg_0_higher_last, reg_0_lower_last;

    reg [31:0] transmitted_values; 

    genvar k;
    integer l;



//    assign lower_byte_invalid = bit_is_valid[15:0] == 16'h0;
//    assign higher_byte_invalid = bit_is_valid[31:16] == 16'h0;

    generate
        for (k = 0; k < 16; k = k +1) begin
            ram_multiplex_unit_32 #(.INPUT_WIDTH(32), .REG_WIDTH(5)) ram_bit (ram, multiplex_regs[k], single_data[k]);
        end    
    endgenerate

    always @(posedge clk ) begin
        if ( !rstn) begin
            for(l = 0; l < 16; l = l +1) begin
                // bit_is_valid[l] <= 0;
                multiplex_regs[l] <= l;
            end
            received_byte_id <= 0;
            transmitted_values <= 0;
            initialized <= 0;
        end
        else begin
            last_multiplex_regs_0_msb <= multiplex_regs[0][4];
            if (!initialized[0] ) begin
                ram [15:0] <=  rcv_data ;
                if (rcv_valid & rcv_ready) begin
                    initialized[0] <= 1'b1;
                end
            end
            else if (!initialized[1]) begin
                ram [31:16] <=  rcv_data ;
                if (rcv_valid & rcv_ready) begin
                    initialized[1] <= 1'b1;
                end                
            end
            else begin
                if ( trm_ready & !done) begin
                    for (l = 0;l < 16;l = l +1) begin
                        multiplex_regs[l] <= multiplex_regs[l] + bitwidth_d;
                    end   
                    
                ram [15:0] <=  rcv_ready & ~received_byte_id ? rcv_data : ram[15:0];

                ram [31:16] <= rcv_ready & received_byte_id ? rcv_data : ram[31:16];

                received_byte_id <=   rcv_ready ? !received_byte_id : received_byte_id;

                transmitted_values <= trm_valid ? transmitted_values +1 : transmitted_values;
                end
            end
        end
    end

    assign trm_data = single_data & mask_valid_bits;
    assign trm_valid = ( trm_ready & (initialized == 2'b11 )  & !done);


    assign rcv_ready = !done & ( (rcv_valid & (reg_0_higher_last | reg_0_lower_last)) || (rcv_valid & !(initialized == 2'b11 ) & rstn));
    
    assign done = !rstn ? 1'b0 : transmitted_values >= num_of_input_vals  ? 1'b1 : 1'b0;

`ifdef USE_PERFORMANCE_COUNTER
    assign finished = done;
`endif


    assign reg_0_higher_last = multiplex_regs[0] > 31 - bitwidth_d;
    assign reg_0_lower_last = multiplex_regs[0] < 16 && multiplex_regs[0] > 15-bitwidth_d;

endmodule

