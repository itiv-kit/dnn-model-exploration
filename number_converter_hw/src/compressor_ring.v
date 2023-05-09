/*
BACKLOG:
receives signed ints with 16 bitwidth and returns stacked 32-bit values
*/
`include "definitions.v"

module compressor_ring
#(  
    MAXBITWIDTH = 16,
    OUTPUT_BITWIDTH = 16 
) (
    input wire clk, rstn,
    //layer static data
    input wire [4:0] bitwidth_d,
    //input handshake
    input wire rcv_valid,
    input wire [MAXBITWIDTH-1:0] rcv_data,
    input wire [31:0] num_of_output_values, 
    output wire rcv_ready, 

    //output handshake
    output wire trm_valid,
    output wire [OUTPUT_BITWIDTH-1:0] trm_data,
    output wire trm_last,
    input wire trm_ready

`ifdef USE_DBG_VIVADO 
    ,output wire [3:0] DBG_fsm,
    output wire [BUFFER_SIZE-1:0] DBG_loop_register__valid 
`endif 
);

/*

    higher byte    lower byte 
        1               0
  |--------------|-------------|
        ram / bit_is_valid

*/

    parameter RAM_SIZE = 32;

    wire [RAM_SIZE-1:0] ram ;
    // reg  [RAM_SIZE-1:0] bit_is_valid ;
    reg [4:0] multiplex_regs [31:0];

    wire lower_byte_valid, higher_byte_valid;

    reg transmit_byte_id;


    reg [31:0] received_values; 
    
    genvar k;
    integer l;


    assign lower_byte_valid = multiplex_regs[16] < bitwidth_d;
    assign higher_byte_valid =  multiplex_regs[0] < bitwidth_d;



    generate
        for (k = 0; k < 32; k = k +1) begin
            ram_multiplex_unit_16 #(.INPUT_WIDTH(16), .REG_WIDTH(4)) ram_bit (rcv_data, multiplex_regs[k][3:0], multiplex_regs[k][4], ram[k]);
        end    
    endgenerate

    always @(posedge(clk)) begin
        if(!rstn) begin
            for (l = 0;l < 32;l = l +1) begin
                // bit_is_valid[l] <= 0;
                multiplex_regs[l] <= l;
            end
            transmit_byte_id <= 0;
            received_values <= 0;
        end
        else begin    
            if (rcv_valid & trm_ready) begin
                for (l = 0;l < 16;l = l +1) begin
                    // bit_is_valid[l] <= bit_is_valid[l]  ? (lower_byte_valid ? (multiplex_regs[l] < bitwidth_d) : bit_is_valid[l]) : (multiplex_regs[l] < bitwidth_d);
                    multiplex_regs[l] <= multiplex_regs[l] - bitwidth_d;
                end   
                for (l = 16;l < 32;l = l +1) begin
                    // bit_is_valid[l] <= bit_is_valid[l]  ? (higher_byte_valid ? ((multiplex_regs[l] < bitwidth_d)) : bit_is_valid[l]) : (multiplex_regs[l] < bitwidth_d);
                    multiplex_regs[l] <= multiplex_regs[l] - bitwidth_d;
                end   
            end
            transmit_byte_id <= trm_valid ? ~transmit_byte_id : transmit_byte_id; 
            received_values <= rcv_valid ? received_values + 1 : received_values;
        end
    end

    assign trm_valid = (!transmit_byte_id & lower_byte_valid ) | ( transmit_byte_id & higher_byte_valid) ;
    assign trm_data = transmit_byte_id ? ram[31:16] : ram[15:0];

    assign trm_last = received_values >= num_of_output_values ? 1 : 0;

    assign rcv_ready = 1'b1;

endmodule

