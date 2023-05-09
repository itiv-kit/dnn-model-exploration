/*

receives signed ints with 16 bitwidth and returns stacked 32-bit values


*/
`include "definitions.v"

module compressor
#(  parameter MAXBITWIDTH = 16,
    parameter OUTPUT_BITWIDTH = 16, 
    parameter BUFFER_SIZE = OUTPUT_BITWIDTH+2*MAXBITWIDTH, //doubled maxbitwidth to be able to udndo overflows (48 bits)
    parameter OUTPUT_LSB = MAXBITWIDTH,
    parameter OUTPUT_MSB = OUTPUT_LSB + OUTPUT_BITWIDTH-1
) (
    input wire clk, rstn,
    //layer static data
    input wire [MAXBITWIDTH-1:0] mask_valid_bits, 
    input wire [$clog2(MAXBITWIDTH):0] bitwidth_d,
    //input handshake
    input wire rcv_valid,
    input wire [MAXBITWIDTH-1:0] rcv_data,
    // input wire rcv_last,
    input wire [31:0] num_of_output_values, 
    output reg rcv_ready, 

    //output handshake
    output reg trm_valid,
    output wire [OUTPUT_BITWIDTH-1:0] trm_data,
    output wire trm_last,
    input wire trm_ready

`ifdef USE_DBG_VIVADO 
    ,output wire [3:0] DBG_fsm,
    output wire [BUFFER_SIZE-1:0] DBG_loop_register__valid 
`endif 
);

    // // REGISTERS - START
    // // output
    reg tmp__rcv_ready,  tmp__trm_last, tmp__trm_valid;
    wire tmp__rcvd_last;

    // other
    // TODO: length of fsm reg ok?
    reg [2:0] tmp__fsm;  
    
    //TODO: calculate how many bits are really necessary behind the real output bits
    reg [(BUFFER_SIZE)-1:0] loop_register__data;
    reg [(BUFFER_SIZE)-1:0] loop_register__valid;

    reg [$clog2(MAXBITWIDTH):0] temp__number_shifts, shifting_module__shifts; 

    wire tmp__reg__lower_byte__complete_valid;
    wire tmp__reg__buffer_excpet_output_valid_somewhere;
    wire tmp__reg__buffer_overflow_happend;


    reg [31:0] tmp__cnt_num_of_values;

    wire tmp__ram_expect_overflow, tmp__ram_expect_ready;

    reg [4:0] prev_tzc;
    wire [MAXBITWIDTH-1:0] pow_shifts;

    wire tzd_z;
    wire [3:0] tzd_out;

    wire tzd_z_l;
    wire [3:0] tzd_out_l;
    wire tzd_z_h;
    wire [3:0] tzd_out_h;
    
    // //REGISTERS - END


    always @(posedge(clk)) begin
        if (!rstn)
        begin
            loop_register__valid <= 0;
            tmp__fsm <= 0;
            rcv_ready <= 0;
            trm_valid <= 0;
            tmp__trm_last <= 0;
            temp__number_shifts <= 0;
            tmp__cnt_num_of_values <= 0;
        end
        else 
        begin
            prev_tzc <= bitwidth_d - tzd_out;
            case (tmp__fsm)
                0: begin
                    trm_valid <= 0;
                    if (!tmp__trm_last) begin
                        if (rcv_valid) begin
                            tmp__fsm <= 1;
                        end
                    end
                end
                1: begin // new data ready for storing
                    if (trm_valid) begin
                        trm_valid <= 1'b0;
                        loop_register__valid[OUTPUT_MSB:OUTPUT_LSB] <= 0;
                    end
                    if(rcv_valid) begin
                        loop_register__data[BUFFER_SIZE-1:BUFFER_SIZE-MAXBITWIDTH] <= rcv_data; 
                        loop_register__valid[BUFFER_SIZE-1:BUFFER_SIZE-MAXBITWIDTH] <= mask_valid_bits; 
                        tmp__cnt_num_of_values <= tmp__cnt_num_of_values + 1;
                        rcv_ready <= 1'b1;
                        tmp__fsm <= 2;
                    end
                end
                2: begin // shift buffer with new value
                    loop_register__data <= loop_register__data >> bitwidth_d;
                    loop_register__valid <= loop_register__valid >> bitwidth_d;
                    rcv_ready <= 1'b0;
                    if (tmp__ram_expect_overflow) begin
                        tmp__fsm <= 4;
                        loop_register__valid <= 0;
                    end
                    else if (tmp__rcvd_last) begin
                        if (loop_register__valid[OUTPUT_LSB]) begin
                            if (trm_ready) begin
                                trm_valid <= 1'b1;
                                tmp__trm_last <= 1'b1;
                                tmp__fsm <= 0;
                            end
                        end
                        else begin
                            tmp__fsm <= 3;
                        end
                    end
                    else if (rcv_valid) begin
                        if (trm_ready) begin
                            trm_valid <= tmp__ram_expect_ready;
                            tmp__fsm <= 1;
                        end
                    end
                    else begin
                        tmp__fsm <= 0;
                    end
                end
                3: begin // last bit received but data isn't assigned to the right
                    if (!tzd_z) begin // diff is smaler than maxbitwidth
                        if (trm_ready) begin
                            loop_register__data <= loop_register__data >> tzd_out;
                            // loop_register__valid <= loop_register__valid >> tzd_out;
                            trm_valid <= 1'b1;
                            tmp__trm_last <= 1'b1;
                            tmp__fsm <= 0;
                        end
                    end 
                    else begin //diff is larger than maxbitwidth
                        loop_register__data <= loop_register__data >> MAXBITWIDTH;
                        loop_register__valid <= loop_register__valid >> MAXBITWIDTH;
                    end
                end
                4: begin //overflow happend -> rework buffer to send data
                    loop_register__data <= loop_register__data << (prev_tzc);
                    loop_register__valid[BUFFER_SIZE-1:OUTPUT_MSB+1] <= pow_shifts;
                    temp__number_shifts <= prev_tzc; //TODO: maybe just do pipeline globally with two stages

                    if (trm_ready) begin
                        trm_valid <= 1'b1;
                        tmp__fsm <= 5;
                    end
                end
                5: begin //shift values back after overflow
                    loop_register__data <= loop_register__data >> temp__number_shifts;
                    loop_register__valid <= loop_register__valid >> temp__number_shifts;
                    temp__number_shifts <= 0;
                    trm_valid <= 0;

                    if (tmp__rcvd_last) begin
                        if (loop_register__valid[OUTPUT_LSB]) begin
                            if (trm_ready) begin
                                trm_valid <= 1'b1;
                                tmp__trm_last <= 1'b1;
                                tmp__fsm <= 0;
                            end
                        end
                        else begin
                            tmp__fsm <= 3;
                        end
                    end
                    else if (rcv_valid) begin
                        tmp__fsm <= 1;
                    end
                    else begin
                        tmp__fsm <= 0;
                    end
                end
                default: begin
                    tmp__fsm <= 0;
                end
            endcase
        end
    end

    assign trm_last = tmp__trm_last;
    assign trm_data = loop_register__data[OUTPUT_MSB:OUTPUT_LSB];

    assign  tmp__rcvd_last = (tmp__cnt_num_of_values == num_of_output_values) ? 1'b1 : 1'b0; //TODO: shouldn't it be enough to do this as as a wire and the incremtn in this state, check of last bit happens in next state (which will alwaxys be reached)


    assign tmp__ram_expect_overflow = tzd_out < bitwidth_d  && !tzd_z ? 1'b1 : 1'b0; 
    assign tmp__ram_expect_ready = tzd_out == bitwidth_d && !tzd_z ? 1'b1 : 1'b0; 

    assign pow_shifts = (2**prev_tzc)-1;


`ifdef USE_DBG_VIVADO
    assign DBG_fsm = tmp__fsm;
    assign DBG_loop_register__valid = loop_register__valid;
`endif


    tzd tzd_low (loop_register__valid[OUTPUT_LSB+15:OUTPUT_LSB], tzd_z_l, tzd_out_l); // TODO: 15 works out as long as dma output doesn't get less

    // tzd tzd_high (loop_register__valid[OUTPUT_MSB:OUTPUT_LSB+16], tzd_z_h, tzd_out_h); // TODO: 15 works out as long as dma output doesn't get less

    assign tzd_z = tzd_z_l ;//& tzd_z_h;
    assign tzd_out[3:0] = tzd_out_l ; //tzd_z_l ? tzd_out_h : tzd_out_l;
    // assign tzd_out[4] = 1'b0;//tzd_z_l;

endmodule


