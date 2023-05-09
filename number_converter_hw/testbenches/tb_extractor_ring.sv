
module tb_extractor_ring ();
	reg clk, rstn;

	wire [15:0] mask_valid_bits; //TODO: del
    reg [4:0] bitwidth_d;
    reg rcv_valid;
    reg [15:0] rcv_data;
    reg rcv_ready; 

    reg trm_valid;
    reg [15:0] trm_data;
    reg trm_last;
    reg trm_ready;
    reg [31:0] transmitted_values; 

	reg done;

	extractor_ring dut(
		clk, rstn,
		mask_valid_bits, bitwidth_d,  transmitted_values,
		rcv_valid, rcv_data, rcv_ready,
		trm_valid, trm_data,  trm_ready
`ifdef USE_PERFORMANCE_COUNTER
    ,done 
`endif
	);

	always begin
		#5 clk = ~ clk;
	end

	assign mask_valid_bits = (2**bitwidth_d)-1;

	initial begin
		clk <=0 ;
		rstn <= 0;
		rcv_data <= 16'h4321;
		bitwidth_d <= 4;
		transmitted_values <= 32;
		rcv_valid <= 1;
		trm_ready <= 1;

		#25 rstn <= 1;


		#40 rcv_valid <= 0;

		#40 rcv_valid <= 1;


		#100 trm_ready <= 0;

		#40 trm_ready <= 1;


		#40 rcv_data <= 16'h8765;
		#40 rcv_data <= 16'hcba9;

	end
endmodule