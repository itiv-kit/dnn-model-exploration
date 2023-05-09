
module tb_compressor_ring ();
	
	reg clk, rstn;

    reg [15:0] mask_valid_bits; //TODO: del
    reg [3:0] bitwidth_d;
    reg rcv_valid;
    reg [15:0] rcv_data;
    reg rcv_ready; 

    reg trm_valid;
    reg [15:0] trm_data;
    reg trm_last;
    reg trm_ready;
    reg [31:0] transmitted_values; 

	compressor_ring dut (
		clk, rstn, 
		mask_valid_bits, bitwidth_d,
		rcv_valid, rcv_data, transmitted_values, rcv_ready,
		trm_valid, trm_data, trm_last, trm_ready);

		always begin
        #5 clk = ~clk;
    end

	initial begin
		clk <= 0;
		rstn <= 0;
		rcv_data <= 16'h0f03;
		bitwidth_d <= 4'd3;
		transmitted_values <= 32'h4;
		rcv_valid <= 1;

		#30 rstn <= 1;

		#40 rcv_valid <= 0;

		#40 rcv_valid <= 1;
	end
endmodule