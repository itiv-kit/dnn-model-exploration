

module tb_ram_multiplex_unit ();
	
	reg [15:0] data ;
	reg [3:0] regs;
	reg is_valid, ram_bit;

	integer i;

	ram_multiplex_unit bit_0 (data, regs, is_valid, ram_bit);


	initial begin
		data <= 16'b1000111010001110;
		is_valid <= 0;
		regs <= 0;

		for (i = 0; i < 20; i = i +1) begin
			#30 regs <= regs + 1;
		end


		#90 is_valid <= 1;
		regs <= 0;

		for (i = 0; i < 20; i = i +1) begin
			#30 regs <= regs + 1;
		end
	end
	
endmodule