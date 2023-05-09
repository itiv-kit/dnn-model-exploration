
module ram_multiplex_unit_32 #(
	parameter INPUT_WIDTH = 16,
				REG_WIDTH = 4
)(
	input wire [INPUT_WIDTH-1:0] in_data, // data of  input
	input wire [REG_WIDTH-1:0] in_reg, 	// register storing the number of desired bit
	// input wire is_valid,	// says if the current stored value is valid
	output wire ram_bit	// stored value
);

	reg ram_bit_storage;

	
	always @(*) begin
		case (in_reg)
		5'd0 : ram_bit_storage <=  in_data[0];
		5'd1 : ram_bit_storage <=  in_data[1];
		5'd2 : ram_bit_storage <=  in_data[2];
		5'd3 : ram_bit_storage <=  in_data[3];
		5'd4 : ram_bit_storage <=  in_data[4];
		5'd5 : ram_bit_storage <=  in_data[5];
		5'd6 : ram_bit_storage <=  in_data[6];
		5'd7 : ram_bit_storage <=  in_data[7];
		5'd8 : ram_bit_storage <=  in_data[8];
		5'd9 : ram_bit_storage <=  in_data[9];
		5'd10 : ram_bit_storage <=  in_data[10];
		5'd11 : ram_bit_storage <=  in_data[11];
		5'd12 : ram_bit_storage <=  in_data[12];
		5'd13 : ram_bit_storage <=  in_data[13];
		5'd14 : ram_bit_storage <=  in_data[14];
		5'd15 : ram_bit_storage <=  in_data[15];
		5'd16 : ram_bit_storage <=  in_data[16];
		5'd17 : ram_bit_storage <=  in_data[17];
		5'd18 : ram_bit_storage <=  in_data[18];
		5'd19 : ram_bit_storage <=  in_data[19];
		5'd20 : ram_bit_storage <=  in_data[20];
		5'd21 : ram_bit_storage <=  in_data[21];
		5'd22 : ram_bit_storage <=  in_data[22];
		5'd23 : ram_bit_storage <=  in_data[23];
		5'd24 : ram_bit_storage <=  in_data[24];
		5'd25 : ram_bit_storage <=  in_data[25];
		5'd26 : ram_bit_storage <=  in_data[26];
		5'd27 : ram_bit_storage <=  in_data[27];
		5'd28 : ram_bit_storage <=  in_data[28];
		5'd29 : ram_bit_storage <=  in_data[29];
		5'd30 : ram_bit_storage <=  in_data[30];
		5'd31 : ram_bit_storage <=  in_data[31];
		endcase
	end

	assign ram_bit = ram_bit_storage;
	
endmodule