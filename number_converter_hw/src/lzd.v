/*
	leading zero detection: counts leading zeros in 16 bit word
*/

module lzd_4_2 (
	input wire [3:0] in,
	output wire [1:0] out,
	output wire zero
);

	assign out[0] = ~in[3] & ( ~in[2] & ~in[1] || in[2]);
	assign out[1] = ~(in[3] || in[2]);
	assign zero = (in == 0) ? 1'b1 : 1'b0;

endmodule

module lzd (
	input wire [15:0] in,
	output wire zero,
	output wire [3:0] out
);

wire [1:0] res_0, res_1, res_2, res_3, res_4;
wire z0, z1, z2, z3;

lzd_4_2 mod_lzd_0 (in[3:0], res_0, z0);
lzd_4_2 mod_lzd_1 (in[7:4], res_1, z1);
lzd_4_2 mod_lzd_2 (in[11:8], res_2, z2);
lzd_4_2 mod_lzd_3 (in[15:12], res_3, z3);

lzd_4_2 mod_lzd_4 ({~z3,~z2,~z1,~z0},res_4, zero);

assign out[3] = res_4[1];
assign out[2] = res_4[0]; 

assign out[1:0] = res_4[1:0]  == 2'b00 ? res_3 : (res_4[1:0]  == 2'b01 ? res_2 : (res_4[1:0]  == 2'b10 ? res_1 : res_0));

endmodule 

// ------------------------------------------------------------------------
/*
	trailing zero detection: counts trailing zeros in 16 bit word
*/

module tzd_4_2 (
	input wire [0:3] in,
	output wire [1:0] out,
	output wire zero
);

	assign out[0] = ~in[3] & ( ~in[2] & ~in[1] || in[2]);
	assign out[1] = ~(in[3] || in[2]);
	assign zero = (in == 0) ? 1'b1 : 1'b0;

endmodule

module tzd (
	input wire [15:0] in,
	output wire zero,
	output wire [3:0] out
);

wire [1:0] res_0, res_1, res_2, res_3, res_4;
wire z0, z1, z2, z3;

tzd_4_2 mod_lzd_0 (in[3:0], res_0, z0);
tzd_4_2 mod_lzd_1 (in[7:4], res_1, z1);
tzd_4_2 mod_lzd_2 (in[11:8], res_2, z2);
tzd_4_2 mod_lzd_3 (in[15:12], res_3, z3);

tzd_4_2 mod_lzd_4 ({~z3,~z2,~z1,~z0},res_4, zero);

assign out[3] = res_4[1];
assign out[2] = res_4[0]; 

assign out[1:0] = res_4[1:0]  == 2'b00 ? res_0 : (res_4[1:0]  == 2'b01 ? res_1 : (res_4[1:0]  == 2'b10 ? res_2 : res_3));

endmodule 