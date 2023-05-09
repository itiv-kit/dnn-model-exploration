module Memory_TrueDualPortSingleClk #(
	DATA_WIDTH = 32,
	BUFFER_SIZE = 8,
	ADDR_SIZE = $clog2(BUFFER_SIZE)
)(A1, WEN1, D1, CLK, Q1, A2, WEN2, D2, Q2);
	input [ADDR_SIZE-1:0] A1;
	input WEN1;
	input [DATA_WIDTH-1:0] D1;
	input CLK;
	output wire [DATA_WIDTH-1:0] Q1;
	input [ADDR_SIZE-1:0] A2;
	input WEN2;
	input [DATA_WIDTH-1:0] D2;
	output wire [DATA_WIDTH-1:0] Q2;
	reg [DATA_WIDTH-1:0] data [BUFFER_SIZE-1:0];

	assign Q1 = data[A1];
	always @(posedge CLK) // 1st port on CLK
		if (WEN1) // With Write Enable
			data[A1] <= D1;

	assign Q2 = data[A2];
	always @(posedge CLK) // 2nd port on CLK
		if (WEN2)
			data[A2] <= D2; // With Write Enable
	
endmodule