module performance_counter  (
	input wire clk, rstn,
	input wire start, stop,
	input wire [31:0] counter_in_reference,
	output reg [31:0] counter,
	output reg [31:0] counter_value_reference
);
	reg active, stopped;
	
	always @(posedge clk ) begin
		if(!rstn   ) begin
			active <= 1'b0;
            stopped <= 1'b0;
		end 
		else begin
            if (!stopped) begin
                if((stop && active)) begin
                      stopped <= 1'b1;
                end
                else if(!active & start ) begin
                    active <= 1'b1;
                    counter <= 1;
                    counter_value_reference <= counter_in_reference;
                end
                else if (active ) begin
                    counter <= counter + 1;
                end
            end
		end
	end

endmodule