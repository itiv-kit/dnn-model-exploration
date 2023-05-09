module tb_dequantization #(
    parameter MAXBITWIDTH = 16,
    parameter MAX_GOLDEN_DATA = 16,
    parameter PRINT_GOLDEN_DATA = 0,
    parameter MAX_WAIT = 600,
    parameter ABORT_TIME = 200
)();
    
    // global signals
    reg tb_clk, tb_rstn;
    string filename = {"./_run/","tb_dequantization",".vcd"}; //TODO: get this one from module name
    
    /* START signals: DUT */
    reg tb_DUT__values_rdy, tb_DUT__result_rdy, tb_DUT__rdy, tb_DUT__next_module_rdy;
    reg [MAXBITWIDTH-1:0] tb_DUT__value;
    reg [3:0] tb_DUT__bitwidth;
    reg [31:0] tb_DUT__result, tb_DUT_scale_fp;
    reg tb_DUT__result__sign;
    reg [7:0] tb_DUT__result__exponent;
    reg [22:0] tb_DUT__result__mantissa;
    reg [31:0] tb_DUT_shifts;
    /* END signals: DUT */

    /* START signals: golden data */
    reg [31:0] golden_results [0:MAX_GOLDEN_DATA-1] ;
    reg [3:0] golden_bitwidth [0:MAX_GOLDEN_DATA-1];
    reg [31:0] golden_scale_fp [0:MAX_GOLDEN_DATA-1];
    reg [MAXBITWIDTH-1:0] golden_quantized [0:MAX_GOLDEN_DATA-1];
    reg [7:0] cnt_golden_results, cnt_golden_input;
    integer file_pointer, timestamp_start [0:MAX_GOLDEN_DATA], k, i;
    /* END signals: golden data */

    /* DUT signaling */
    dequantization #(.MAXBITWIDTH(MAXBITWIDTH)) DUT (tb_clk, tb_rstn, tb_DUT__values_rdy, tb_DUT__bitwidth,
        tb_DUT__value, tb_DUT_scale_fp,
        tb_DUT__rdy, tb_DUT__next_module_rdy, tb_DUT__result_rdy, tb_DUT__result);

    /* CLK generation */
    always begin
        #5 tb_clk = ~tb_clk;
    end

    /* READ golden data and transmit to DUT */
    initial begin
        /* VCD storage */
        $display("%m");
        $dumpfile(filename); 
        $dumpvars(0);

        /* START SECTION: reading golden data */
        // results - fp32 values
        file_pointer = $fopen("./../golden_data/dequantization/results.txt", "r");
        cnt_golden_results = 0;
        while($fscanf(file_pointer, "%b", golden_results[cnt_golden_results]) == 1) begin
            cnt_golden_results = cnt_golden_results +1;
        end
        $fclose(file_pointer);
        cnt_golden_results = 0;

        // bitwidth of quantized values
        file_pointer = $fopen("./../golden_data/dequantization/bitwidth.txt", "r");
        cnt_golden_input = 0;
        while($fscanf(file_pointer, "%d", golden_bitwidth[cnt_golden_input] ) == 1) begin
            cnt_golden_input = cnt_golden_input +1;
        end
        $fclose(file_pointer);
        cnt_golden_input = 0;

        // quantized values in MAXBITWIDTH
        file_pointer = $fopen("./../golden_data/dequantization/quantized_values.txt", "r");
        cnt_golden_input = 0;
        while($fscanf(file_pointer, "%b", golden_quantized[cnt_golden_input] ) == 1) begin
            cnt_golden_input = cnt_golden_input +1;
        end
        $fclose(file_pointer);
        cnt_golden_input = 0;


        // scale_fp
        file_pointer = $fopen("./../golden_data/dequantization/scale_fp.txt", "r");
        cnt_golden_input = 0;
        while($fscanf(file_pointer, "%b", golden_scale_fp[cnt_golden_input] ) == 1) begin
            cnt_golden_input = cnt_golden_input +1;
        end
        $fclose(file_pointer);
        cnt_golden_input = 0;

        /* verify golden data */
        $display("golden data: ");
        //TODO: = %f - real data type has wrong format (double not single precision!)
        if (PRINT_GOLDEN_DATA == 1) begin
            for (int i= 0; i < MAX_GOLDEN_DATA; i++) begin 
            $display("( %b and %d ) - ( %b | %b | %b ) => %b, %b, %b ", 
            golden_quantized[i], golden_bitwidth[i], 
            golden_scale_fp[i][31], golden_scale_fp[i][30:23], golden_scale_fp[i][22:0],
            golden_results[i][31], golden_results[i][30:23], golden_results[i][22:0]);
            // golden_results[i][31] - sign bit
            // golden_results[i][30:23] - exponent
            // golden_results[i][22:0] - mantissa
            end
        end
        /* END SECTION: reading golden data */

        /* START SECTION: prepare DUT for data */
        tb_rstn = 1'b0;
        tb_clk = 1'b0;
        tb_DUT__values_rdy = 0;
        tb_DUT__bitwidth = 0;
        tb_DUT__value = 0;
        k = 0;
        i = 0;
        tb_DUT__next_module_rdy = 1'b1;
        #5 tb_rstn = 1'b0;
        $display("%0t, Reseting system", $time);
        #45 tb_rstn = 1'b1;
        $display("%0t,start testing...", $time);
        /* END SECTION: prepare DUT for data */
        
        // /* START SECTION: transmit data to DUT */
        // for (int i = 0; i < MAX_GOLDEN_DATA; i++) 
        // begin
        //     #15 $display("%0t, SENDING VALUE %d: ", $time, i);
        //     $display("( %b and %d ) - ( %b | %b | %b )", golden_quantized[i], golden_bitwidth[i],
        //     golden_scale_fp[i][31], golden_scale_fp[i][30:23], golden_scale_fp[i][22:0]);
        //     tb_DUT__next_module_rdy = 1'b0;
        //     tb_DUT__bitwidth = golden_bitwidth[i];
        //     tb_DUT__value = golden_quantized[i];
        //     tb_DUT_scale_fp = golden_scale_fp[i];
        //     tb_DUT__values_rdy = 1'b1;  
        //     timestamp_start = $time;
        //     #15 tb_DUT__values_rdy = 1'b0;
        //     tb_DUT__next_module_rdy = 1'b1;

        //     //SET TIME TO MAX WAIT
        //     #MAX_WAIT cnt_golden_input = cnt_golden_input +1;  
        // end
        // /* END SECTION: transmit data to DUT */

        /* emergency shut off of simulation */
        #10000 $error("%0t, EMERGENCY SHUT OFF", $time);
        $stop();


    end

    always @(posedge tb_DUT__rdy) begin
        if (i < MAX_GOLDEN_DATA) 
        begin
            $display("%0t, SENDING VALUE %d: ", $time, i);
            $display("( %b and %d ) - ( %b | %b | %b )", golden_quantized[i], golden_bitwidth[i],
            golden_scale_fp[i][31], golden_scale_fp[i][30:23], golden_scale_fp[i][22:0]);
            // tb_DUT__next_module_rdy = 1'b0;
            tb_DUT__bitwidth = golden_bitwidth[i];
            tb_DUT__value = golden_quantized[i];
            tb_DUT_scale_fp = golden_scale_fp[i];
            tb_DUT__values_rdy = 1'b1;  
            timestamp_start[i] = $time;
            #15 tb_DUT__values_rdy = 1'b0;
            // tb_DUT__next_module_rdy = 1'b1;
            i = i + 1;
        end

        //SET TIME TO MAX WAIT
        // #MAX_WAIT cnt_golden_input = cnt_golden_input +1; 
    end


    always @(posedge(tb_DUT__result_rdy)) begin
        assert(golden_results[k] == tb_DUT__result) 
        begin
            $display("%0t, CORRECT %d", $time, k );
            $display("\tgolden: \t%b | %b | %b", golden_results[k][31], golden_results[k][30:23], golden_results[k][22:0]);
            // $display("\treceived:\t%b", tb_DUT__result);
            $display("\tperformance: %d\n", $time - timestamp_start[k]);
        end
        else
        begin
            $display("\n%0t, ERROR %d", $time, k );
            $display("\tgolden: \t%b | %b | %b", golden_results[k][31], golden_results[k][30:23], golden_results[k][22:0]);
            $display("\treceived:\t%b | %b | %b", tb_DUT__result[31], tb_DUT__result[30:23], tb_DUT__result[22:0]);
            $display("\tperformance: %d\n", $time - timestamp_start[k]);
            #ABORT_TIME $error("%0t, ABORT", $time);
            $stop();
        end
        #20 k = k +1;
        if(k == MAX_GOLDEN_DATA) begin
            #ABORT_TIME $display("%0t, SIMULATION DONE", $time);
            $finish();
        end
    end


endmodule
