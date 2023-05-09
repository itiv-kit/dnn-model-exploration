module tb_fifo_buffer #(
    parameter MAXBITWIDTH = 16,
    parameter MAX_GOLDEN_DATA = 10,
    parameter PRINT_GOLDEN_DATA = 0,
    parameter MAX_WAIT = 2
)();
    
    // global signals
    reg tb_clk, tb_rstn;
    string filename = {"./_run/","tb_fifo_buffer",".vcd"}; //TODO: get this one from module name
    reg tb_wait, tb_value_printed;
    reg tb_last_data_word_transmitted;
    reg tb_simulation_run_id;
    reg tb_sent_printed;
    reg [5:0] tb_fsm;
    integer cnt__transmitted_data_words;
    reg tb_DUT__result_rdy_delay;
    
    /* START signals: DUT */
    reg tb_DUT__values_rdy, tb_DUT__result_rdy, tb_DUT__rdy, tb_DUT__next_module_rdy;
    reg [31:0] tb_DUT__value;
    reg [3:0] tb_DUT__bitwidth;
    reg [31:0] tb_DUT__result;
    reg tb_DUT__result__sign;
    reg [7:0] tb_DUT__result__exponent;
    reg [22:0] tb_DUT__result__mantissa;
    reg [31:0] tb_DUT_shifts;
    /* END signals: DUT */

    /* START signals: golden data */
    reg [31:0] golden_results [0:MAX_GOLDEN_DATA-1] ;
    reg [3:0] golden_bitwidth [0:MAX_GOLDEN_DATA-1];
    reg [3:0] golden_shifts [0:MAX_GOLDEN_DATA-1];
    reg [31:0] golden_quantized [0:MAX_GOLDEN_DATA-1];
    reg [7:0] cnt_golden_results, cnt_golden_input;
    integer file_pointer, timestamp_start;
    /* END signals: golden data */

    /* DUT signaling */
    fifo_buffer #(.MAX_BUFFER(3)) DUT (tb_clk, tb_rstn, tb_DUT__values_rdy, tb_DUT__value, tb_DUT__rdy, tb_DUT__next_module_rdy,
            tb_DUT__result_rdy, tb_DUT__result);

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
        file_pointer = $fopen("./../golden_data/fp2int/fp_value.txt", "r");
        cnt_golden_results = 0;
        while($fscanf(file_pointer, "%b", golden_results[cnt_golden_results]) == 1) begin
            cnt_golden_results = cnt_golden_results +1;
        end
        $fclose(file_pointer);
        cnt_golden_results = 0;

        // // bitwidth of quantized values
        // file_pointer = $fopen("./../golden_data/bitwidth.txt", "r");
        // cnt_golden_input = 0;
        // while($fscanf(file_pointer, "%d", golden_bitwidth[cnt_golden_input] ) == 1) begin
        //     cnt_golden_input = cnt_golden_input +1;
        // end
        // $fclose(file_pointer);
        // cnt_golden_input = 0;

        // quantized values in MAXBITWIDTH
        file_pointer = $fopen("./../golden_data/fp2int/fp_value.txt", "r");
        cnt_golden_input = 0;
        while($fscanf(file_pointer, "%b", golden_quantized[cnt_golden_input] ) == 1) begin
            cnt_golden_input = cnt_golden_input +1;
        end
        $fclose(file_pointer);
        cnt_golden_input = 0;

        /* verify golden data */
        // $display("golden data: ");
        // //TODO: = %f - real data type has wrong format (double not single precision!)
        // if (PRINT_GOLDEN_DATA == 1) begin
        //     for (int i= 0; i < MAX_GOLDEN_DATA; i++) begin 
        //     $display("%t, %b and %d => %b, %b, %b ", $time, golden_quantized[i], golden_bitwidth[i], golden_results[i][31], golden_results[i][30:23], golden_results[i][22:0]);
        //     // golden_results[i][31] - sign bit
        //     // golden_results[i][30:23] - exponent
        //     // golden_results[i][22:0] - mantissa
        //     end
        // end
        /* END SECTION: reading golden data */

        /* START SECTION: prepare DUT for data */
        tb_rstn = 1'b0;
        tb_clk = 1'b0;
        tb_DUT__values_rdy = 0;
        tb_DUT__bitwidth = 0;
        tb_DUT__value = 0;
        tb_fsm = 0;

        tb_DUT__next_module_rdy = 0;
        #5 tb_rstn = 1'b0;
        $display("%0t, Reseting system", $time);
        #45 tb_rstn = 1'b1;
        #60 $display("%0t,start testing...", $time);
        /* END SECTION: prepare DUT for data */
        
        /* START SECTION: transmit data to DUT */
        // for (int i = 0; i < MAX_GOLDEN_DATA; i++) 
        // begin
        //     $display("%0t, SENDING VALUE: %h and %d", $time, golden_quantized[i], golden_bitwidth[i]);
        //     // tb_DUT__next_module_rdy = 1'b0;
        //     tb_DUT__bitwidth = golden_bitwidth[i];
        //     tb_DUT__value = golden_quantized[i];
        //     tb_DUT__values_rdy = 1'b1;  
        //     timestamp_start = $time;
        //     #10 tb_DUT__values_rdy = 1'b0;
        //     // tb_DUT__next_module_rdy = 1'b1;

        //     //SET TIME TO MAX WAIT
        //     #40 cnt_golden_input = cnt_golden_input +1;  
        // end
        // /* END SECTION: transmit data to DUT */
        
        // #100 $display("%0t, new test", $time);
        // tb_rstn = 0;
        // cnt_golden_results = 0;
        // #40 tb_rstn = 1;

        // /* START SECTION: transmit data to DUT */
        // for (int i = 0; i < MAX_GOLDEN_DATA; i++) 
        // begin
        //     $display("%0t, SENDING VALUE: %h and %d", $time, golden_quantized[i], golden_bitwidth[i]);
        //     // tb_DUT__next_module_rdy = 1'b0;
        //     tb_DUT__bitwidth = golden_bitwidth[i];
        //     tb_DUT__value = golden_quantized[i];
        //     tb_DUT__values_rdy = 1'b1;  
        //     timestamp_start = $time;
        //     #10 tb_DUT__values_rdy = 1'b0;
        //     // #80 tb_DUT__next_module_rdy = 1'b1;

        //     //SET TIME TO MAX WAIT
        //     #40 cnt_golden_input = cnt_golden_input +1;  
        // end
        /* END SECTION: transmit data to DUT */

        /* emergency shut off of simulation */
        //TODO: add a SIMULATION DONE / EMERGENCY OFF statement
        #900 $display("%0t, SIMULATION DONE", $time);
        $finish();


    end

    // initial begin
    //     #550 tb_DUT__next_module_rdy = 0;
    //     for (int k = 0; k < MAX_GOLDEN_DATA; k++) begin
    //         #30 tb_DUT__next_module_rdy = 1;
    //         #10 tb_DUT__next_module_rdy = 0;
    //     end
    // end
    // always @(posedge tb_DUT__result_rdy) begin
    //     #80 tb_DUT__next_module_rdy = 1;
    //     #20 tb_DUT__next_module_rdy = 0;

    // end

    always @(posedge tb_clk ) begin
        if (tb_rstn) begin
            if(tb_fsm == 0) begin

                tb_fsm = 1;

            end
            else if (tb_fsm == 1) begin
                // load new data and send configuration data for new data set  

               
                cnt__transmitted_data_words = 0;

                
                tb_last_data_word_transmitted = 1'b0;
                #20 tb_fsm = 2;
            end 
            else if (tb_fsm == 2) begin
                // DUT__AXI_LITE_activate = 1'b0;
                // streaming data to AXIS_receiver
                if (!tb_last_data_word_transmitted) begin
                    tb_DUT__bitwidth = golden_bitwidth[cnt__transmitted_data_words];
                    tb_DUT__value = golden_quantized[cnt__transmitted_data_words];
                    tb_DUT__values_rdy = 1'b1;  


                    if (tb_DUT__rdy) begin
                        tb_fsm = 3;
                        tb_DUT__values_rdy = 1'b0;
                    end
                end

                

                // if(cnt__received_data_words == max_data_words) begin
                //     #300 $display("%0t, SIMULATION DONE %d", $time, cnt__received_data_words);
                //     if(tb_simulation_run_id == 0) begin
                //         tb_fsm = 4;
                //         tb_simulation_run_id = 1;
                //         DUT__AXI_LITE_activate = 1'b0;
                //         tb_sent_printed = 0;
                //         $display("%0t, Reseting system", $time);
                //     end
                //     else begin
                //         $finish();
                //     end
                // end
            end
            else if (tb_fsm == 3) begin
                // DUT__AXIS_RCV_valid = 1'b0;
                // DUT__AXIS_RCV_last = 1'b0;
                tb_DUT__values_rdy = 1'b0;
                if (cnt__transmitted_data_words < MAX_GOLDEN_DATA-1) begin
                    cnt__transmitted_data_words = cnt__transmitted_data_words + 1;
                    
                    tb_sent_printed = 0;
                end
                else begin
                    tb_last_data_word_transmitted = 1'b1;
                end
                tb_fsm = 2;
            end
            // else if (tb_fsm < 16) begin
            //     tb_fsm = tb_fsm +1;
            //     DUT__AXIS_RCV_last = 1'b0;
            // end
            // else if (tb_fsm == 16) begin
            //     tb_fsm = 0;
            // end

            if (tb_DUT__result_rdy && !tb_DUT__next_module_rdy) begin //optic test //TODO: need to get done automatically
                assert(golden_results[cnt_golden_results] == tb_DUT__result) 
                begin
                    $display("%0t, CORRECT", $time );
                    $display("\tgolden:  \t%h", golden_results[cnt_golden_results]);
                    $display("\treceived:\t%h", tb_DUT__result);
                    $display("\tperformance: %d\n", $time - timestamp_start);
                end
                else
                begin
                    $display("\n%0t, ERROR", $time );
                    $display("\tgolden:  \t%h", golden_results[cnt_golden_results]);
                    $display("\treceived:\t%h", tb_DUT__result);
                    $display("\tperformance: %d\n", $time - timestamp_start);
                    #200 $error("%0t, ABORT", $time);
                    $stop();
                end

                tb_DUT__next_module_rdy = 1'b1;

                cnt_golden_results = cnt_golden_results +1;
                if(cnt_golden_results == MAX_GOLDEN_DATA) begin
                    $display("%0t, SIMULATION DONE", $time);
                    // $finish();
                    cnt_golden_results = 0;
                end
            end
            else begin
                tb_DUT__next_module_rdy = 0;
            end

        // if(!tb_last_data_word_transmitted && DUT__AXI_LITE_done && tb_fsm == 3) begin
        //     $error("%0t, ERROR: DONE RECEIVED TOO EARLY!!", $time);
        //     #300 $stop();
        // end

        
    
        end
        tb_DUT__result_rdy_delay <= tb_DUT__result_rdy;
    end


    // always @(posedge(tb_DUT__result_rdy)) begin
    //     assert(golden_results[cnt_golden_results] == tb_DUT__result) 
    //     begin
    //         $display("%0t, CORRECT", $time );
    //         $display("\tgolden:  \t%h", golden_results[cnt_golden_results]);
    //         $display("\treceived:\t%h", tb_DUT__result);
    //         $display("\tperformance: %d\n", $time - timestamp_start);
    //     end
    //     else
    //     begin
    //         $display("\n%0t, ERROR", $time );
    //         $display("\tgolden:  \t%h", golden_results[cnt_golden_results]);
    //         $display("\treceived:\t%h", tb_DUT__result);
    //         $display("\tperformance: %d\n", $time - timestamp_start);
    //         #200 $error("%0t, ABORT", $time);
    //         $stop();
    //     end

    //     #20 cnt_golden_results = cnt_golden_results +1;
    //     if(cnt_golden_results == MAX_GOLDEN_DATA) begin
    //          $display("%0t, SIMULATION DONE", $time);
    //         // $finish();
    //         cnt_golden_results =0;
    //     end
    // end


endmodule
