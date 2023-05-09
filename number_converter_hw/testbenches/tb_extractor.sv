`include "/home/ulnor/Dokumente/master/accel_fpga_model/hw/src/definitions.v"

module tb_extractor #(
    parameter MAXBITWIDTH = 16,
    parameter DMA_BITWIDTH = 32,
    parameter MAX_GOLDEN_DATA_INPUT = 2352,
    parameter MAX_GOLDEN_DATA_OUTPUT = 9408,
    parameter PRINT_GOLDEN_DATA = 0,
    parameter MAX_WAIT  = 350,
    parameter ABORT_TIME = 300,
    parameter EMERGENCY_TIME = 9000000,
    parameter RESULT_ALWAYS_READY = 1,
    parameter MAX_RUNS = 100,
    parameter CONFIG_LENGTH = 7,
    parameter DO_RESULTS_CHECK = 1,
    parameter data_path = "/home/ulnor/Dokumente/master/accel_fpga_model/hw/testbenches/golden_data/extractor/v1/"
)();
//TODO: get better test data
    
    // ********************************************
    /* START gloabal testbench signals */
    // ********************************************
    reg tb_clk, tb_rstn;
    string filename = {"./_run/","tb_top",".vcd"}; //TODO: get this one from module name
    // string data_path = "../testbenches/golden_data/top/v0/";
    reg tb_wait, tb_value_printed;
    reg tb_last_data_word_transmitted;
    // reg tb_simulation_run_id;
    reg tb_sent_printed;
    reg [5:0] tb_fsm;
    /*
        FSM Description: 
        this testbench controls the model like the processor later on

        initial workspace: one directory that contains:
            - configuration.txt - configuration data with following structure (32-bit data words)
                #0 Run id
                #1 number of data words
                #2 mask for valid bits 
                #3 number for bitwidth
                #4 scale_fp
                #5 inv_scale_fp
                #6 number of single values
            - <run_id>/ - directory containing following text files
                # quantized_values.txt
                # fp_values.txt
                # streamed_values.txt

        
        0 - once -load configuration data


        1 - loop - load new data and send configuration data for new data set  
            - load text files containing new data
            - set AXI LITE registers 
            - set activate bit separately
        
        2 - loop - streaming data 
            - when RCV ready send new data
            - this side is always ready and performs check
            - when last data word is sent, raise internal bit and wait until last data word is received again from model

    
    */
    // ********************************************
    /* END gloabal testbench signals */
    // ********************************************
    
    // ********************************************
    /* START signals: DUT */
    // ********************************************

    /* AXI LITE REGISTERS*/
    reg DUT__AXI_LITE_activate;
    reg [31:0] DUT__AXI_LITE_num_of_input_values; 
    reg [31:0] DUT__AXI_LITE_bitwidth; 
    reg [31:0] DUT__AXI_LITE_scale_fp;
    reg [31:0] DUT__AXI_LITE_inv_scale_fp;
    reg [31:0] DUT__AXI_LITE_num_of_values;
    reg DUT__AXI_LITE_done; 
    reg [31:0] DUT__AXI_LITE_counter_values;
    
    /* AXIS RECEIVER */
    reg DUT__AXIS_RCV_valid, DUT__AXIS_RCV_last;
    wire [31:0] DUT__AXIS_RCV_data;
    reg DUT__AXIS_RCV_ready;
    wire axis_out_exec;

    /* AXIS TRANSMITTER */
    reg DUT__AXIS_TRM_valid, DUT__AXIS_TRM_last;
    reg [31:0] DUT__AXIS_TRM_data;
    reg DUT__AXIS_TRM_ready;

    /* SPECIAL DUT REGISTERS */
    reg tb_DUT__values_rdy, tb_DUT__result_rdy, tb_DUT__rdy, tb_DUT__next_module_rdy;
    reg [MAXBITWIDTH-1:0] tb_DUT__maxbitwidth_valid_bits;
    reg [31:0] tb_DUT__bitwidth;
    reg [31:0] tb_num_of_input_values;
    reg [31:0] tb_DUT__value;
    reg [30:0] tb_DUT_validation_bits;
    reg [MAXBITWIDTH:0] tb_DUT__result; // additional bit for sign
    reg tb_DUT__result__sign, tb_DUT__last_value;
    reg [7:0] tb_DUT__result__exponent;
    reg [22:0] tb_DUT__result__mantissa;
    
    // ********************************************
    /* END signals: DUT */
    // ********************************************

    /* START signals: golden data */
    reg [31:0] cnt_golden_input;
    integer file_pointer, i, k,s, max_values ;
    integer timestamp_start [0:MAX_GOLDEN_DATA_INPUT-1] ;
    /* END signals: golden data */

    // ********************************************
    /* START signals: golden data */
    // ********************************************
    //files
    integer file_in, file_out;
    // configuration
    reg [31:0] golden_configuration [0:(MAX_RUNS*CONFIG_LENGTH)-1];
    integer cnt__current_config_run, max_config_runs;
    // input data
    reg [31:0] golden_streamed_data [0:MAX_GOLDEN_DATA_INPUT-1] ;
    integer max_data_words_input;
    // output data
    reg [MAXBITWIDTH:0] golden_result_data [0:MAX_GOLDEN_DATA_OUTPUT-1] ;
    integer max_data_words_output, cnt__received_data_words;

    //counter
    integer cnt__transmitted_data_words,cnt__total_transmitted_data_words, cnt__transmitter_last_value;
    integer max_single_values, cnt__receiver_last_value;

    // integer i, k,s, q;
    //integer timestamp_start [0:MAX_GOLDEN_DATA_INPUT-1] ;
    // reg tb_result_check;
    // reg [7:0] result_value_to_check, golden_result_value_to_check;
    // ********************************************
    /* END signals: golden data */
    // ********************************************

    extractor_v0 #(.MAX_BITWIDTH_QUANTIZED_DATA(MAXBITWIDTH), .MEMORY_BITWIDTH(32)) DUT (tb_clk, tb_rstn,
                // input
                tb_DUT__values_rdy,  
                tb_DUT__maxbitwidth_valid_bits,
                tb_DUT__bitwidth, 
                tb_num_of_input_values,
                tb_DUT__value,
                tb_DUT__rdy, 
                // output
                tb_DUT__next_module_rdy,
                tb_DUT__result_rdy,
                tb_DUT__result,
                // DBGs
                ,,,,,,,,,,,,,,, );

    /* CLK generation */
    always begin
        #5 tb_clk = ~tb_clk;
    end

   // does initialization of testbench and prepare internal fsm
    initial begin
        /* VCD storage */
        $display("%m");
        $dumpfile(filename); 
        $dumpvars(0);
        $display("%0t,  ##### START TESTBENCH #####", $time);
        $display("%0t,  \t- Version: %s", $time, data_path);
        // $display("%0t,  ",$time);
        $display("%0t,  \t- Flow: Initialization - Reading Files - Execution of seperated Runs - Done",$time);
        $display("%0t,  ",$time);
        $display("%0t,  ##### INITIALIZATION #####",$time);
        file_out = $fopen({data_path, "streamed_results.txt"}, "w");
        $fclose(file_out);
        tb_fsm = 0;
        tb_last_data_word_transmitted = 0;
        cnt__transmitter_last_value = 0;
        cnt__receiver_last_value = 0;
        tb_sent_printed = 0;
        cnt__total_transmitted_data_words = 0;

        // #### INIT INPUTS: START ####

        // tb_DUT__values_rdy = 1'b0;
        // tb_DUT__maxbitwidth_valid_bits = 0;
        // tb_DUT__bitwidth = 0;
        // tb_num_of_input_values = 0;
        // tb_DUT__value = 0;
        // tb_DUT__rdy = 0;
        // tb_DUT__next_module_rdy = 0;
        // tb_DUT__result_rdy = 0;
        // tb_DUT__result = 0;

        // #### INIT INPUTS: END ####

        tb_rstn = 1'b0;
        tb_clk = 1'b0;
        #5 tb_rstn = 1'b0;
        #45 tb_rstn = 1'b1;
        $display("%0t, \t- design resetted", $time);

        #EMERGENCY_TIME 
        $error("%0t, STOP: EMERGENCY TIME", $time);
        $stop();
    end

    always @(posedge tb_clk) 
    begin /* always_block: INPUT_HANLDER */
        if (tb_rstn) 
        begin /* tb_rstn */
            if(tb_fsm == 0) begin /* tb_fsm = 0 */
                $display("%0t,  ##### READING FILES #####",$time);
                $display("%0t, \t- configuration file", $time);
                // load configuration data
                file_in = $fopen({data_path, "configuration.txt"}, "r");
                max_config_runs = 0;
                while($fscanf(file_in, "%b", golden_configuration[max_config_runs] ) == 1) begin
                    max_config_runs = max_config_runs +1;
                end
                $fclose(file_in);
                max_config_runs = max_config_runs / CONFIG_LENGTH;
                $display("%0t, \t\t- number of runs : %d", $time, max_config_runs);
                

                // data words to stream
                $display("%0t, \t- memory data file", $time);
                file_in = $fopen({data_path, "streamed_values.txt"}, "r");
                max_data_words_input = 0;
                while($fscanf(file_in, "%b", golden_streamed_data[max_data_words_input] ) == 1) begin
                    max_data_words_input = max_data_words_input +1;
                end
                $fclose(file_in);
                $display("%0t, \t\t- number of input memory words : %d", $time, max_data_words_input);

                // data words to check
                $display("%0t, \t- result data file", $time);
                file_in = $fopen({data_path, "results.txt"}, "r");
                max_data_words_output = 0;
                while($fscanf(file_in, "%b", golden_result_data[max_data_words_output] ) == 1) begin
                    max_data_words_output = max_data_words_output +1;
                end
                $fclose(file_in);
                $display("%0t, \t\t- number of output results : %d", $time, max_data_words_output);

                cnt__current_config_run = 0;
                cnt__received_data_words = 0;
                tb_fsm = 1;

            end /* tb_fsm = 0 */
            else if (tb_fsm == 1) begin /* tb_fsm = 1 */
                $display("%0t,  ##### EXECUTION RUN ID %d of %d #####",$time, cnt__current_config_run+1, max_config_runs);

                /* send configuration data for new data set */
                // cnt__current_config_run = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+0][31:0] -1;
                max_data_words_input = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+1][31:0];
                DUT__AXI_LITE_num_of_input_values = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+2][31:0] ;
                DUT__AXI_LITE_bitwidth = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+3][31:0];
                DUT__AXI_LITE_scale_fp = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+4][31:0];
                DUT__AXI_LITE_inv_scale_fp = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+5][31:0];
                DUT__AXI_LITE_num_of_values = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+6][31:0];
                $display("%0t, \t- mem words: %d", $time, max_data_words_input);
                $display("%0t, \t- num of values to send: %d", $time, DUT__AXI_LITE_num_of_input_values);
                $display("%0t, \t- bitwidth: %d", $time, DUT__AXI_LITE_bitwidth);
                $display("%0t, \t- bias: %h", $time, DUT__AXI_LITE_scale_fp);
                $display("%0t, \t- inverted bias: %h", $time, DUT__AXI_LITE_inv_scale_fp);
                $display("%0t, \t- num of values to received: %d", $time, DUT__AXI_LITE_num_of_values);

                /* activate DUT */
                #20 DUT__AXI_LITE_activate = 1'b1;

                /* prepare intern signals */
                cnt__transmitted_data_words = 0;
                // cnt__received_data_words = 0;
                tb_last_data_word_transmitted = 1'b0;

                /* run FSM = 2 */
                #20 tb_fsm = 2;

            end  /* tb_fsm = 1 */
            else if (tb_fsm == 2) 
            begin /* tb_fsm = 2 */

                /* streaming data */
                // IF NECESSARY REPLACE: DUT__AXIS_RCV_valid 
                // IF NECESSARY REPLACE: DUT__AXIS_RCV_last 
                if (!tb_last_data_word_transmitted) begin
                    DUT__AXIS_RCV_valid = 1'b1;
                    if(cnt__transmitted_data_words == max_data_words_input-2) begin
                        DUT__AXIS_RCV_last = 1'b1;
                    end

                    if(!tb_sent_printed) begin    
                        $display("%0t, + SENT: %d, RECEIVED: %d, TOTAL: %d",$time, cnt__transmitted_data_words+1, cnt__received_data_words, max_data_words_input);

                        tb_sent_printed = 1;
                    end

                    if (axis_out_exec) begin
                        // tb_fsm = 3;
                        cnt__total_transmitted_data_words = cnt__total_transmitted_data_words + 1;
                        if (cnt__transmitted_data_words < max_data_words_input-1) begin
                            cnt__transmitted_data_words = cnt__transmitted_data_words + 1;
                            tb_sent_printed = 0;
                        end
                        else begin
                            tb_last_data_word_transmitted = 1'b1;
                            DUT__AXIS_RCV_valid = 1'b0;
                        end
                    end
                end

                DUT__AXIS_TRM_ready = 1'b1;
                

                if(cnt__received_data_words == DUT__AXI_LITE_num_of_values) begin
                    #300 $display("%0t, +++ RUN %d DONE +++", $time, cnt__current_config_run);
                    if(!DUT__AXIS_TRM_last) begin
                        $error("%0t, ERROR: last bit not received", $time);
                        #300 $stop();
                    end
                    if(cnt__current_config_run < max_config_runs-1) begin
                        tb_fsm = 4;
                        cnt__current_config_run = cnt__current_config_run + 1;
                        DUT__AXI_LITE_activate = 1'b0;
                        tb_sent_printed = 0;
                        $display("%0t, - reseting system", $time);
                    end
                    else begin
                        $display("%0t, #### TESTBENCH DONE ####", $time);
                        $finish();
                    end
                end
            end /* tb_fsm = 2 */
            else if (tb_fsm < 32) begin
                tb_fsm = tb_fsm +1;
                DUT__AXIS_RCV_last = 1'b0;
            end
            else begin
                tb_fsm = 1;
            end
        end  /* tb_rstn */

        if(!tb_last_data_word_transmitted && DUT__AXIS_TRM_last && tb_fsm == 2) 
        begin /* CHECK: LAST_BIT */
            $error("%0t, ERROR: last bit received too early", $time);
            #300 $stop();
        end  /* CHECK: LAST_BIT */

end /* always_block: INPUT_HANLDER */



always @(posedge DUT__AXIS_TRM_valid) 
begin /* always_block: OUTPUT_HANLDER */
    if (DO_RESULTS_CHECK == 1) begin /* DO_RESULTS_CHECK */
        assert(golden_result_data[cnt__received_data_words] == tb_DUT__result) 
        begin
            k = cnt__received_data_words/((MAX_GOLDEN_DATA_OUTPUT/MAX_GOLDEN_DATA_INPUT));
            $display("%0t, CORRECT %d", $time, cnt__received_data_words);
            // $display("\tgolden: \t%b", golden_result_data[cnt__received_data_words]);
            // $display("\treceived:\t%b", tb_DUT__result);
            // $display("\tperformance: %d\n", $time - timestamp_start);
        end
        else
        begin
            $display("\n%0t, ERROR", $time );
            $display("\tgolden: \t%b", golden_result_data[cnt__received_data_words]);
            $display("\treceived:\t%b", tb_DUT__result);
            // $display("\tperformance: %d\n", $time - timestamp_start);
            #40 $error("%0t, ABORT", $time);
            $stop();
        end
        cnt__received_data_words = cnt__received_data_words +1;
    end /* DO_RESULTS_CHECK */
    else begin /* !DO_RESULTS_CHECK */
        cnt__received_data_words = cnt__received_data_words +1;
        $display("%0t, + SENT: %d, RECEIVED: %d, TOTAL: %d",$time, cnt__transmitted_data_words+1, cnt__received_data_words, max_data_words_input);
        file_out = $fopen({data_path, "streamed_results.txt"}, "a");
        $fwriteb(file_out,DUT__AXIS_TRM_data );
        $fwrite(file_out, "\n");
        $fclose(file_out);
    end /* !DO_RESULTS_CHECK */
end /* always_block: OUTPUT_HANLDER */


assign axis_out_exec = DUT__AXIS_RCV_ready & DUT__AXIS_RCV_valid;
assign DUT__AXIS_RCV_data = golden_streamed_data[cnt__total_transmitted_data_words];



// ********************************************
/* START set assigns for DUT */
// ********************************************

// constant data
assign tb_DUT__maxbitwidth_valid_bits = (2**(DUT__AXI_LITE_bitwidth))-1;
assign tb_DUT__bitwidth = DUT__AXI_LITE_bitwidth;
assign tb_num_of_input_values = DUT__AXI_LITE_num_of_values;
// input handler
assign tb_DUT__values_rdy = DUT__AXIS_RCV_valid;
assign DUT__AXIS_RCV_ready = tb_DUT__rdy;
assign tb_DUT__value = DUT__AXIS_RCV_data;
// output handler
assign tb_DUT__next_module_rdy = DUT__AXIS_TRM_ready;
assign DUT__AXIS_TRM_valid = tb_DUT__result_rdy;
assign DUT__AXIS_TRM_data = tb_DUT__result;

// ********************************************
/* END set assigns for DUT */
// ********************************************
endmodule
