`include "../src/definitions.v"

module tb_top #(
    parameter MAX_IMPLEMENTATIONS = 16,
    parameter MAXBITWIDTH = 16,
    parameter DMA_BITWIDTH = 16*MAX_IMPLEMENTATIONS,
    parameter MAX_GOLDEN_DATA_INPUT = 10000,
    parameter MAX_GOLDEN_DATA_RESULT = 100000,
    parameter PRINT_GOLDEN_DATA = 0,
    parameter MAX_WAIT  = 350,
    parameter ABORT_TIME = 300,
    parameter EMERGENCY_TIME = 9000000,
    parameter data_path = "/home/ulnor/Dokumente/paper_ARC/accel_fpga_model/hw/testbenches/golden_data/top/v15/",
    parameter RESULT_ALWAYS_READY = 1,
    parameter MAX_RUNS = 100,
    parameter CONFIG_LENGTH = 7

)();
    
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
    reg [15:0] DUT__AXI_LITE_scale_fp;
    reg [15:0] DUT__AXI_LITE_inv_scale_fp;
    reg [31:0] DUT__AXI_LITE_num_of_values;
    reg DUT__AXI_LITE_done; 
    reg [31:0] DUT__AXI_LITE_counter_values;

    // Performance counter
    reg [31:0] DUT__AXI_LITE_counter__axilite_setup;  
    reg [31:0] DUT__AXI_LITE_counter__dequantization_flow; 
    reg [31:0] DUT__AXI_LITE_counter__quantization_flow;   
    reg [31:0] DUT__AXI_LITE_counter__start_of_quantization_flow; 
    reg [31:0] DUT__AXI_LITE_counter__extractor;  
    reg [31:0] DUT__AXI_LITE_counter__start_of_extractor; 
    reg [31:0] DUT__AXI_LITE_counter__dequantization;  
    reg [31:0] DUT__AXI_LITE_counter__start_of_dequantization;     
    reg [31:0] DUT__AXI_LITE_counter__quantization;  
    reg [31:0] DUT__AXI_LITE_counter__compression; 
    reg [31:0] DUT__AXI_LITE_counter__start_of_compression;  
    

    /* AXIS RECEIVER */
    reg DUT__AXIS_RCV_valid, DUT__AXIS_RCV_last;
    wire [DMA_BITWIDTH-1:0] DUT__AXIS_RCV_data;
    reg DUT__AXIS_RCV_ready;
    wire axis_out_exec;

    /* AXIS TRANSMITTER */
    reg DUT__AXIS_TRM_valid, DUT__AXIS_TRM_last;
    reg [DMA_BITWIDTH-1:0] DUT__AXIS_TRM_data;
    reg DUT__AXIS_TRM_ready;
    wire axis_in_exec;

    reg [31:0] DUT__scenes;
    reg DUT__rstn_cnt_layer, DUT__compressor_last_value, DUT__wrapper_master_last_value, DUT__rstn_internal_out, DUT__compressor_done;

    // ********************************************
    /* END signals: DUT */
    // ********************************************

    // ********************************************
    /* START signals: golden data */
    // ********************************************
    integer file_in, file_out;
    reg [31:0] golden_configuration [0:(MAX_RUNS*CONFIG_LENGTH)-1];
    integer cnt__current_config_run, max_config_runs;
    reg [DMA_BITWIDTH-1:0] golden_streamed_data [0:MAX_GOLDEN_DATA_INPUT-1] ;
    integer cnt__transmitted_data_words,cnt__total_transmitted_data_words, max_data_words, cnt__transmitter_last_value;
    reg [DMA_BITWIDTH-1:0] golden_quantized_values [0:MAX_GOLDEN_DATA_RESULT-1];
    integer cnt__received_data_words, max_single_values, cnt__receiver_last_value;

    // integer i, k,s, q;
    integer timestamp_start [0:MAX_GOLDEN_DATA_INPUT-1] ;
    // reg tb_result_check;
    // reg [7:0] result_value_to_check, golden_result_value_to_check;
    // ********************************************
    /* END signals: golden data */
    // ********************************************

    top #(.MAX_BITWIDTH_QUANTIZED_DATA(MAXBITWIDTH), .INPUT_BITWIDTH(DMA_BITWIDTH), .MAX_IMPLEMENTATIONS(MAX_IMPLEMENTATIONS)) DUT (
                tb_clk, tb_rstn, 
                //control data
                DUT__AXI_LITE_activate, 
                DUT__AXI_LITE_num_of_input_values, 
                DUT__AXI_LITE_bitwidth, 
                DUT__AXI_LITE_scale_fp, 
                DUT__AXI_LITE_inv_scale_fp, 
                DUT__AXI_LITE_num_of_values,

                // Performance counter
                DUT__AXI_LITE_counter__axilite_setup,  
                DUT__AXI_LITE_counter__dequantization_flow,  
                DUT__AXI_LITE_counter__quantization_flow,   
                DUT__AXI_LITE_counter__start_of_quantization_flow,  
                DUT__AXI_LITE_counter__extractor,      
                DUT__AXI_LITE_counter__dequantization,      
                DUT__AXI_LITE_counter__start_of_dequantization,   
                DUT__AXI_LITE_counter__quantization,  
                DUT__AXI_LITE_counter__compression,  
                DUT__AXI_LITE_counter__start_of_compression,     

                //input stream data
                DUT__AXIS_RCV_valid, DUT__AXIS_RCV_last, DUT__AXIS_RCV_data, DUT__AXIS_RCV_ready,
                //output stream data
                DUT__AXIS_TRM_valid, DUT__AXIS_TRM_last, DUT__AXIS_TRM_data, DUT__AXIS_TRM_ready);

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
        DUT__AXI_LITE_activate = 1'b1;
        DUT__AXI_LITE_num_of_input_values = 0;
        DUT__AXI_LITE_bitwidth = 0;
        DUT__AXI_LITE_scale_fp = 0;
        DUT__AXI_LITE_inv_scale_fp = 0;
        DUT__AXI_LITE_num_of_values = 0;
        DUT__AXIS_RCV_valid = 0;
        DUT__AXIS_RCV_last = 0;
        // DUT__AXIS_RCV_data = 0;
        DUT__AXIS_TRM_ready = 0;
        DUT__rstn_cnt_layer = 0;
        tb_last_data_word_transmitted = 0;
        cnt__transmitter_last_value = 0;
        cnt__receiver_last_value = 0;
        // tb_simulation_run_id = 0;
        tb_sent_printed = 0;
        cnt__total_transmitted_data_words = 0;

        tb_rstn = 1'b0;
        tb_clk = 1'b0;
        #5 tb_rstn = 1'b0;
        #45 tb_rstn = 1'b1;
        #50 DUT__AXI_LITE_activate = 1'b0;
        $display("%0t, \t- design resetted", $time);

        #EMERGENCY_TIME 
        $error("%0t, STOP: EMERGENCY TIME", $time);
        $stop();
    end


    always @(posedge tb_clk) begin
        if (tb_rstn) begin
        if(tb_fsm == 0) begin
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
            max_data_words = 0;
            while($fscanf(file_in, "%b", golden_streamed_data[max_data_words] ) == 1) begin
                max_data_words = max_data_words +1;
            end
            $fclose(file_in);
            $display("%0t, \t\t- number of memory words : %d", $time, max_data_words);

            cnt__current_config_run = 0;
            tb_fsm = 1;

        end
        else if (tb_fsm == 1) begin
            $display("%0t,  ##### EXECUTION RUN ID %d of %d #####",$time, cnt__current_config_run+1, max_config_runs);
            // send configuration data for new data set 
             
            // cnt__current_config_run = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+0][31:0] -1;
            `ifndef _HW_AXILITE_
            max_data_words = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+1][31:0] + 4;
            `else
            max_data_words = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+1][31:0];
            `endif
            DUT__AXI_LITE_num_of_input_values = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+2][31:0] ;
            DUT__AXI_LITE_bitwidth =  golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+3][31:0];
            DUT__AXI_LITE_scale_fp = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+4][31:0];
            DUT__AXI_LITE_inv_scale_fp = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+5][31:0];
            DUT__AXI_LITE_num_of_values = golden_configuration[(cnt__current_config_run*CONFIG_LENGTH)+6][31:0] ;
            $display("%0t, \t- mem words: %d", $time, max_data_words);
            $display("%0t, \t- num of values to send: %d", $time, DUT__AXI_LITE_num_of_input_values);
            $display("%0t, \t- bitwidth: %d", $time, DUT__AXI_LITE_bitwidth);
            $display("%0t, \t- bias: %h", $time, DUT__AXI_LITE_scale_fp);
            $display("%0t, \t- inverted bias: %h", $time, DUT__AXI_LITE_inv_scale_fp);
            $display("%0t, \t- num of values to received: %d", $time, DUT__AXI_LITE_num_of_values);
            #50 DUT__AXI_LITE_activate = 1'b1;

            cnt__transmitted_data_words = 0;
            cnt__received_data_words = 0;
            tb_last_data_word_transmitted = 1'b0;
            #20 tb_fsm = 2;
        end 
        else if (tb_fsm == 2) begin
            // DUT__AXI_LITE_activate = 1'b0;
            // streaming data to AXIS_receiver
            if (!tb_last_data_word_transmitted) begin
                DUT__AXIS_RCV_valid = 1'b1;
                if(cnt__transmitted_data_words == max_data_words-1) begin
                    DUT__AXIS_RCV_last = 1'b1;
                end

                if(!tb_sent_printed) begin    
                   // $display("%0t, + SENT: %d, RECEIVED: %d, TOTAL: %d",$time, cnt__transmitted_data_words+1, cnt__received_data_words, max_data_words);

                    tb_sent_printed = 1;
                end

                if (axis_out_exec) begin
                    // tb_fsm = 3;
                    cnt__total_transmitted_data_words = cnt__total_transmitted_data_words + 1;
                    if (cnt__transmitted_data_words < max_data_words-1) begin
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
            
`ifndef _HW_AXILITE_
            if(cnt__received_data_words == max_data_words-4) begin
`else 
            if(cnt__received_data_words == max_data_words) begin
`endif
                #400 $display("%0t, +++ RUN %d DONE +++", $time, cnt__current_config_run);
                    $display("%0t, COUNTER AXI LITE: %d", $time, DUT__AXI_LITE_counter__axilite_setup);
                    $display("%0t, COUNTER DEQUANTIZATION FLOW: %d", $time, DUT__AXI_LITE_counter__dequantization_flow);
                    $display("%0t, COUNTER EXTRACTION : %d", $time, DUT__AXI_LITE_counter__extractor);
                    $display("%0t, COUNTER DEQUANTIZATION : %d", $time, DUT__AXI_LITE_counter__dequantization);
                    $display("%0t, COUNTER DEQUANTIZATION START: %d", $time, DUT__AXI_LITE_counter__start_of_dequantization);
                    $display("%0t, COUNTER QUANTIZATION FLOW: %d", $time, DUT__AXI_LITE_counter__quantization_flow);
                    $display("%0t, COUNTER QUANTIZATION FLOW START: %d", $time, DUT__AXI_LITE_counter__start_of_quantization_flow);
                    $display("%0t, COUNTER QUANTIZATION : %d", $time, DUT__AXI_LITE_counter__quantization);
                    $display("%0t, COUNTER COMPRESSION : %d", $time, DUT__AXI_LITE_counter__compression);
                    $display("%0t, COUNTER COMPRESSION START: %d", $time, DUT__AXI_LITE_counter__start_of_compression);
                if(!DUT__AXIS_TRM_last) begin
                    $error("%0t, ERROR: last bit not received", $time);
                    // #300 $stop();
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
        end
        else if (tb_fsm < 32) begin
            tb_fsm = tb_fsm +1;
            DUT__AXIS_RCV_last = 1'b0;
        end
        else begin
            tb_fsm = 1;
        end
    end

    if(!tb_last_data_word_transmitted && DUT__AXIS_TRM_last && tb_fsm == 2) begin
        $error("%0t, ERROR: last bit received too early", $time);
        #300 $stop();
    end


end

always @(posedge tb_clk) begin
    if (axis_in_exec ) begin
        cnt__received_data_words = cnt__received_data_words +1;
        //$display("%0t, + SENT: %d, RECEIVED: %d, TOTAL: %d",$time, cnt__transmitted_data_words+1, cnt__received_data_words, max_data_words);
        file_out = $fopen({data_path, "streamed_results.txt"}, "a");
        $fwriteb(file_out,DUT__AXIS_TRM_data );
        $fwrite(file_out, "\n");
        $fclose(file_out);
        // if (DUT__AXIS_TRM_last) begin
        //     $fclose(file_out);
     end
end

assign axis_out_exec = DUT__AXIS_RCV_ready & DUT__AXIS_RCV_valid;

assign axis_in_exec = DUT__AXIS_TRM_ready & DUT__AXIS_TRM_valid;

assign DUT__AXIS_RCV_data = golden_streamed_data[cnt__total_transmitted_data_words];


endmodule