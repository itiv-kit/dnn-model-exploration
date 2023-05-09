#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include "sims_fpga.h"
#include "sims.h"

extern int number_of_runs ;
extern int number_of_mem_bytes ;
int sum_errors;

int memory_width = 16;
int memory_width_mask = 0xFFFF;

int main(int argc, char * argv[]){

	printf("\n ----- entering main ------\n");

	char golden_data_path [100], result_data_path[100], configs_data_path[200] ;
	u32 current_run_id, current_mem_offset_output, current_mem_offset_input;

	//init global vals
	number_of_runs = 0;
	number_of_mem_bytes = 0;

	int reconfig = 0;
	if(strcmp(argv[3], "r" ) == 0){
		reconfig = 1;
	}   

	//set file paths
	strcpy(configs_data_path, "../golden_data/");
	strcpy(golden_data_path, "../golden_data/");
	strcpy(result_data_path, "../golden_data/");

	strcat(configs_data_path, argv[1]);
	strcat(golden_data_path, argv[1]);
	strcat(result_data_path, argv[1]);

	strcat(configs_data_path, "/v");
	strcat(golden_data_path, "/v");
	strcat(result_data_path, "/v");

	strcat(configs_data_path, argv[2]);
	strcat(golden_data_path, argv[2]);
	strcat(result_data_path, argv[2]);


	strcat(configs_data_path, "/configuration_sims.txt");
	strcat(golden_data_path, "/streamed_values.txt");
	strcat(result_data_path, "/streamed_results.txt");

	// printf("DEBUG 0\n");

	// printf("- setting up configuration data\n");


	printf("test_id =  %s - %s\n", argv[1], argv[3] );
	//load data
	printf("- loading data from files \n");
	printf("\t- %s\n", configs_data_path);
	readConfigurationFile(configs_data_path, config_runs);
	printf("\t- %s\n", golden_data_path);
	readBitsFromFile(golden_data_path, number_of_mem_bytes+4*number_of_runs, TxBufferPtr);
	printf("\t- %s\n", result_data_path);
	readBitsFromFile(result_data_path, number_of_mem_bytes, RxBufferPtr);

	printf("num of runs: %d\n" , number_of_runs);

	// printf("\t- bitwidth = %d\n", config_runs[0].bitwidth_d);
	// printf("\t- num_of_fp_values_out = %d\n", config_runs[0].num_of_fp_values);
	// printf("\t- num_of_mem_bytes = %d\n", config_runs[0].num_of_mem_bytes);
	// printf("\t- mask_for_bitwidth = %x\n", config_runs[0].mask_for_bitwidth);

	//run check
	sum_errors = 0;

	for (current_run_id = 0, current_mem_offset_input = 0, current_mem_offset_output = 0; current_run_id < number_of_runs; current_run_id++)
	{
		printf("- checking data results of run %d - (reconf = %d)\n", current_run_id+1, reconfig);
		if (reconfig == 1){
			current_mem_offset_input += 4;
		} 
		CheckData(&config_runs[current_run_id], current_mem_offset_input, current_mem_offset_output);
		current_mem_offset_input += config_runs[current_run_id].num_of_mem_bytes;
		current_mem_offset_output += config_runs[current_run_id].num_of_mem_bytes;
		
	}

	printf("\n -----  SUM OF ERRORS : %d ------- \n", sum_errors);

	

	printf("\n\n ---- exiting main ---- \n\n");
	return 0;
} 