#include "sims.h"


void readBitsFromFile(char * path, int mem_bytes, u32 *  Buffer ){
	FILE * fp;
	char buffer[40];
	int mem_data;
	fp = fopen(path, "r");
	if (!fp){
		printf("Err: could not open file");
		return;
	} 
	for(int index = 0; index < MAX_BUFFER_SIZE && index < mem_bytes ; index++){
		fread(buffer, memory_width+1,1, fp);
		// printf("\t- %d - %s\n", index,buffer);
		mem_data = 0;
		u32 p = memory_width-1;
		for (int ci = 0; ci < 40 ; ci++){
			if (buffer[ci] == '1' ){
				mem_data = mem_data + (1 << p);
				p--;
			}  
			if (buffer[ci] == '0' ){
				p--;
			}  
			// printf("%d - %u\n", ci, mem_data);
		}  
		// printf("\t- %d -  %#010x, %u\n",index, mem_data, (mem_data));
		Buffer[index] = mem_data; 	
		// break;	
	} 
	fclose(fp);
} 


void readConfigurationFile(char  path[] , axi_lite_config_data arr[]){
	FILE * fp;
	int val;
	char buffer[40];
	number_of_runs = 0;
	fp = fopen(path, "r");
	if(fp){
		for(int run_id = 0; run_id < MAX_CONFIG_RUNS; run_id++){
			if( fscanf(fp,"%d", &val) != EOF){
				number_of_runs += 1;
				arr[run_id].id = val;
			}  
			if( fscanf(fp,"%d", &val) != EOF){
				arr[run_id].bitwidth_d = val;
				arr[run_id].mask_for_bitwidth = (1 << val) - 1;
			}  
			if( fscanf(fp,"%d", &val) != EOF){
				arr[run_id].num_of_fp_values = val;
			}  
			if( fscanf(fp,"%d", &val) != EOF){
				arr[run_id].num_of_mem_bytes = val;
				number_of_mem_bytes += val;
				// printf("%d memory size", number_of_mem_bytes);
			}  
		} 
		fclose(fp);
		// printf("%d runs", number_of_runs);
	} 
	else{
		printf("ERR: cannot open file\n");
	}  
} 