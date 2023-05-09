#ifndef SIMS_FPGA_H
#define SIMS_FPGA_H

#include <stdio.h>
#include <math.h>

#define XST_FAILURE 0
#define XST_SUCCESS 1
#define MAX_BUFFER_SIZE 10000000
#define MAX_CONFIG_RUNS 100

typedef int u32;

typedef struct  axi_lite_config_data{
	u32  mask_for_bitwidth, bitwidth_d, scale_fp, inv_scale_fp, num_of_fp_values, num_of_mem_bytes;
	u32* memoryBytes;
	u32 id;

} axi_lite_config_data;

typedef struct buffer_data{
	u32 rx, tx;
} buffer_data;

typedef struct single_buffer{
	buffer_data data;
	u32 valid;
	u32 last_mask;
	u32 last_shift;
	u32 bitwidth;
	u32 id;
	u32 init_mask;
	u32 init_bit_shifts;
	u32 last_value_was_partially_extracted;
} single_buffer;


u32 TxBufferPtr[MAX_BUFFER_SIZE] ,  RxBufferPtr[MAX_BUFFER_SIZE];
single_buffer glb_buffer_1, glb_buffer_2;
axi_lite_config_data config_runs[MAX_CONFIG_RUNS]; 
int number_of_runs ;
int number_of_mem_bytes ;

int max_dist;
extern int sum_errors;
extern int memory_width;
extern int memory_width_mask;

u32 updateBufferCheck(single_buffer* bf, u32 mem_rx, u32 mem_tx);

u32 extractNextCompressedValue(single_buffer* bf_active, single_buffer* bf_backup, buffer_data* data);

u32 checkReceivedData(buffer_data* data, u32 mask_highest_bit, u32 mask_complement);

u32 CheckData( axi_lite_config_data *config, u32 mem_offset_in, u32 mem_offset_out);


#endif