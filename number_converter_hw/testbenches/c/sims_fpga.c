#include "sims_fpga.h"

u32 updateBufferCheck(single_buffer* bf, u32 mem_rx, u32 mem_tx){
	if(bf->valid == 0){
		// printf("DBG \tLOAD buffer %d: Rx =  %x | Tx = %x\n",bf->id, mem_rx, mem_tx );
		bf->data.rx = mem_rx;
		bf->data.tx = mem_tx;
		bf->valid = 1;
		bf->last_mask = 0;
		bf->last_shift = 0;
		bf->last_value_was_partially_extracted = 0;
		return 1;
	}
	return 0;
}


u32 extractNextCompressedValue(single_buffer* bf_active, single_buffer* bf_backup, buffer_data* data){

	u32 mask; //mask to get the current active bits from buffer
	u32 bit_shifts; //number of shifts left in active buffer so far
	u32 bit_shifts__active_buffer, bit_shifts__backup_buffer;
	u32 mask__active_buffer, mask__backup_buffer;
	u32 data_active, data_backup;

	// setting mask for extracting data from buffer
	//TODO: possible to merge if and else if ?
	if(bf_active->last_shift == 0 && bf_active->last_mask == 0){ // first time entry
		mask = bf_active->init_mask;
		bit_shifts = bf_active->last_shift;
		// printf("DBG\tEXTRACTION: IF0 = 0");
	} else if (bf_active->last_value_was_partially_extracted == 1) { // last iter: partly accessed this as backup_buffer
		bit_shifts = bf_active->last_shift; 
		mask = (bf_active->init_mask) << bit_shifts; //bitwidth mask shifted by previous taken bits
		// printf("DBG\tEXTRACTION: IF0 = 1");
 	} else { // last iter: simple accessed
		mask = bf_active->last_mask << bf_active->bitwidth;
		bit_shifts = bf_active->last_shift + bf_active->bitwidth;
		// printf("DBG\tEXTRACTION: IF0 = 2");
	}
	// printf(", mask = 0x%08x, bitshifts = %d\n", mask, bit_shifts);


	if(bit_shifts >= memory_width - bf_active->bitwidth){ // complex data extracting (2 buffers)
		// printf("DBG\tEXTRACTION: IF1 = 0\n");
		
		// preparations
		bit_shifts__backup_buffer = - memory_width + bit_shifts + bf_active->bitwidth ;
		bit_shifts__active_buffer = memory_width - bf_active->bitwidth + bit_shifts__backup_buffer;
		mask__active_buffer = ~(memory_width_mask << bf_active->bitwidth - bit_shifts__backup_buffer);
		mask__backup_buffer = ~(memory_width_mask  << bit_shifts__backup_buffer);
		// printf("DBG\tEXTRACTION: mask_active = 0x%08x, mask_buffer = 0x%08x\n", mask__active_buffer, mask__backup_buffer);
		// printf("DBG\tEXTRACTION: bits_active = %d, bits_buffer = %d\n", bit_shifts__active_buffer, bit_shifts__backup_buffer);
		// retrieve rx
		data_active = (bf_active->data.rx  >> bit_shifts__active_buffer ) & mask__active_buffer;
		data_backup = (bf_backup->data.rx & mask__backup_buffer) << (bf_active->bitwidth - bit_shifts__backup_buffer) ;
		data->rx = data_backup | data_active;
		// printf("DBG\tEXTRACTION: rx = %#010x, buffer = %#010x, active = %#010x\n", data->rx, data_backup,data_active );
		// retrieve tx
		data_active = (bf_active->data.tx  >> bit_shifts__active_buffer ) & mask__active_buffer;
		data_backup = (bf_backup->data.tx & mask__backup_buffer) << (bf_active->bitwidth - bit_shifts__backup_buffer) ;
		data->tx = data_backup | data_active;
		
		// printf("DBG\tEXTRACTION: tx = %#010x, buffer = %#010x, active = %#010x,\n",  data->tx, data_backup,data_active);
		// printf("DBG\tEXTRACTION: tx.bf_active = %#010x, tx.bf_backup = %#010x\n",  bf_active->data.tx, bf_backup->data.tx);

		// set values for next iter
		bf_backup->last_mask = mask__backup_buffer;
		bf_backup->last_shift = bit_shifts__backup_buffer;
		bf_active->valid = 0;
		bf_backup->last_value_was_partially_extracted = 1;
		return bf_backup->id;
	} 
	else { // simple data extracting (1 buffer)
		// printf("DBG\tEXTRACTION: IF1 = 1");
		// retrieve rx
		data->rx = bf_active->data.rx & mask;
		data->rx = data->rx >> bit_shifts;
		// retrieve tx
		data->tx = bf_active->data.tx & mask;
		data->tx = data->tx >> bit_shifts;
		// printf(", rx = %#010x, tx = %#010x\n", data->rx, data->tx);

		// set values for next iter
		// if (bf_active->last_value_was_partially_extracted == 1 ){
		// 	bf_active->last_mask = mask;
		// 	bf_active->last_shift = bit_shifts+ bf_active->bitwidth;
		// 	bf_active->last_value_was_partially_extracted = 0;
		// 	return bf_active->id;
		// } else 
		if (bit_shifts < memory_width ){
			bf_active->last_mask = mask;
			bf_active->last_shift = bit_shifts;
			bf_active->last_value_was_partially_extracted = 0;
			return bf_active->id;
		} else {
			bf_active->valid = 0;
			return bf_backup->id;
		} 
	} 
}

u32 checkReceivedData(buffer_data* data, u32 mask_highest_bit, u32 mask_complement){
	if ( (data->rx & mask_highest_bit) > 0 ){
		data->rx = data->rx | mask_complement;
	} 
	if ( (data->tx & mask_highest_bit) > 0 ){
		data->tx = data->tx | mask_complement;
	} 
	// printf("DBG\tCHECK: ");
	printf("DBG\tCHECK: rx: %x (%u)| tx: %x (%u)\n", data->rx,data->rx, data->tx, data->tx);
	int dist = data->rx - data->tx;
	dist = dist > 0 ? dist : -dist;
	max_dist = dist > max_dist ? dist : max_dist;
	if(data->rx == data->tx || data->rx + 1 == data->tx || data->rx -1 == data->tx  ){
		// printf("--- S ---\n"); 
		return XST_SUCCESS;
	} 
	// printf("--- E ---\n"); 
	return XST_FAILURE;
}


u32 CheckData( axi_lite_config_data *config, u32 mem_offset_in, u32 mem_offset_out)
{
	u32 *RxPacket, *TxPacket;
	int Index = 0;
	single_buffer * buffer_1, *buffer_2;
	u32  active_buffer = 1; //1: buffer_1, 2: buffer_2
	buffer_data data ;
	u32 iter = 0, max_iter = MAX_BUFFER_SIZE;
	u32 error_vals = 0;
	u32 diff_offset = mem_offset_in - mem_offset_out;

	RxPacket = RxBufferPtr;
	TxPacket = TxBufferPtr;

	buffer_1 = &glb_buffer_1;
	buffer_2 = &glb_buffer_2;

	printf("\t- offset memory: %0d_d - %0d_d\n", mem_offset_in, mem_offset_out);
	printf("\t- num of mem_bytes: %0d_d\n", config->num_of_mem_bytes);
	printf("\t- bitwidth: %0d_d\n", config->bitwidth_d);

	//TODO: generally many duplications..better solution?
	buffer_1->bitwidth = config->bitwidth_d;
	buffer_2->bitwidth = config->bitwidth_d;
	buffer_1->id = 1; //TODO: necessary information
	buffer_2->id = 2;
	buffer_1->init_mask = config->mask_for_bitwidth;
	buffer_2->init_mask = config->mask_for_bitwidth;
	buffer_1->init_bit_shifts = 0; //TODO: not necessary
	buffer_2->init_bit_shifts = buffer_1->init_bit_shifts;
	buffer_1->valid = 0;
	buffer_2->valid = 0;

	max_dist = 0;

	for(Index = mem_offset_in, iter = 0; iter < max_iter; iter++) {
		if(updateBufferCheck(buffer_1, RxPacket[Index-diff_offset], TxPacket[Index] ) == 1){
			// printf("DBG\tRx[%d]=%x, Tx[%d]=%x",Index-diff_offset, RxPacket[Index-diff_offset], Index, TxPacket[Index] );
			Index++;
				//  printf(" %d - new Index = %d\n", iter,Index);
		}
		if(updateBufferCheck(buffer_2, RxPacket[Index-diff_offset], TxPacket[Index] ) == 1){
			// printf("DBG\tRx[%d]=%x, Tx[%d]=%x",Index-diff_offset, RxPacket[Index-diff_offset], Index, TxPacket[Index] );
			Index++;
				//  printf(" %d - new Index = %d\n",iter, Index);
		}

		// printf("DBG\tEXTRACTION: iter = %d , active_buffer = %d\n",iter, active_buffer);
		if(active_buffer == 1){ //TODO: move this logic into function and set 'active_buffer_id' as parameter
			active_buffer = extractNextCompressedValue(buffer_1, buffer_2, &data);
		} else {
			active_buffer = extractNextCompressedValue(buffer_2, buffer_1, &data);
		}
//		// printf("%d - valid: %d, %d\n", iter, buffer_1->valid, buffer_2->valid);
		// printf("%d --  ", iter);
		if (checkReceivedData(&data, 1 << (config->bitwidth_d - 1), ~(config->mask_for_bitwidth)) != XST_SUCCESS){
			// printf("%d - ERROR\n ", iter);
			error_vals++;
			// break;
		} else{
			// printf("%d - SUCCESS\n ", iter);
		}


		if(iter == config->num_of_fp_values-1){
			break;
		}
	}
	printf("\tCHECKING DONE \t iter = %d (dec), err_vals = %d (dec), max_distance = %d (dec)\n", iter, error_vals, max_dist);

	sum_errors = sum_errors + error_vals;

	return XST_SUCCESS;
}
