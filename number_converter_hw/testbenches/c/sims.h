#ifndef SIMS_H
#define SIMS_H

#include <stdio.h>

#include "sims_fpga.h"


void readBitsFromFile(char * path, int mem_bytes, u32 *  Buffer );
void readConfigurationFile(char * path, axi_lite_config_data arr[]);

extern int sum_errors;
extern int memory_width;

#endif