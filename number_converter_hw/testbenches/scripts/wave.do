add wave -group dut_IF2 {-in}  /tb_top/DUT/*
add wave -group wrapper_slave  {-in} /tb_top/DUT/top_intern/wrapper_axis_slave/*
add wave -group extractor   {-in} /tb_top/DUT/top_intern/extractor/*
add wave -group arbiter_to_dequant   {-in} /tb_top/DUT/top_intern/arbiter_to_dequantizers/*
add wave -group dequant0 
add wave -group dequant1 
add wave -group arbiter_to_accel   {-in} /tb_top/DUT/top_intern/arbiter_to_accel/*
add wave -group fifo  {-in} /tb_top/DUT/top_intern/fifo/*
add wave -group arbiter_to_quant   {-in} /tb_top/DUT/top_intern/arbiter_to_quantizers/*
add wave -group quant0 
add wave -group quant1  
add wave -group arbiter_to_dma   {-in} /tb_top/DUT/top_intern/arbiter_to_dma/*
add wave -group compression   {-in} /tb_top/DUT/top_intern/compression/*
add wave -group wrapper_master   {-in} /tb_top/DUT/top_intern/wrapper_axis_master/*

add wave -group dut_IF2 {-out}  /tb_top/DUT/*
add wave -group wrapper_slave  {-out} /tb_top/DUT/top_intern/wrapper_axis_slave/*
add wave -group extractor   {-out} /tb_top/DUT/top_intern/extractor/*
add wave -group arbiter_to_dequant   {-out} /tb_top/DUT/top_intern/arbiter_to_dequantizers/*
add wave -group dequant0 
add wave -group dequant1 
add wave -group arbiter_to_accel   {-out} /tb_top/DUT/top_intern/arbiter_to_accel/*
add wave -group fifo  {-out} /tb_top/DUT/top_intern/fifo/*
add wave -group arbiter_to_quant   {-out} /tb_top/DUT/top_intern/arbiter_to_quantizers/*
add wave -group quant0 
add wave -group quant1  
add wave -group arbiter_to_dma   {-out} /tb_top/DUT/top_intern/arbiter_to_dma/*
add wave -group compression   {-out} /tb_top/DUT/top_intern/compression/*
add wave -group wrapper_master   {-out} /tb_top/DUT/top_intern/wrapper_axis_master/*