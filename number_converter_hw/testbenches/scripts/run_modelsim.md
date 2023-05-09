# internally


set main_lib "_run/work"

vlib $main_lib
vmap main_lib $main_lib  
	source sources.tcl 

set TESTCASE tb_top

vsim $main_lib.$SIM_TOP_LEVEL
run 600 ns