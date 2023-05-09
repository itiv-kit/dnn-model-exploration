# design
vlog -novopt -O0 +acc=mnprt -work $main_lib ../../hw/*.v

# testbench
vlog -novopt -O0 +acc=mnprt -work $main_lib ../$TESTCASE.sv

set SIM_TIME "3 ms"
set SIM_TOP_LEVEL "$TESTCASE"
