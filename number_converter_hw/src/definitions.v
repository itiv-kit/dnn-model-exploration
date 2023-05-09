`resetall
// `define USE_DBG_VIVADO
// `define USE_LAYER_CNT
`define USE_PERFORMANCE_COUNTER
`define HW_FULL_FP32
// `define HW_FULL_INT
//  `define HW_SINGLE_FP32
// `define HW_SINGLE_INT
// `define HW_NO_FPGA
//`define OPT_FIFOS


`ifdef HW_FULL_FP32
	`define _HW_AXILITE_
	`define _HW_FP32_ // TODO: rename to bfloat16
`endif

`ifdef HW_FULL_INT
	`define _HW_AXILITE_
	`undef _HW_FP32_
`endif

`ifdef HW_SINGLE_FP32
	`undef _HW_AXILITE_
	`define _HW_FP32_
`endif

`ifdef HW_SINGLE_INT
	`undef _HW_AXILITE_
	`undef _HW_FP32_
`endif


`ifdef HW_NO_FPGA
	`define _HW_AXILITE_
	`undef _HW_FP32_
`endif
