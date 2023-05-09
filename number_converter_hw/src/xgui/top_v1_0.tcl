# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "BUFFER_SIZE_FIFO_AKA_ACCEL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "INPUT_BITWIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAXBITWIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_QUANTIZERS" -parent ${Page_0}


}

proc update_PARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL { PARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL } {
	# Procedure called to update BUFFER_SIZE_FIFO_AKA_ACCEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL { PARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL } {
	# Procedure called to validate BUFFER_SIZE_FIFO_AKA_ACCEL
	return true
}

proc update_PARAM_VALUE.INPUT_BITWIDTH { PARAM_VALUE.INPUT_BITWIDTH } {
	# Procedure called to update INPUT_BITWIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.INPUT_BITWIDTH { PARAM_VALUE.INPUT_BITWIDTH } {
	# Procedure called to validate INPUT_BITWIDTH
	return true
}

proc update_PARAM_VALUE.MAXBITWIDTH { PARAM_VALUE.MAXBITWIDTH } {
	# Procedure called to update MAXBITWIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAXBITWIDTH { PARAM_VALUE.MAXBITWIDTH } {
	# Procedure called to validate MAXBITWIDTH
	return true
}

proc update_PARAM_VALUE.MAX_QUANTIZERS { PARAM_VALUE.MAX_QUANTIZERS } {
	# Procedure called to update MAX_QUANTIZERS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_QUANTIZERS { PARAM_VALUE.MAX_QUANTIZERS } {
	# Procedure called to validate MAX_QUANTIZERS
	return true
}


proc update_MODELPARAM_VALUE.MAXBITWIDTH { MODELPARAM_VALUE.MAXBITWIDTH PARAM_VALUE.MAXBITWIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAXBITWIDTH}] ${MODELPARAM_VALUE.MAXBITWIDTH}
}

proc update_MODELPARAM_VALUE.MAX_QUANTIZERS { MODELPARAM_VALUE.MAX_QUANTIZERS PARAM_VALUE.MAX_QUANTIZERS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_QUANTIZERS}] ${MODELPARAM_VALUE.MAX_QUANTIZERS}
}

proc update_MODELPARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL { MODELPARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL PARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL}] ${MODELPARAM_VALUE.BUFFER_SIZE_FIFO_AKA_ACCEL}
}

proc update_MODELPARAM_VALUE.INPUT_BITWIDTH { MODELPARAM_VALUE.INPUT_BITWIDTH PARAM_VALUE.INPUT_BITWIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.INPUT_BITWIDTH}] ${MODELPARAM_VALUE.INPUT_BITWIDTH}
}

