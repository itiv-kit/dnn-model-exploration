import pickle
import pandas as pd
import torch
import io
import sys
import gzip
import tensorflow as tf
from pytorch_quantization import tensor_quant

import sys

def int2bin(num, num_bits):
    """converts integer into bit string and does twos complement"""
    bits = num_bits -1
    neg = False
    if (num < 0):
        neg = True
        num = num +1

    msb = 2**bits
    num = abs(num)
    str_result = ""
    while (msb >= 1):
        # print(msb, num)
        if(num >= msb):
            num = num - msb
            if neg:
                str_result += '0'
            else:
                str_result += '1'
        else:
            if neg:
                str_result += '1'
            else:
                str_result += '0'
        msb = msb / 2
    
    # if (num == 1):
    #     str_result += '1'
    # else:
    #     str_result += '0' 
    return str_result

def int2bin_v1(num, num_bits):
    """converts integer into bit string and does twos complement"""
    bits = num_bits -1
    neg = False
    str_result = ""
    if (num < 0):
        str_result = '1'
    else:
        str_result = '0'

    msb = 2**bits
    num = abs(num)
    while (msb >= 1):
        # print(msb, num)
        if(num >= msb):
            num = num - msb
            if neg:
                str_result += '0'
            else:
                str_result += '1'
        else:
            if neg:
                str_result += '1'
            else:
                str_result += '0'
        msb = msb / 2
    
    # if (num == 1):
    #     str_result += '1'
    # else:
    #     str_result += '0' 
    return str_result


def twos(val_str, bytes):
    val = int(val_str, 2)
    b = val.to_bytes(bytes, byteorder=sys.byteorder, signed=False)                                                          
    return int.from_bytes(b, byteorder=sys.byteorder, signed=True)

def float2bin(num): #https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    v = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
    return v[:-16]

    
import struct, string
import shutil, os
from pytorch_quantization import tensor_quant

class FileHandler:

	def __init__(self, path, remove_existing=True, reconf= False) -> None:
		self.dir_path = path
		self.reconf = reconf
		if(remove_existing):
			shutil.rmtree(self.dir_path)
			os.mkdir(self.dir_path)
			# open(dir_path+'fp_values.txt', 'a')
			# open(dir_path+'quantized_values_ext.txt', 'a')
			# open(dir_path+'quantized_values.txt', 'a')
			# open(dir_path+'streamed_values.txt', 'a')
			# open(dir_path + 'configuration.txt', 'a')

	def write_fp_values(self,version,vals_fp, scale, filename="fp_values"):
		if (version == 0):
			with open(self.dir_path+filename+'.txt', 'a') as f:
				for v in vals_fp:
					f.write(str(v) + '\t'+str(float2bin(v)) + '\n')
				f.write('\nScalar:\n')
				for fp_item in [1/scale, scale]:
					f.write(str(fp_item) + '  \t'+str(float2bin(fp_item)) + '\n')
				f.write('\n')
		elif (version == 1):
			with open(self.dir_path+filename+'.txt', 'a') as f:
				for v in vals_fp:
					f.write(str(float2bin(float(v)))+ '\n')
			# with open(self.dir_path+filename+'_1.txt', 'a') as f:
			# 	for v in vals_fp:
			# 		f.write(str(float2bin(float(v+1)))+ '\n')
			# with open(self.dir_path+filename+'_2.txt', 'a') as f:
			# 	for v in vals_fp:
			# 		f.write(str(float2bin(float(v-1)))+ '\n')
		elif (version == 2):
			with open(self.dir_path+filename+'.txt', 'a') as f:
				for v in vals_fp:
					f.write(str(float2bin(float(v)*scale))+ '\n')


	def write_memory_bytes(self,quantized_vals_int, bitwidth, mem_bitwidth, filename="streamed_values", scale=0.0, tensor_length = 0):
		bits_in_line = 0
		bits_taken_from_val = 0
		binary_value = ""
		line = ''
		ci = bitwidth-1
		print_DBGs = False
		cnt_memory_bytes = 0

		with open(self.dir_path+filename+'.txt', 'a') as f:
			if (self.reconf):
				f.write(int2bin(tensor_length, mem_bitwidth)+ '\n')
				f.write(int2bin(tensor_length, mem_bitwidth)+ '\n')
				f.write((float2bin(1/scale))+ '\n')
				f.write((float2bin(scale))+ '\n')
			for v in quantized_vals_int:
				if (((bits_in_line + bitwidth) <= mem_bitwidth) and ci == bitwidth-1):
					line = 'A' + line if print_DBGs else line 
					line = (int2bin(int(v), bitwidth)) + line
					bits_in_line = bits_in_line + bitwidth
				elif (ci == bitwidth-1):
					line = 'B' + line if print_DBGs else line 
					binary_value = (int2bin(int(v), bitwidth))
					while(bits_in_line != mem_bitwidth):
						line = binary_value[ci] + line 
						ci = ci -1
						bits_in_line = bits_in_line + 1
				
				line = "." + line if print_DBGs else line 
				if(bits_in_line == mem_bitwidth) :
					line = 'C' + line if print_DBGs else line 
					f.write(line + '\n')
					line = ''
					cnt_memory_bytes = cnt_memory_bytes +1
					bits_in_line = 0
					if ( ci != bitwidth-1):
						line = 'D' + line if print_DBGs else line 
						while (ci != -1):
							line = binary_value[ci] + line
							ci = ci -1
							bits_in_line = bits_in_line + 1
						ci = bitwidth-1
						line = "." + line if print_DBGs else line 
						binary_value = ''

			if (bits_in_line != 0):
				while(bits_in_line != mem_bitwidth):
					line = '0' + line 
					bits_in_line = bits_in_line + 1
				line = 'E' + line if print_DBGs else line 
				f.write(line + '\n')
				cnt_memory_bytes = cnt_memory_bytes +1

		return cnt_memory_bytes;

	def write_quantized_values(self,version, quantized_vals_int, bitwidth=8, filename="results"):
		if(version == 1):
			with open(self.dir_path+filename+'_ext.txt', 'a') as f:
				for v in quantized_vals_int:
					f.write((int2bin(int(v), bitwidth))+  '  \t'+ hex((int(v)+2**32) & (2**bitwidth)-1)+  '\n')
		elif (version == 2):
			with open(self.dir_path+filename+'_ext1.txt', 'a') as f:
				for v in quantized_vals_int:
					f.write(str(v) + '  \t' +str(int(v)) + '  \t'+ (int2bin(int(v), bitwidth)) + '\n')
		elif (version == 3):
			with open(self.dir_path+filename+'.txt', 'a') as f:
				for v in quantized_vals_int:
					f.write((int2bin_v1(int(v), bitwidth)) + '\n')
		elif (version == 4):
			with open(self.dir_path+filename+'.txt', 'a') as f:
				for v in quantized_vals_int:
					f.write((int2bin(int(v), bitwidth)) + '\n')

	def write_configurationFile(self, mem_bitwidth, run_id, memory_bytes, tensor_length, bitwidth, scale, filename="configuration"):
		with open(self.dir_path+filename+'.txt', 'a') as f:
			f.write(int2bin( run_id, mem_bitwidth)+ '\n')
			f.write(int2bin(memory_bytes, mem_bitwidth)+ '\n')
			f.write(int2bin(tensor_length, mem_bitwidth)+ '\n')
			f.write(int2bin(bitwidth, mem_bitwidth)+ '\n')
			f.write(float2bin(scale)+ '\n')
			f.write(float2bin(1/scale)+ '\n')
			f.write(int2bin(tensor_length, mem_bitwidth)+ '\n')



import struct, string
import shutil, os
from pytorch_quantization import tensor_quant

class TestdataConfiguration:
	def __init__(self,tb, version,mem_bits=32, remove_existing=False, reconf = False) -> None:
		self.max_bitwidth = 16
		self.vals_fp = []
		self.quantized_vals_int = []  
		self.bitwidth = 0
		self.tensor_numel = 0
		self.mem_bitwidth = mem_bits
		self.dir_path = './'+tb+'/v'+str(version) + '/'
		self.run_id = 0
		self.tb = tb
		self.total_membytes = 0
		self.reconf = reconf
		self.files = FileHandler(self.dir_path, remove_existing, reconf)

	def addSmallRun(self, num_bits) -> None:
		torch.manual_seed(0)
		x = torch.randn(32)
		self.addNewRun(x,num_bits)

	def addLargeRun(self, num_bits) -> None:
		torch.manual_seed(0)
		x = torch.randn(1000)
		self.addNewRun(x,num_bits)

	
	def addNewRun(self,tensor, num_bits) -> None:
		# tensor = tf.cast(tensor, dtype=tf.bfloat16)
		self.run_id = self.run_id +1
		self.getQuantizedData(tensor, num_bits)
		print("NEW RUN")
		print(f"- bitwidth: {self.bitwidth}")
		print(f"- num of values: {self.tensor_numel}")
		if(self.tb == "top"):
			self.appendToFilesTop()
		elif (self.tb in ["fifo_extractor_fifo", "extractor"] ):
			self.appendToFilesExtractor()
		elif (self.tb in [ "compressor", "fifo_compressor"] ):
			self.appendToFilesCompressor()
		elif (self.tb in ["int2fp"]  ):
			self.appendToFilesInt2fp()
		elif (self.tb in ["fp2int"]  ):
			self.appendToFilesFp2Int()
		elif (self.tb in ["fp_multiplication"]  ):
			self.appendToFilesFPMulti()
		elif (self.tb in ["extract_dequant"]  ):
			self.appendToFilesExtractDequant()
		elif (self.tb in ["fifo_quant_compr"]  ):
			self.appendToFilesQuantCompr()
		elif (self.tb in ["quantization"]  ):
			self.appendToFilesQuantizer()
		else:
			print("ERROR TB NOT DEFINED")
		print(f"- run id: {self.run_id}")
		print(f"- membytes in total: {self.total_membytes}")
	
	def getQuantizedData(self,tensor, num_bits) -> None:
		self.bitwidth = num_bits
		self.tensor_numel = tensor.numel()
		self.vals_fp.clear()
		self.quantized_vals_int.clear()

		#compute quantized vals
		self.quant_x, self.scale = tensor_quant._tensor_quant(tensor,tensor.max(), num_bits=self.bitwidth, narrow_range=True, unsigned=False) # TODO: amax?

		#retrieve values
		for lvl1 in tensor:
			self.vals_fp.append(lvl1)
		for lvl1 in self.quant_x:
			self.quantized_vals_int.append(lvl1)
	
	def appendToFilesExtractor(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth)
		self.total_membytes = self.total_membytes + cnt_memory_bytes
		print("\t- results file ")
		self.files.write_quantized_values(3, self.quantized_vals_int, self.max_bitwidth)
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id, cnt_memory_bytes, self.tensor_numel, self.bitwidth, self.scale )

	def appendToFilesCompressor(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth, "results")
		self.total_membytes = self.total_membytes + cnt_memory_bytes
		print("\t- results file ")
		self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "streamed_values")
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id,  self.tensor_numel, cnt_memory_bytes,self.bitwidth, self.scale )

	def appendToFilesInt2fp(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		self.files.write_quantized_values(3, self.quantized_vals_int, self.max_bitwidth, "input")
		print("\t- results file ")
		self.files.write_fp_values(1,self.quantized_vals_int, self.scale)
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id,  self.tensor_numel, self.tensor_numel,self.bitwidth, self.scale )

	def appendToFilesFp2Int(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "input")
		print("\t- results file ")
		self.files.write_fp_values(1,self.quantized_vals_int, self.scale)
		cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth)
		self.total_membytes = self.total_membytes + cnt_memory_bytes
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id,  self.tensor_numel, self.tensor_numel,self.bitwidth, self.scale )

	def appendToFilesFPMulti(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		self.files.write_fp_values(1,self.quantized_vals_int, self.scale, "input")
		print("\t- results file ")
		self.files.write_fp_values(2,self.quantized_vals_int, 1/self.scale, "results")
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id,  self.tensor_numel, self.tensor_numel,self.bitwidth, self.scale )

	def appendToFilesExtractDequant(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth)
		self.total_membytes = self.total_membytes + cnt_memory_bytes
		print("\t- results file ")
		self.files.write_fp_values(2,self.quantized_vals_int, self.scale, "results")
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id, cnt_memory_bytes, self.tensor_numel, self.bitwidth, self.scale)
		self.files.write_quantized_values(3, self.quantized_vals_int, self.max_bitwidth, "quantized_vals")

	def appendToFilesQuantCompr(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		self.files.write_fp_values(2,self.quantized_vals_int, 1/self.scale, "input")
		print("\t- results file ")
		cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth, "streamed_values")
		self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "results_ext")
		self.total_membytes = self.total_membytes + cnt_memory_bytes
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id,  cnt_memory_bytes, self.tensor_numel,self.bitwidth, self.scale )
		print("\t- configuration file for simulation results")
		with open(self.dir_path + 'configuration_sims.txt', 'a')as f:
			f.write(str(self.run_id) + '\n')
			f.write(str(self.bitwidth)+ '\n')
			f.write(str(self.tensor_numel)+ '\n')
			f.write(str(cnt_memory_bytes)+ '\n')

	def appendToFilesQuantizer(self):
		print("-- Writing Files --")
		print("\t- stream file ")
		self.files.write_fp_values(2,self.quantized_vals_int, 1/self.scale, "input")
		print("\t- results file ")
		self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "results")
		print('\t- configuration file')
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id,  self.tensor_numel, self.tensor_numel,self.bitwidth, self.scale )




	def appendToFilesTop(self):
		print("-- Writing Files --")
		# print("path: "+self.dir_path)
		
		print("\t- fp_values file ")
		self.files.write_fp_values(0,self.vals_fp, self.scale)

		print("\t- quantized_values_ext file ")
		self.files.write_quantized_values(2, self.quantized_vals_int, self.max_bitwidth)

		print("\t- quantized_values file ")
		self.files.write_quantized_values(1, self.quantized_vals_int, self.bitwidth)
				
		print("\t- streamed_values file")
		cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth, "streamed_values", self.scale, self.tensor_numel)
		self.total_membytes = self.total_membytes + cnt_memory_bytes

		print("\t- c_mem_array file -- NOT IMPLEMENTED YET")
		# with open(self.dir_path + "testdata.bin", "w") as f:
		# 	my_hexdata = "1a"
		# 	scale = 16
		# 	num_of_bits = 8
		# 	f.write(bin(int(my_hexdata, scale))[2:].zfill(num_of_bits)) 

		print("\t- configuration file")
		self.files.write_configurationFile(self.mem_bitwidth, self.run_id, cnt_memory_bytes, self.tensor_numel, self.bitwidth, 0.5/(2**(self.bitwidth-1)))

		print("\t- configuration file for simulation results")
		with open(self.dir_path + 'configuration_sims.txt', 'a')as f:
			f.write(str(self.run_id) + '\n')
			f.write(str(self.bitwidth)+ '\n')
			f.write(str(self.tensor_numel)+ '\n')
			f.write(str(cnt_memory_bytes)+ '\n')
