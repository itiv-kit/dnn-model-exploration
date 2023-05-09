import shutil
import os

from basics23 import int2bin, int2bin_v1, twos, float2bin


class FileHandler:
    def __init__(self, path, remove_existing=True, reconf=False) -> None:
        self.dir_path = path
        self.reconf = reconf
        if (remove_existing):
            shutil.rmtree(self.dir_path)
            os.mkdir(self.dir_path)
            # open(dir_path+'fp_values.txt', 'a')
            # open(dir_path+'quantized_values_ext.txt', 'a')
            # open(dir_path+'quantized_values.txt', 'a')
            # open(dir_path+'streamed_values.txt', 'a')
            # open(dir_path + 'configuration.txt', 'a')

    def write_fp_values(self, version, vals_fp, scale, filename="fp_values"):
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
                    f.write(str(float2bin(float(v))) + '\n')
            # with open(self.dir_path+filename+'_1.txt', 'a') as f:
            # 	for v in vals_fp:
            # 		f.write(str(float2bin(float(v+1)))+ '\n')
            # with open(self.dir_path+filename+'_2.txt', 'a') as f:
            # 	for v in vals_fp:
            # 		f.write(str(float2bin(float(v-1)))+ '\n')
        elif (version == 2):
            with open(self.dir_path+filename+'.txt', 'a') as f:
                for v in vals_fp:
                    f.write(str(float2bin(float(v)*scale)) + '\n')

    def write_memory_bytes(self, quantized_vals_int, bitwidth, mem_bitwidth, filename="streamed_values", scale=0.0, tensor_length=0):
        bits_in_line = 0
        bits_taken_from_val = 0
        binary_value = ""
        line = ''
        ci = bitwidth-1
        print_DBGs = False
        cnt_memory_bytes = 0

        with open(self.dir_path+filename+'.txt', 'a') as f:
            if (self.reconf):
                f.write(int2bin(tensor_length, mem_bitwidth) + '\n')
                f.write(int2bin(tensor_length, mem_bitwidth) + '\n')
                f.write((float2bin(1/scale)) + '\n')
                f.write((float2bin(scale)) + '\n')
            for v in quantized_vals_int:
                if (((bits_in_line + bitwidth) <= mem_bitwidth) and ci == bitwidth-1):
                    line = 'A' + line if print_DBGs else line
                    line = (int2bin(int(v), bitwidth)) + line
                    bits_in_line = bits_in_line + bitwidth
                elif (ci == bitwidth-1):
                    line = 'B' + line if print_DBGs else line
                    binary_value = (int2bin(int(v), bitwidth))
                    while (bits_in_line != mem_bitwidth):
                        line = binary_value[ci] + line
                        ci = ci - 1
                        bits_in_line = bits_in_line + 1

                line = "." + line if print_DBGs else line
                if (bits_in_line == mem_bitwidth):
                    line = 'C' + line if print_DBGs else line
                    f.write(line + '\n')
                    line = ''
                    cnt_memory_bytes = cnt_memory_bytes + 1
                    bits_in_line = 0
                    if (ci != bitwidth-1):
                        line = 'D' + line if print_DBGs else line
                        while (ci != -1):
                            line = binary_value[ci] + line
                            ci = ci - 1
                            bits_in_line = bits_in_line + 1
                        ci = bitwidth-1
                        line = "." + line if print_DBGs else line
                        binary_value = ''

            if (bits_in_line != 0):
                while (bits_in_line != mem_bitwidth):
                    line = '0' + line
                    bits_in_line = bits_in_line + 1
                line = 'E' + line if print_DBGs else line
                f.write(line + '\n')
                cnt_memory_bytes = cnt_memory_bytes + 1

        return cnt_memory_bytes

    def write_quantized_values(self, version, quantized_vals_int, bitwidth=8, filename="results"):
        if (version == 1):
            with open(self.dir_path+filename+'_ext.txt', 'a') as f:
                for v in quantized_vals_int:
                    f.write((int2bin(int(v), bitwidth)) + '  \t' + hex((int(v)+2**32) & (2**bitwidth)-1) + '\n')
        elif (version == 2):
            with open(self.dir_path+filename+'_ext1.txt', 'a') as f:
                for v in quantized_vals_int:
                    f.write(str(v) + '  \t' + str(int(v)) + '  \t' + (int2bin(int(v), bitwidth)) + '\n')
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
            f.write(int2bin(run_id, mem_bitwidth) + '\n')
            f.write(int2bin(memory_bytes, mem_bitwidth) + '\n')
            f.write(int2bin(tensor_length, mem_bitwidth) + '\n')
            f.write(int2bin(bitwidth, mem_bitwidth) + '\n')
            f.write((float2bin(1/scale)) + '\n')
            f.write((float2bin(scale)) + '\n')
            f.write(int2bin(tensor_length, mem_bitwidth) + '\n')
