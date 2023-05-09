import torch
from pytorch_quantization import tensor_quant

from FileHandler import FileHandler


class TestdataConfiguration:
    def __init__(self, tb, version, mem_bits=32, remove_existing=False, reconf=False) -> None:
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
        x = torch.randn(4, 2, 2, 2)
        self.addNewRun(x, num_bits)

    def addLargeRun(self, num_bits) -> None:
        torch.manual_seed(0)
        x = torch.randn(1, 10, 10, 10)
        self.addNewRun(x, num_bits)

    def addNewRun(self, tensor, num_bits) -> None:
        self.run_id = self.run_id + 1
        self.getQuantizedData(tensor, num_bits)
        print("NEW RUN")
        print(f"- bitwidth: {self.bitwidth}")
        print(f"- num of values: {self.tensor_numel}")
        if (self.tb == "top"):
            self.appendToFilesTop()
        elif (self.tb in ["fifo_extractor_fifo", "extractor"]):
            self.appendToFilesExtractor()
        elif (self.tb in ["compressor", "fifo_compressor"]):
            self.appendToFilesCompressor()
        elif (self.tb in ["int2fp"]):
            self.appendToFilesInt2fp()
        elif (self.tb in ["fp2int"]):
            self.appendToFilesFp2Int()
        elif (self.tb in ["fp_multiplication"]):
            self.appendToFilesFPMulti()
        elif (self.tb in ["extract_dequant"]):
            self.appendToFilesExtractDequant()
        elif (self.tb in ["fifo_quant_compr"]):
            self.appendToFilesQuantCompr()
        elif (self.tb in ["quantization"]):
            self.appendToFilesQuantizer()
        else:
            print("ERROR TB NOT DEFINED")
        print(f"- run id: {self.run_id}")
        print(f"- membytes in total: {self.total_membytes}")

    def getQuantizedData(self, tensor, num_bits) -> None:
        self.bitwidth = num_bits
        self.tensor_numel = tensor.numel()
        self.vals_fp.clear()
        self.quantized_vals_int.clear()

        # compute quantized vals
        self.quant_x, self.scale = tensor_quant._tensor_quant(tensor, tensor.max(
        ), num_bits=self.bitwidth, narrow_range=True, unsigned=False)  # TODO: amax?

        # retrieve values
        for lvl1 in tensor:
            for lvl2 in lvl1:
                for lvl3 in lvl2:
                    for lvl4 in lvl3:
                        self.vals_fp.append(lvl4)
        for lvl1 in self.quant_x:
            for lvl2 in lvl1:
                for lvl3 in lvl2:
                    for lvl4 in lvl3:
                        self.quantized_vals_int.append(lvl4)

    def appendToFilesExtractor(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth)
        self.total_membytes = self.total_membytes + cnt_memory_bytes
        print("\t- results file ")
        self.files.write_quantized_values(3, self.quantized_vals_int, self.max_bitwidth)
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           cnt_memory_bytes, self.tensor_numel, self.bitwidth, self.scale)

    def appendToFilesCompressor(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        cnt_memory_bytes = self.files.write_memory_bytes(
            self.quantized_vals_int, self.bitwidth, self.mem_bitwidth, "results")
        self.total_membytes = self.total_membytes + cnt_memory_bytes
        print("\t- results file ")
        self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "streamed_values")
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           self.tensor_numel, cnt_memory_bytes, self.bitwidth, self.scale)

    def appendToFilesInt2fp(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        self.files.write_quantized_values(3, self.quantized_vals_int, self.max_bitwidth, "input")
        print("\t- results file ")
        self.files.write_fp_values(1, self.quantized_vals_int, self.scale)
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           self.tensor_numel, self.tensor_numel, self.bitwidth, self.scale)

    def appendToFilesFp2Int(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "input")
        print("\t- results file ")
        self.files.write_fp_values(1, self.quantized_vals_int, self.scale)
        cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth)
        self.total_membytes = self.total_membytes + cnt_memory_bytes
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           self.tensor_numel, self.tensor_numel, self.bitwidth, self.scale)

    def appendToFilesFPMulti(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        self.files.write_fp_values(1, self.quantized_vals_int, self.scale, "input")
        print("\t- results file ")
        self.files.write_fp_values(2, self.quantized_vals_int, 1/self.scale, "results")
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           self.tensor_numel, self.tensor_numel, self.bitwidth, self.scale)

    def appendToFilesExtractDequant(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        cnt_memory_bytes = self.files.write_memory_bytes(self.quantized_vals_int, self.bitwidth, self.mem_bitwidth)
        self.total_membytes = self.total_membytes + cnt_memory_bytes
        print("\t- results file ")
        self.files.write_fp_values(2, self.quantized_vals_int, self.scale, "results")
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           cnt_memory_bytes, self.tensor_numel, self.bitwidth, self.scale)
        self.files.write_quantized_values(3, self.quantized_vals_int, self.max_bitwidth, "quantized_vals")

    def appendToFilesQuantCompr(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        self.files.write_fp_values(2, self.quantized_vals_int, 1/self.scale, "input")
        print("\t- results file ")
        cnt_memory_bytes = self.files.write_memory_bytes(
            self.quantized_vals_int, self.bitwidth, self.mem_bitwidth, "streamed_values")
        self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "results_ext")
        self.total_membytes = self.total_membytes + cnt_memory_bytes
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           cnt_memory_bytes, self.tensor_numel, self.bitwidth, self.scale)
        print("\t- configuration file for simulation results")
        with open(self.dir_path + 'configuration_sims.txt', 'a')as f:
            f.write(str(self.run_id) + '\n')
            f.write(str(self.bitwidth) + '\n')
            f.write(str(self.tensor_numel) + '\n')
            f.write(str(cnt_memory_bytes) + '\n')

    def appendToFilesQuantizer(self):
        print("-- Writing Files --")
        print("\t- stream file ")
        self.files.write_fp_values(2, self.quantized_vals_int, 1/self.scale, "input")
        print("\t- results file ")
        self.files.write_quantized_values(4, self.quantized_vals_int, self.max_bitwidth, "results")
        print('\t- configuration file')
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           self.tensor_numel, self.tensor_numel, self.bitwidth, self.scale)

    def appendToFilesTop(self):
        print("-- Writing Files --")
        # print("path: "+self.dir_path)

        print("\t- fp_values file ")
        self.files.write_fp_values(0, self.vals_fp, self.scale)

        print("\t- quantized_values_ext file ")
        self.files.write_quantized_values(2, self.quantized_vals_int, self.max_bitwidth)

        print("\t- quantized_values file ")
        self.files.write_quantized_values(1, self.quantized_vals_int, self.bitwidth)

        print("\t- streamed_values file")
        cnt_memory_bytes = self.files.write_memory_bytes(
            self.quantized_vals_int, self.bitwidth, self.mem_bitwidth, "streamed_values", self.scale, self.tensor_numel)
        self.total_membytes = self.total_membytes + cnt_memory_bytes

        print("\t- c_mem_array file -- NOT IMPLEMENTED YET")
        # with open(self.dir_path + "testdata.bin", "w") as f:
        # 	my_hexdata = "1a"
        # 	scale = 16
        # 	num_of_bits = 8
        # 	f.write(bin(int(my_hexdata, scale))[2:].zfill(num_of_bits))

        print("\t- configuration file")
        self.files.write_configurationFile(self.mem_bitwidth, self.run_id,
                                           cnt_memory_bytes, self.tensor_numel, self.bitwidth, self.scale)

        print("\t- configuration file for simulation results")
        with open(self.dir_path + 'configuration_sims.txt', 'a')as f:
            f.write(str(self.run_id) + '\n')
            f.write(str(self.bitwidth) + '\n')
            f.write(str(self.tensor_numel) + '\n')
            f.write(str(cnt_memory_bytes) + '\n')
