import sys
import struct


def int2bin(num, num_bits):
    """converts integer into bit string and does twos complement"""
    bits = num_bits - 1
    neg = False
    if (num < 0):
        neg = True
        num = num + 1

    msb = 2**bits
    num = abs(num)
    str_result = ""
    while (msb >= 1):
        # print(msb, num)
        if (num >= msb):
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
    bits = num_bits - 1
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
        if (num >= msb):
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


def float2bin(num):  # https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
