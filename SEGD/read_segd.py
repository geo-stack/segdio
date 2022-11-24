from struct import unpack, pack
from datetime import datetime, timedelta
import numpy

def pbcd2dec(pbcd):
    ''' Returns decimal number from packed Binary Coded Decimal'''
    
    decode = 0
    
    for n in pbcd:
        unpck   =   divmod(n,16)
        decode  =   (10*unpck[0]+unpck[1])+100*decode
    
    return decode

def convert_24bit(byte_string1, byte_string2, byte_string3):
    return (byte_string1<<24|byte_string2<<16|byte_string3<<8)>>8

# Convert 32 bits to their IEEEE 754
# Taken from https://gist.github.com/AlexEshoo/d3edc53129ed010b0a5b693b88c7e0b5
def ieee_754_conversion(n, sgn_len=1, exp_len=8, mant_len=23):
    """
    Converts an arbitrary precision Floating Point number.
    Note: Since the calculations made by python inherently use floats, the accuracy is poor at high precision.
    :param n: An unsigned integer of length `sgn_len` + `exp_len` + `mant_len` to be decoded as a float
    :param sgn_len: number of sign bits
    :param exp_len: number of exponent bits
    :param mant_len: number of mantissa bits
    :return: IEEE 754 Floating Point representation of the number `n`
    """
    if n >= 2 ** (sgn_len + exp_len + mant_len):
        raise ValueError("Number n is longer than prescribed parameters allows")

    sign = (n & (2 ** sgn_len - 1) * (2 ** (exp_len + mant_len))) >> (exp_len + mant_len)
    exponent_raw = (n & ((2 ** exp_len - 1) * (2 ** mant_len))) >> mant_len
    mantissa = n & (2 ** mant_len - 1)

    sign_mult = 1
    if sign == 1:
        sign_mult = -1

    if exponent_raw == 2 ** exp_len - 1:  # Could be Inf or NaN
        if mantissa == 2 ** mant_len - 1:
            return float('nan')  # NaN

        return sign_mult * float('inf')  # Inf

    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)

    if exponent_raw == 0:
        mant_mult = 0  # Gradual Underflow
    else:
        mant_mult = 1

    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult


def read_traces(file_ptr, samples, traces, hdr_length, format):
    data = numpy.empty((traces,samples),dtype=numpy.float)
    data_raw = 0

    for trace in range(traces):
        file_ptr.seek(hdr_length, 1)
        if format == 8058:
            # 32 bit data
            data_raw = file_ptr.read(4 * samples)
        else:
            # 24 bit data
            data_raw = file_ptr.read(3 * samples)

        for sample in range(samples):
            if format == 8058:
                data[trace, sample] = ieee_754_conversion(int.from_bytes(data_raw[4*sample:4*sample+4], 'big'))
            else:
                data[trace, sample] = data[trace, sample] = convert_24bit(data_raw[3*sample], data_raw[3*sample + 1], data_raw[3*sample+2])
    return data

def read_traces_header(file_ptr, traces, format, channel_set_header):
    for trace in range(traces):
      # check extended header length
        trc_hdr_1 = unpack('>20B',file_ptr.read(20))
        channel_set_header._hdr_length = 20+32*trc_hdr_1[9]
        
        # read extended trace header
        ext_trc_hdr_1 = unpack('>32B',file_ptr.read(32))
        
        # calculate number of samples per trace
        # can be extracted from extended header byte pos 7-10
        channel_set_header._samples = int.from_bytes(ext_trc_hdr_1[7:10], 'big')
        
        # calculate trace length for ease of use with a file pointer
        if format == 8058:
            # 32 bit data
            channel_set_header._trace_length = channel_set_header._hdr_length+channel_set_header._samples*4
        else:
            # 24 bit data
            channel_set_header._trace_length = channel_set_header._hdr_length+channel_set_header._samples*3
        
        channel_set_header.trace_headers[trace] = SEGD_trace(trc_hdr_1, ext_trc_hdr_1)
        
        file_ptr.seek(channel_set_header._hdr_length - 20 - 32, 1)
        if format == 8058:
            # 32 bit data
            file_ptr.read(4 * channel_set_header._samples)
        else:
            # 24 bit data
            file_ptr.read(3 * channel_set_header._samples)
            
class SEGD_trace(object):

    def __init__(self, trc_hdr, ext_trc_hdr_1):

        self.trace_number           = int.from_bytes(trc_hdr[4:6], 'big')
        self.receiver_line_number   = int.from_bytes(ext_trc_hdr_1[0:3], 'big')
        self.receiver_point_number  = int.from_bytes(ext_trc_hdr_1[3:6], 'big')
        self.receiver_point_index   = ext_trc_hdr_1[6]

    def __str__(self):
        # Trace header information
        readable_output = 'Trace number: \t\t\t {0}\n'.format(self.trace_number)
        readable_output += 'Receiver line number: \t\t {0}\n'.format(self.receiver_line_number)
        readable_output += 'Receiver point number: \t\t {0}\n'.format(self.receiver_point_number)
        readable_output += 'Receiver point index: \t\t {0}\n\n'.format(self.receiver_point_index)

        return readable_output

class SEGD_channel(object):

    def __init__(self, hdr_block):
        
        ch_set_hdr      =   unpack('>32B', hdr_block)
        
        self._ch_set    =   pbcd2dec(ch_set_hdr[1:2])
        self.start      =   (ch_set_hdr[2]*2**8+ch_set_hdr[3])*2
        self.stop       =   (ch_set_hdr[4]*2**8+ch_set_hdr[5])*2

        mp_factor  =   bin(ch_set_hdr[7])+bin(ch_set_hdr[6])[2:].zfill(8)

        # If the gain field is set to zero and the above field will only be 10 characters long.
        # For the calculation below, the number needs to be padded to 12 bits.
        if mp_factor=='0b000000000':
            mp_factor = '0b0000000000000000'
        
        channel_gain   =   0
        for idx, gain_step in enumerate(range(4,-11,-1)):
            channel_gain +=2**gain_step*eval(mp_factor[idx+3])

        if mp_factor[2]=='1':
            channel_gain *= -1

        self.mp_factor      =   2**channel_gain

        self.channels       =   pbcd2dec(ch_set_hdr[8:10])
        self.type           =   divmod(ch_set_hdr[10],16)[0]
        self.sample_per_channel =   2**divmod(ch_set_hdr[11],16)[0]
        self.alias_filter_freq  =   pbcd2dec(ch_set_hdr[12:14])
        self.alias_filter_slope =   divmod(ch_set_hdr[14],16)[0]*100+pbcd2dec(ch_set_hdr[15:16])
        self.hp_filter_freq     =   pbcd2dec(ch_set_hdr[16:18])
        self.hp_filter_slope    =   divmod(ch_set_hdr[18],16)[0]*100+pbcd2dec(ch_set_hdr[19:20])
        self.streamer_no    =   ch_set_hdr[30]
        self.array_forming  =   ch_set_hdr[31]
        
        self.trace_headers = numpy.empty(self.channels, dtype=object)

    def __str__(self):
        # Channel set header information
        readable_output = 'Start of record: \t {0}ms\n'.format(self.start)
        readable_output += 'Stop of record: \t {0}ms\n'.format(self.stop)
        readable_output += 'MP factor: \t\t {0}\n'.format(self.mp_factor)
        readable_output += 'Channels: \t\t {0} \t(type {1}, samples per channel {2})\n'\
            .format(self.channels,self.type,self.sample_per_channel)
        readable_output += 'Alias filter freq: \t {0} \t(slope {1})\n'\
            .format(self.alias_filter_freq,self.alias_filter_slope)
        readable_output += 'Low cut filter freq: \t {0} \t(slope {1})\n'\
                    .format(self.hp_filter_freq,self.hp_filter_slope)
        readable_output += 'Streamer number: \t {0}\n'.format(self.streamer_no)
        readable_output += 'Array forming: \t\t {0}\n\n'.format(self.array_forming)

        return readable_output

class SEGD(object):

    def __init__(self,file_name = ''):
        self.file_name = file_name
    
        if self.file_name:
            self.populate_header()

    def populate_header(self):

        f = open(self.file_name,'rb')

        # General header block #1
        gen_hdr_1   =   unpack('>32B',f.read(32))
        # General header block #2
        gen_hdr_2   =   unpack('>32B',f.read(32))
        # General header block #3
        gen_hdr_3   =   f.read(32)
        
        # Header block #1
        self.file_number    =   pbcd2dec(gen_hdr_1[0:2])
        self.segd_format    =   pbcd2dec(gen_hdr_1[2:4])
        
        # Decode timestamp and place result in a datetime object
        year                =   pbcd2dec(gen_hdr_1[10:11])
        julian_day          =   divmod(gen_hdr_1[11],16)[1]*100+pbcd2dec(gen_hdr_1[12:13])
        hour                =   pbcd2dec(gen_hdr_1[13:14])
        minute              =   pbcd2dec(gen_hdr_1[13:14])
        second              =   pbcd2dec(gen_hdr_1[15:16])
        
        self.time_stamp     =   datetime(year,1,1,hour,minute,second)+timedelta(days=julian_day-1)
        
        self._additional_hdr_blocks =   divmod(gen_hdr_1[11],16)[0]

        self.dt             =   divmod(gen_hdr_1[22],16)[0]*1e-3 # skip fractions
        self.trace_length   =   (divmod(gen_hdr_1[25],16)[1]*10+pbcd2dec(gen_hdr_1[26:27])/10.)*1.024
        
        self.channel_sets   =   pbcd2dec(gen_hdr_1[28:29])
        self._skew_blocks   =   pbcd2dec(gen_hdr_1[29:30])

        self._extended_hdr_blocks   =   pbcd2dec(gen_hdr_1[30:31])
        self._external_hdr_blocks   =   pbcd2dec(gen_hdr_1[31:32])


        # Header block #2
        # If using extended file_number (First 4 bytes FFFF), take 3 first bytes of second header
        if(self.file_number == pbcd2dec((0xff, 0xff))):
            self.file_number = int.from_bytes(gen_hdr_2[0:3], 'big')
            
        if self._extended_hdr_blocks  == 165:
            self._extended_hdr_blocks   =   gen_hdr_2[5]*256+gen_hdr_2[6]
                
        if self._external_hdr_blocks  == 165:
            self._external_hdr_blocks   =   gen_hdr_2[7]*256+gen_hdr_2[8]
    
        if self.trace_length == 170.496:
            self.trace_length   =   (gen_hdr_2[14]*2**16+gen_hdr_2[15]*2**8+gen_hdr_2[16])*1e-3
    
        self.segd_rev   =   pbcd2dec(gen_hdr_2[10:11])+pbcd2dec(gen_hdr_2[11:12])/10.
        self._extended_trace_length     =   gen_hdr_2[31]
        
        # Header block #3
        self.source_line_number = int.from_bytes(gen_hdr_3[3:6], 'big') + int.from_bytes(gen_hdr_3[6:8], 'big') / 100
        self.source_point_number = int.from_bytes(gen_hdr_3[8:11], 'big') + int.from_bytes(gen_hdr_3[11:13], 'big') / 100
        self.source_point_index = gen_hdr_3[13]
        # rev 3.0 introduced a fine grain timestamp in bytes 1-8 in Header block #3
        self._gps_timestamp = unpack('>q', gen_hdr_3[:8])

        self.channel_set_headers = [SEGD_channel(f.read(32)) for _ in range(self.channel_sets)]

        # skip Host recording sys, Line ID for cables and Shot time/reel number
        f.seek(32*3,1)
    
        self.client_name    =   unpack('>32s',f.read(32))[0].split(b'\x00')[0]
        self.contractor     =   unpack('>32s',f.read(32))[0].split(b'\x00')[0]
        self.survey         =   unpack('>32s',f.read(32))[0].split(b'\x00')[0]
        self.project        =   unpack('>32s',f.read(32))[0].split(b'\x00')[0]

        # jump to first trace hdr (4 default hdr blocks plus extended and external hdr blocks).
        f.seek((self._extended_hdr_blocks+self._external_hdr_blocks-7)*32,1)
        self._channel_set_entry_points(f)
    
        f.close()

    def _channel_set_entry_points(self, file_ptr):
        for ch_hdr in self.channel_set_headers:
            
            # store entry point position
            ch_hdr._file_ptr = file_ptr.tell()
            
            if(ch_hdr.channels > 0):
                # jump to next channel set
                read_traces_header(file_ptr, ch_hdr.channels , self.segd_format, ch_hdr)
    
    def data(self, channel_set):
        '''Returns a numpy array of the data in the selected channelset'''
        f = open(self.file_name, 'rb')
    
        f.seek(self.channel_set_headers[channel_set]._file_ptr, 0)
        samples     = self.channel_set_headers[channel_set]._samples
        traces      = self.channel_set_headers[channel_set].channels
        hdr_length  = self.channel_set_headers[channel_set]._hdr_length
    
        return read_traces(f,samples,traces,hdr_length, self.segd_format)

    def __str__(self):
    
        # Global header information
        readable_output = 'SEG-D file header:\n\n'
        readable_output += 'File name:   \t\t {0}\n'.format(self.file_name)
        readable_output += 'File number: \t\t {0}\n'.format(self.file_number)
        readable_output += 'File Format: \t\t {0} rev {1}\n'.format(self.segd_format,self.segd_rev)
        readable_output += 'Time stamp:  \t\t {0}\n'.format(self.time_stamp.ctime())
        readable_output += 'Trace length:\t\t {0}s\n'.format(self.trace_length)
        readable_output += 'Sample rate: \t\t {0}s\n\n'.format(self.dt)
        readable_output += 'Source line number: \t\t {0}\n'.format(self.source_line_number)
        readable_output += 'Source point number: \t\t {0}\n'.format(self.source_point_number)
        readable_output += 'Source point index: \t\t {0}\n\n\n'.format(self.source_point_index)
        
        for idx,ch_set in enumerate(self.channel_set_headers):
            readable_output += 'Channel set {0}:\n'.format(idx)
            readable_output += ch_set.__str__()
        
        return readable_output

def read_header(file):

    '''Returns SEGD_header object'''

    header = SEGD(file)

    return header