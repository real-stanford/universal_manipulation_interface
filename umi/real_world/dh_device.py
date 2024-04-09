import serial

class dh_device(object):

    def __init__(self):
        self.serialPort = serial.Serial()

    def connect_device(self, portname, Baudrate):
        ret = -1
        # print('portname: ', portname)
        self.serialPort.port = portname
        self.serialPort.baudrate = Baudrate
        self.serialPort.bytesize = 8
        self.serialPort.parity = "N"
        self.serialPort.stopbits = 1
        self.serialPort.set_output_flow_control = "N"
        self.serialPort.set_input_flow_control = "N"

        self.serialPort.open()
        if self.serialPort.isOpen():
            print("Serial Open Success")
            ret = 0
        else:
            print("Serial Open Error")
            ret = -1
        return ret

    def disconnect_device(self):
        if self.serialPort.isOpen():
            self.serialPort.close()
        else:
            return

    def device_wrire(self, write_data):
        write_lenght = 0
        if self.serialPort.isOpen():
            write_lenght = self.serialPort.write(write_data)
            if write_lenght == len(write_data):
                return write_lenght
            else:
                print("write error ! send_buff :", write_data)
                return 0
        else:
            return -1

    def device_read(self, wlen):
        responseData = [0, 0, 0, 0, 0, 0, 0, 0]
        if self.serialPort.isOpen():
            responseData = self.serialPort.readline(wlen)
            # print('read_buff: ',responseData.hex())
            return responseData
        else:
            return -1

    """description of class"""
