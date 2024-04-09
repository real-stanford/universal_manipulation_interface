import serial

serialPort = serial.Serial()


class dh_device(object):

    def connect_device(self, portname, Baudrate):
        ret = -1
        # print('portname: ', portname)
        serialPort.port = portname
        serialPort.baudrate = Baudrate
        serialPort.bytesize = 8
        serialPort.parity = "N"
        serialPort.stopbits = 1
        serialPort.set_output_flow_control = "N"
        serialPort.set_input_flow_control = "N"

        serialPort.open()
        if serialPort.isOpen():
            print("Serial Open Success")
            ret = 0
        else:
            print("Serial Open Error")
            ret = -1
        return ret

    def disconnect_device():
        if serialPort.isOpen():
            serialPort.close()
        else:
            return

    def device_wrire(self, write_data):
        write_lenght = 0
        if serialPort.isOpen():
            write_lenght = serialPort.write(write_data)
            if write_lenght == len(write_data):
                return write_lenght
            else:
                print("write error ! send_buff :", write_data)
                return 0
        else:
            return -1

    def device_read(self, wlen):
        responseData = [0, 0, 0, 0, 0, 0, 0, 0]
        if serialPort.isOpen():
            responseData = serialPort.readline(wlen)
            # print('read_buff: ',responseData.hex())
            return responseData
        else:
            return -1

    """description of class"""
