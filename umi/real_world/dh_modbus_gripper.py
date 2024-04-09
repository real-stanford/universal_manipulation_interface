import dh_device

m_device = dh_device.dh_device()


class dh_modbus_gripper(object):
    gripper_ID = 0x01

    def CRC16(self, nData, wLength):
        if nData == 0x00:
            return 0x0000
        wCRCWord = 0xFFFF
        poly = 0xA001
        for num in range(wLength):
            date = nData[num]
            wCRCWord = (date & 0xFF) ^ wCRCWord
            for bit in range(8):
                if (wCRCWord & 0x01) != 0:
                    wCRCWord >>= 1
                    wCRCWord ^= poly
                else:
                    wCRCWord >>= 1
        return wCRCWord

    def open(self, PortName, BaudRate):
        ret = 0
        ret = m_device.connect_device(PortName, BaudRate)
        if ret < 0:
            print("open failed")
            return ret
        else:
            print("open successful")
            return ret

    def close():
        m_device.disconnect_device()

    def WriteRegisterFunc(self, index, value):
        send_buf = [0, 0, 0, 0, 0, 0, 0, 0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x06
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = (value >> 8) & 0xFF
        send_buf[5] = value & 0xFF

        crc = self.CRC16(send_buf, len(send_buf) - 2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        send_temp = send_buf
        ret = False
        retrycount = 3

        while ret == False:
            ret = False

            if retrycount < 0:
                break
            retrycount = retrycount - 1

            wdlen = m_device.device_wrire(send_temp)
            if len(send_temp) != wdlen:
                print("write error ! write : ", send_temp)
                continue

            rev_buf = m_device.device_read(8)
            if len(rev_buf) == wdlen:
                ret = True
        return ret

    def ReadRegisterFunc(self, index):
        send_buf = [0, 0, 0, 0, 0, 0, 0, 0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x03
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = 0x00
        send_buf[5] = 0x01

        crc = self.CRC16(send_buf, len(send_buf) - 2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        send_temp = send_buf
        ret = False
        retrycount = 3

        while ret == False:
            ret = False

            if retrycount < 0:
                break
            retrycount = retrycount - 1

            wdlen = m_device.device_wrire(send_temp)
            if len(send_temp) != wdlen:
                print("write error ! write : ", send_temp)
                continue

            rev_buf = m_device.device_read(7)
            if len(rev_buf) == 7:
                value = (rev_buf[4] & 0xFF) | (rev_buf[3] << 8)
                ret = True
            # ('read value : ', value)
        return value

    def Initialization(self):
        self.WriteRegisterFunc(0x0100, 0xA5)

    # set gripper target position 0-1000
    def SetTargetPosition(self, refpos):
        self.WriteRegisterFunc(0x0103, refpos)

    # set gripper target force 20-100 %
    def SetTargetForce(self, force):
        self.WriteRegisterFunc(0x0101, force)

    # set gripper target speed 1-100 %
    def SetTargetSpeed(self, speed):
        self.WriteRegisterFunc(0x0104, speed)

    # get gripper current position
    def GetCurrentPosition(self):
        return self.ReadRegisterFunc(0x0202)

    # get gripper target position 
    def GetTargetPosition(self):
        return self.ReadRegisterFunc(0x0103)

    # get gripper current target force (Notice: Not actual force)
    def GetTargetForce(self):
        return self.ReadRegisterFunc(0x0101)

    # get gripper current target speed (Notice: Not actual speed)
    def GetTargetSpeed(self):
        return self.ReadRegisterFunc(0x0104)

    # get gripper initialization state
    def GetInitState(self):
        return self.ReadRegisterFunc(0x0200)

    # get gripper grip state
    def GetGripState(self):
        return self.ReadRegisterFunc(0x0201)

    # get states: initialize, grip, position, target position, target force
    def GetRunStates(self):
        states = [0, 0, 0, 0, 0]
        states[0] = self.GetInitState()
        states[1] = self.GetGripState()
        states[2] = self.GetCurrentPosition()
        states[3] = self.GetTargetPosition()
        states[4] = self.GetTargetForce()
        return states

    """description of class"""
