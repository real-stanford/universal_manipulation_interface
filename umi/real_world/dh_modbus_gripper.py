from umi.real_world.dh_device import dh_device
# import dh_device
import time 

class dh_modbus_gripper(object):

    def __init__(
        self, port_name, baud_rate, max_width=0.08, max_speed=0.07273, max_force=140
    ):
        self.gripper_ID = 0x01
        self.m_device = dh_device()
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.max_width = max_width
        self.max_speed = max_speed
        self.max_force = max_force

    def __enter__(self):
        self.open()
        return self

    def __exit__(self):
        self.close()
        return self

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

    def open(self):
        ret = 0
        ret = self.m_device.connect_device(self.port_name, self.baud_rate)
        if ret < 0:
            print("open failed")
            return ret
        else:
            print("open successful")
            return ret

    def close(self):
        self.m_device.disconnect_device()

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

            wdlen = self.m_device.device_wrire(send_temp)
            if len(send_temp) != wdlen:
                print("write error ! write : ", send_temp)
                continue

            rev_buf = self.m_device.device_read(8)
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

            wdlen = self.m_device.device_wrire(send_temp)
            if len(send_temp) != wdlen:
                print("write error ! write : ", send_temp)
                continue

            rev_buf = self.m_device.device_read(7)
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

    # set gripper target position
    def SetTargetAbsPosition(self, refpos):
        self.WriteRegisterFunc(0x0103, int(refpos / self.max_width * 1000.0))

    # set gripper target force
    def SetTargetAbsForce(self, force):
        self.WriteRegisterFunc(0x0101, int(force / self.max_force * 100.0))

    # set gripper target speed 1-100 %
    def SetTargetAbsSpeed(self, speed):
        self.WriteRegisterFunc(0x0104, int(speed / self.max_speed * 100.0))

    # get gripper current position
    def GetCurrentAbsPosition(self):
        return self.ReadRegisterFunc(0x0202) / 1000.0 * self.max_width

    # get gripper target position
    def GetTargetAbsPosition(self):
        return self.ReadRegisterFunc(0x0103) / 1000.0 * self.max_width

    # get gripper current target force (Notice: Not actual force)
    def GetTargetAbsForce(self):
        return self.ReadRegisterFunc(0x0101) / 100.0 * self.max_force

    # get gripper current target speed (Notice: Not actual speed)
    def GetTargetAbsSpeed(self):
        return self.ReadRegisterFunc(0x0104) / 100.0 * self.max_speed

    # get gripper initialization state
    # 0，未初始化，1，初始化成功，2，初始化中
    def GetInitState(self):
        return self.ReadRegisterFunc(0x0200)

    # get gripper grip state
    # 0，运动中，1，到达位置，2，夹住物体，3，物体掉落
    def GetGripState(self):
        return self.ReadRegisterFunc(0x0201)

    # get states: initialize, grip, position, target position, target force
    def GetRunStates(self):
        states = {}
        states["state"] = self.GetGripState()
        states["position"] = self.GetCurrentAbsPosition()  # 单位m
        states["velocity"] = (
            self.GetTargetAbsSpeed()
        )  # 单位m/s，注意返回的是预设的速度，而不是当前的速度，跟UMI原始代码可能有出入
        states["force_motor"] = (
            self.GetTargetAbsForce()
        )  # 单位N，注意返回的是预设的力，而不是当前的力，跟UMI原始代码可能有出入
        states["measure_timestamp"] = time.time()
        return states

    """description of class"""
