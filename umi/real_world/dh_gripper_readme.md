# 大寰夹爪使用手册

## 准备工作

1. 请确保485转接到USB；
2. 请确保设备正常供电；
3. 将USB插入到ubuntu上面，并按照[操作指导](https://blog.csdn.net/datase/article/details/108054900)将夹爪的端口名称改成'ttyUSBDH'，不然后续插拔会改变端口号，不利于调试

## 基础测试

运行`umi\real_world\dh_gripper_test.py`，会依次出现夹爪初始化，夹爪完全关闭，最后夹爪完全打开的现象

## 注意事项

1. 关注`GetTargetPosition()`和`GetCurrentPosition()`的输出，有可能是输出的千分比，也有可能宽度，需要转换成以米为单位的宽度；
2. 关注`GetTargetSpeed()`，有可能输出的是百分比，有可能输出的是实际的速度，需要转换成以`m/s`为单位的速度；
3. 关注`GetTargetForce()`需要以`N`为单位的力