import os
from subprocess import Popen, PIPE, DEVNULL
import fcntl
import pathlib


def create_usb_list():
    device_list = list()
    lsusb_out = Popen('lsusb -v', shell=True, bufsize=64, 
                      stdin=PIPE, stdout=PIPE, stderr=DEVNULL, 
                      close_fds=True).stdout.read().strip().decode('utf-8')
    usb_devices = lsusb_out.split('%s%s' % (os.linesep, os.linesep))
    for device_categories in usb_devices:
        if not device_categories:
            continue
        categories = device_categories.split(os.linesep)
        device_stuff = categories[0].strip().split()
        bus = device_stuff[1]
        device = device_stuff[3][:-1]
        device_dict = {'bus': bus, 'device': device}
        device_info = ' '.join(device_stuff[6:])
        device_dict['description'] = device_info
        for category in categories:
            if not category:
                continue
            categoryinfo = category.strip().split()
            if categoryinfo[0] == 'iManufacturer':
                manufacturer_info = ' '.join(categoryinfo[2:])
                device_dict['manufacturer'] = manufacturer_info
            if categoryinfo[0] == 'iProduct':
                device_info = ' '.join(categoryinfo[2:])
                device_dict['device'] = device_info
        path = '/dev/bus/usb/%s/%s' % (bus, device)
        device_dict['path'] = path

        device_list.append(device_dict)
    return device_list

def reset_usb_device(dev_path):
    USBDEVFS_RESET = 21780
    try:
        f = open(dev_path, 'w', os.O_WRONLY)
        fcntl.ioctl(f, USBDEVFS_RESET, 0)
        print('Successfully reset %s' % dev_path)
    except PermissionError as ex:
        raise PermissionError('Try running "sudo chmod 777 {}"'.format(dev_path))

def reset_all_elgato_devices():
    """
    Find and reset all Elgato capture cards.
    Required to workaround a firmware bug.
    """
    
    # enumerate UBS device to find Elgato Capture Card
    device_list = create_usb_list()
    
    for dev in device_list:
        if 'Elgato' in dev['description']:
            dev_usb_path = dev['path']
            reset_usb_device(dev_usb_path)

def get_sorted_v4l_paths(by_id=True):
    """
    If by_id, sort devices by device name + serial number (preserves device order)
    else, sort devices by usb bus id (preserves usb port order)
    """
    
    dirname = 'by-id'
    if not by_id:
        dirname = 'by-path'
    v4l_dir = pathlib.Path('/dev/v4l').joinpath(dirname)

    valid_paths = list()
    for dev_path in sorted(v4l_dir.glob("*video*")):
        name = dev_path.name

        # only keep devices ends with "index0"
        # since they are the only valid video devices
        index_str = name.split('-')[-1]
        assert index_str.startswith('index')
        index = int(index_str[5:])
        if index == 0:
            valid_paths.append(dev_path)

    result = [str(x.absolute()) for x in valid_paths]

    return result
