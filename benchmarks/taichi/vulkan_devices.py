"""Read-only Vulkan physical device inventory, for explicit TI_VISIBLE_DEVICE selection."""
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import struct

class ApplicationInfo(C.Structure):
    _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('pApplicationName', C.c_char_p),
                ('applicationVersion', C.c_uint32), ('pEngineName', C.c_char_p),
                ('engineVersion', C.c_uint32), ('apiVersion', C.c_uint32)]

class InstanceCreateInfo(C.Structure):
    _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('flags', C.c_uint32),
                ('pApplicationInfo', C.POINTER(ApplicationInfo)), ('enabledLayerCount', C.c_uint32),
                ('ppEnabledLayerNames', C.c_void_p), ('enabledExtensionCount', C.c_uint32),
                ('ppEnabledExtensionNames', C.c_void_p)]

vk = C.WinDLL('vulkan-1.dll')
vk.vkCreateInstance.argtypes = [C.POINTER(InstanceCreateInfo), C.c_void_p, C.POINTER(C.c_void_p)]
vk.vkCreateInstance.restype = C.c_int32
vk.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)]
vk.vkEnumeratePhysicalDevices.restype = C.c_int32
vk.vkGetPhysicalDeviceProperties.argtypes = [C.c_void_p, C.c_void_p]
vk.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
app = ApplicationInfo(0, None, b'CCP benchmark inventory', 1, b'none', 1, (1 << 22) | (2 << 12))
info = InstanceCreateInfo(1, None, 0, C.pointer(app), 0, None, 0, None)
instance = C.c_void_p()
assert vk.vkCreateInstance(C.byref(info), None, C.byref(instance)) == 0
try:
    count = C.c_uint32()
    assert vk.vkEnumeratePhysicalDevices(instance, C.byref(count), None) == 0
    devices = (C.c_void_p * count.value)()
    assert vk.vkEnumeratePhysicalDevices(instance, C.byref(count), devices) == 0
    result = []
    for index, device in enumerate(devices):
        buffer = C.create_string_buffer(4096)
        vk.vkGetPhysicalDeviceProperties(device, buffer)
        api, driver, vendor, device_id, typ = struct.unpack_from('5I', buffer.raw)
        result.append(dict(vulkan_index=index, name=buffer.raw[20:276].split(b'\0')[0].decode(),
                           api_version=f'{api >> 22}.{(api >> 12) & 1023}.{api & 4095}',
                           driver_raw=driver, vendor_id=vendor, device_id=device_id, device_type=typ))
finally:
    vk.vkDestroyInstance(instance, None)
out = Path(__file__).with_name('vulkan-devices.json')
out.write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2))
