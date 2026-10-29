import os, json, sys, time
import numpy as np
import taichi as ti
ti.init(arch=ti.vulkan, default_fp=ti.f64, enable_fallback=False, offline_cache=False)
@ti.kernel
def calc(a: ti.types.ndarray(), b: ti.types.ndarray()):
    for i in range(a.shape[0]):
        b[i] = a[i] ** 2.4
a = np.linspace(0.001,1,10000,dtype=np.float64)
b = np.zeros_like(a)
calc(a,b)
ti.sync()
print(json.dumps({'arch':str(ti.lang.impl.current_cfg().arch), 'maxerror':float(np.max(abs(b-a**2.4))), 'first':b[:5].tolist()}))
