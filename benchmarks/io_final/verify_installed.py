"""Installed Python3.8+ proof of exact I/O arithmetic and strict PNG profiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--expect-numba',choices=['yes','no'],required=True)
    args=parser.parse_args()
    import numpy as np
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ImageData
    from ColorCorrectionPipeline import _image_numeric as numeric
    from ColorCorrectionPipeline.io import read_image_data,write_image_data
    expect=args.expect_numba=='yes'
    assert sys.prefix!=sys.base_prefix and Path(sys.prefix).resolve() in Path(package.__file__).resolve().parents
    assert numeric.accel.HAS_NUMBA is expect
    args.output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix=args.output.stem+'-',dir=args.output.parent))
    def module(name,filename):
        spec=importlib.util.spec_from_file_location(name,args.fixtures/filename)
        result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
        return result
    frozen=module('ColorCorrectionPipeline._installed_io_contract_reference','image_contract_p9.py')
    writer=module('ColorCorrectionPipeline.io._installed_io_writer_reference','writers_p9_reference.py')
    source=np.resize(np.arange(65536,dtype=np.uint16),513*683*3).reshape(513,683,3)[::-1,::-1]
    events=[]
    if expect:
        norm,quant=numeric._normalize_unsigned,numeric._quantize_unsigned
        def trace_norm(*args):
            events.append(('normalization',args[0].size));return norm(*args)
        def trace_quant(*args):
            events.append(('quantization',args[0].size));return quant(*args)
        numeric._normalize_unsigned,numeric._quantize_unsigned=trace_norm,trace_quant
    expected=frozen.ImageData.from_array(source,channel_order='BGR',orientation=7)
    actual=ImageData.from_array(source,channel_order='BGR',orientation=7)
    assert actual.to_metadata()==expected.to_metadata()
    assert actual.pixels.tobytes()==expected.pixels.tobytes()
    assert not actual.pixels.flags.writeable
    cases=[]
    for dtype in ('uint8','uint16'):
        old=writer.write_image_data(work/('old-'+dtype+'.png'),actual,dtype=dtype,allow_precision_loss=True)
        compatible=write_image_data(work/('compatible-'+dtype+'.png'),actual,dtype=dtype,allow_precision_loss=True)
        fast=write_image_data(work/('fast-'+dtype+'.png'),actual,dtype=dtype,allow_precision_loss=True,png_profile='fast-lossless')
        def facts(record,exclude):
            result=record.to_dict()
            for key in exclude:result.pop(key)
            return result
        assert facts(old,('path',))==facts(compatible,('path',))
        assert facts(fast,('path','bytes','sha256'))==facts(compatible,('path','bytes','sha256'))
        decoded_old=read_image_data(Path(compatible.path))
        decoded_fast=read_image_data(Path(fast.path))
        assert decoded_old.pixels.tobytes()==decoded_fast.pixels.tobytes()
        assert decoded_old.pixel_digest==decoded_fast.pixel_digest
        cases.append({'dtype':dtype,'default_bytes_and_all_facts_exact':True,'fast_pixels_and_source_facts_exact':True,
            'compatible_bytes':compatible.bytes,'fast_bytes':fast.bytes})
    parallel_verified=all((name,source.size) in events for name in ('normalization','quantization')) if expect else False
    assert parallel_verified is expect
    def worker(_):
        captured=ImageData.from_array(source)
        from ColorCorrectionPipeline.io.writers import _quantize_strict
        storage,error=_quantize_strict(captured.pixels,np.dtype('uint16'),65535)
        assert storage.tobytes()==source.tobytes() and error==0.
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))
    if expect:
        def fail_after_qualification(values,storage,maximum):
            if values.size==source.size:
                storage[:values.size//2]=0
                raise RuntimeError('installed partial quantizer failure')
            return quant(values,storage,maximum)
        numeric._quantize_unsigned=fail_after_qualification
        recovered=write_image_data(work/'recovered.png',actual,dtype='uint16',allow_precision_loss=True)
        assert recovered.sha256==compatible.sha256
    report={'python':platform.python_version(),'package':str(package.__file__),'numba':expect,
        'parallel_kernels_executed':parallel_verified,'worker_serial_route_checked':True,
        'partial_failure_recovered':expect,'capture_bytes_and_metadata_exact':True,'cases':cases,'all_checks_passed':True}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
