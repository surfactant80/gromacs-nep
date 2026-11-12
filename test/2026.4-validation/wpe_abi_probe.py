#!/usr/bin/env python3
"""Probe libwisevolve_potential.so without a GPU (public C ABI only).

Usage: python3 wpe_abi_probe.py /path/to/wpe.so/lib/libwisevolve_potential.so

Checks wpe_get_info(), the ABI record sizes and the reported engine version.
If the library cannot be loaded, prints the loader error (for example a
missing libcufft.so.11), which is exactly what GROMACS reports at mdrun.
"""
import ctypes
import sys


class WpeInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("provider", ctypes.c_char_p),
        ("library_soname", ctypes.c_char_p),
        ("project_url", ctypes.c_char_p),
        ("citation", ctypes.c_char_p),
        ("citation_doi", ctypes.c_char_p),
    ]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        library = ctypes.CDLL(sys.argv[1])
    except OSError as error:
        print(f"load failed: {error}")
        return 1
    library.wpe_get_info.restype = ctypes.c_int
    library.wpe_get_info.argtypes = [ctypes.POINTER(WpeInfo)]
    info = WpeInfo()
    info.struct_size = ctypes.sizeof(info)
    status = library.wpe_get_info(ctypes.byref(info))
    print(f"status={status} struct_size={info.struct_size} sizeof={ctypes.sizeof(info)}")
    for field in ("name", "version", "provider", "library_soname", "citation", "citation_doi"):
        value = getattr(info, field)
        print(f"{field}: {value.decode() if value else value}")
    return 0 if status == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
