# Darwin available-memory repair

The actual source16 macOS CI job 113109407629 rejected the original shared
endpoint fixture: 213273520 required raw bytes exceeded the remaining default
budget 175674336. This was a test failure, unrelated to billing.

The Julia-bundled libuv Darwin query returns only free pages. The current
libuv availability implementation also includes inactive and purgeable pages.
The default now uses that public Mach calculation, the reported OS page size,
and the physical-memory bound. Apple fixes the ABI layout and HOST_VM_INFO
flavor; the count is derived from the Julia representation. The host send
right is released after querying. Query failure retains the conservative
Sys.free_memory fallback. Explicit caller limits and all original scientific,
resource, native-fixture and CI gates remain unchanged.

Sources: https://github.com/libuv/libuv/blob/v1.x/src/unix/darwin.c and
https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h
and https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/host_info.h.

The recorded line ceiling must equal the pinned counter's measurement, with
no additional allowance. Actual macOS hosted behavior remains required.
