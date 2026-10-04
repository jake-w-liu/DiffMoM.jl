# ODB++ UNIX-compress transport fixture

`product.tar.Z` was generated with Windows bsdtar/libarchive 3.8.8 using
`tar.exe --format=ustar -cZf`. Its product contains one round copper pad and
deterministic metadata comments large enough to exercise LZW width changes
and dictionary saturation.

`reference.toml` records the producer, input seed, byte lengths and SHA-256
of both the compressed fixture and the separately generated uncompressed
ustar archive. Tests compare the decoder's complete output to that independent
raw archive hash, then check the imported pad geometry.

Reproducer: `validation/planar_audit/odb_compression_fixture.jl`.
The fixture contains generated test data and no proprietary manufacturing
files. It validates transport; it is not a vendor production corpus.
