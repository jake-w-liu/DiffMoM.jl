# Bulk conductivity component conversion review

Mixed high-precision conductivity was accepted after a nonzero real or imaginary component rounded to zero in ComplexF64. That could remove a finite, representable resistivity component: for example, 1e-400 + j1e-300 has real reciprocal about 1e200, but the former conversion returned zero for that component.

Both via_sigma and volume_sigma now reject nonzero conductivity components that become zero in stored arithmetic. Existing finite/passive validation, the +Inf PEC sentinel, complex reciprocal and all physical Gram reactions remain unchanged. Errors retain the conductor label and level.

Independent 160-digit decimal calculations verify all 12 scalar/vector via/volume BigFloat reproductions on Julia 1.13.1 and 1.12.7. Both versions pass 66 new BigFloat/Rational and dense/dense-FFT/UFFT rejection checks, plus the original dissipation, physical residual/power, gradient and volume-resistance regressions. Five warmed samples on four healthy paths preserve result bits and allocation counts.

The isolated prototype is independently reviewed against all 141 package inputs. The registered test file retains its original byte prefix. Committed-source full-package and hosted CI verification are separate requirements. This change does not establish native volume skin-profile agreement or complete Sonnet/RFIC parity.
