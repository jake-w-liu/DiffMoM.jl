# Chapter 3: Bempp Cross-Validation

## Purpose

Establish external solver consistency by comparing `DiffMoM.jl` results against an independent boundary element method implementation (`Bempp-cl`). External validation provides confidence that numerical results are physically meaningful and not artifacts of a single implementation. For beam-steering applications, we prioritize **beam-centric metrics** that directly impact design utility rather than global RMS errors dominated by deep null regions.

---

## Learning Goals

After this chapter, you should be able to:

1. Run the complete PEC and impedance-loaded cross-validation workflows.
2. Interpret beam-centric metrics: main-beam angle, main-beam level, sidelobe position and suppression.
3. Understand the seven-case impedance validation matrix from the paper.
4. Diagnose convention mismatches between solvers (time sign, normalization, scaling).
5. Generate comprehensive validation reports for publication or verification.

---

## 1) Philosophy of External Validation

### 1.1 Why External Cross-Validation Matters

Internal consistency checks verify mathematical correctness within a single codebase, but external validation ensures:

- **Physical correctness**: Different implementations should agree on physically observable quantities
- **Implementation independence**: Results shouldn't depend on solver-specific discretization choices
- **Robustness**: Confidence that optimization results are not code-specific artifacts

### 1.2 Beam-Centric vs. Global Metrics

For beam-steering applications, global RMS errors can be misleading because:

- Deep null regions (≈ -40 dB) dominate RMS but are irrelevant for beam performance
- Small phase shifts cause large dB differences in nulls but minimal beam impact
- Design utility depends on **main-beam direction and gain**, not null accuracy

Thus, we compare on **aligned cuts** (typically $\phi = 0^\circ$) using:

1. **Main-beam angle difference**: $|\Delta\theta_{\text{main}}|$ (degrees)
2. **Main-beam level difference**: $|\Delta D_{\text{main}}|$ (dB)  
3. **Strongest sidelobe angle difference**: $|\Delta\theta_{\text{SL}}|$ (degrees)
4. **Sidelobe suppression difference**: $|\Delta\text{SLL}|$ (dB)

---

## 2) Implementation and Workflows

### 2.1 Python Validation Suite (`validation/bempp/`)

The validation suite provides a complete pipeline:

#### PEC Cross-Validation
```bash
# Generate Bempp PEC reference
python validation/bempp/run_pec_cross_validation.py

# Compare against Julia PEC baseline
python validation/bempp/compare_pec_to_julia.py
```

#### Single Impedance Case
```bash
# Generate Julia reference
julia --project=. validation/bempp/run_impedance_case_julia_reference.jl \
  --freq-ghz 3.0 --theta-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0

# Generate Bempp results
python validation/bempp/run_impedance_cross_validation.py \
  --freq-ghz 3.0 --zs-imag-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0

# Compare results
python validation/bempp/compare_impedance_to_julia.py \
  --output-prefix impedance --target-theta-deg 30
```

#### Seven-Case Validation Matrix
```bash
# Run complete matrix (paper's seven cases)
python validation/bempp/run_impedance_validation_matrix.py
```

### 2.2 Matching Conditions

For meaningful comparison, ensure exact matching:

- **Frequency**: Same GHz value (3.0 GHz default)
- **Incidence**: Same $\theta_{\text{inc}}$, $\phi_{\text{inc}}$, polarization
- **Geometry**: Same plate dimensions, discretization (structured 12×12 mesh recommended)
- **Angular grid**: Same $(\theta, \phi)$ sampling (180×72 default)
- **Impedance values**: Same $Z_s = i\theta$ (purely reactive)

### 2.3 Output Artifacts

Generated files in `data/`:

- `bempp_cross_validation_report.json`: Complete metrics in machine-readable format
- `bempp_cross_validation_report.md`: Human-readable summary
- `impedance_validation_matrix_summary.csv`: Seven-case results (CSV)
- `impedance_validation_matrix_summary.md`: Matrix analysis report
- Diagnostic plots and current/phase comparisons for detailed investigation

---

## 3) Results from the Paper

### 3.1 Seven-Case Impedance Matrix

The paper (`bare_jrnl.tex`, Section IV) reports a seven-case validation matrix:

| $Z_s$ (Ω) | $f$ (GHz) | $\theta_{\text{inc}}$ (deg) | $|\Delta\theta_{\text{main}}|$ | $|\Delta D_{\text{main}}|$ | $|\Delta\theta_{\text{SL}}|$ | $|\Delta\text{SLL}|$ |
|-----------|-----------|-----------------------------|--------------------------------|-----------------------------|------------------------------|-----------------------|
| 0 (PEC)   | 3.00      | 0.0                         | 0.0°                           | 0.052 dB                    | 0.0°                         | 0.001 dB             |
| 25        | 3.00      | 0.0                         | 0.0°                           | 0.376 dB                    | 1.0°                         | 0.253 dB             |
| 50        | 3.00      | 0.0                         | 0.0°                           | 0.081 dB                    | 0.0°                         | 0.074 dB             |
| 75        | 3.00      | 0.0                         | 0.0°                           | 0.207 dB                    | 1.0°                         | 0.103 dB             |
| 100       | 3.00      | 0.0                         | 0.0°                           | 0.275 dB                    | 1.0°                         | 0.116 dB             |
| 100       | 3.00      | 5.0                         | 0.0°                           | 0.241 dB                    | 1.0°                         | 0.469 dB             |
| 100       | 3.06      | 0.0                         | 0.0°                           | 0.275 dB                    | 1.0°                         | 0.114 dB             |

**Key findings:**
- **Perfect main-beam angle alignment**: $|\Delta\theta_{\text{main}}| = 0^\circ$ for all cases
- **Small main-beam level differences**: $|\Delta D_{\text{main}}| \leq 0.376$ dB
- **Minor sidelobe angle shifts**: $|\Delta\theta_{\text{SL}}| \leq 1^\circ$
- **Modest sidelobe suppression differences**: $|\Delta\text{SLL}| \leq 0.469$ dB

### 3.2 PEC Reference Agreement

The PEC case shows near-perfect agreement:
- Main-beam angle: 0° difference
- Main-beam level: 0.052 dB difference  
- Sidelobe suppression: 0.001 dB difference

This establishes a baseline for impedance-loaded comparisons.

### 3.3 Interpretation of Discrepancies

The observed differences (≤ 0.5 dB, ≤ 1°) are consistent with:

- **Different quadrature rules**: Bempp uses adaptive integration vs. Julia's singularity-extracted semi-analytical
- **Basis function differences**: RWG implementation details vary
- **Impedance weak form**: Different treatment of surface impedance term
- **Mesh discretization**: Slight variations in triangle geometry

These differences are acceptable for design purposes where beam-steering utility depends on relative rather than absolute accuracy.

---

## 4) Practical Workflow Examples

### 4.1 Complete Validation Session

```bash
# 1. Setup environment
pip install -r validation/bempp/requirements.txt

# 2. Run PEC validation (establishes baseline)
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py

# 3. Run single impedance case (debugging)
julia --project=. validation/bempp/run_impedance_case_julia_reference.jl \
  --freq-ghz 3.0 --theta-ohm 100 --output-prefix test_z100

python validation/bempp/run_impedance_cross_validation.py \
  --freq-ghz 3.0 --zs-imag-ohm 100 --output-prefix test_z100 \
  --mesh-mode structured --nx 12 --ny 12

python validation/bempp/compare_impedance_to_julia.py \
  --output-prefix test_z100 --target-theta-deg 30

# 4. Generate diagnostic plots
python validation/bempp/plot_impedance_comparison.py \
  --julia-prefix test_z100 --bempp-prefix test_z100 \
  --output-prefix test_z100_diag
```

### 4.2 Convention Sweep for Reconciliation

If discrepancies exceed expectations, investigate convention mismatches:

```bash
python validation/bempp/sweep_impedance_conventions.py --run-julia
```

Sweeps possible convention variants:
- Time sign: $e^{-i\omega t}$ vs. $e^{+i\omega t}$
- RHS scaling: $\mathbf{v}$ sign and magnitude
- Impedance scaling: $Z_s$ multiplicative factor
- Phase convention: Far-field phase reference

### 4.3 Operator-Aligned Benchmark

For deep investigation, compare currents and phases at element centers:

```bash
python validation/bempp/run_impedance_operator_aligned_benchmark.py \
  --output-prefix opalign_z100 \
  --freq-ghz 3.0 --zs-imag-ohm 100 \
  --mesh-mode structured --nx 12 --ny 12
```

This benchmark compares:
- Far-field beam metrics
- Element-center complex currents
- Phase coherence and residuals
- Forward-solve linear system accuracy

---

## 5) Interpreting and Diagnosing Results

### 5.1 Acceptance Criteria

For beam-steering applications, consider:

- **Primary success**: $|\Delta\theta_{\text{main}}| < 1^\circ$, $|\Delta D_{\text{main}}| < 0.5$ dB
- **Secondary acceptance**: $|\Delta\theta_{\text{SL}}| < 2^\circ$, $|\Delta\text{SLL}| < 1.0$ dB
- **Investigation needed**: Any metric exceeds 2× paper values

### 5.2 Common Issues and Solutions

1. **Large main-beam angle shift ($> 2^\circ$)**
   - Likely cause: Geometry scaling mismatch (meters vs. wavelengths)
   - Check: Plate dimensions, frequency units, mesh generation

2. **Large main-beam level difference ($> 1$ dB)**
   - Likely cause: Excitation normalization or polarization mismatch
   - Check: Incident field magnitude, polarization vector, far-field scaling

3. **Systematic offset across all cases**
   - Likely cause: Time sign convention ($e^{\pm i\omega t}$)
   - Check: Phase sign in far-field computation, Bempp convention profile

4. **Impedance-specific discrepancies**
   - Likely cause: Different weak-form implementation of surface impedance
   - Check: Bempp's `impedance_boundary_condition` parameters, Julia's $Z_s$ scaling

### 5.3 When to Trust Results

External validation provides confidence when:

1. **PEC agreement is excellent** ($|\Delta D_{\text{main}}| < 0.1$ dB)
2. **Impedance trends are consistent** (monotonic with $Z_s$)
3. **Beam-centric metrics align** with design requirements
4. **Multiple independent checks agree** (energy, reciprocity, gradients)

---

## 6) Advanced Topics

### 6.1 Extending the Validation Matrix

The seven-case matrix can be extended for more comprehensive validation:

```bash
# Custom matrix with varied parameters
python validation/bempp/run_impedance_validation_matrix.py \
  --zs-values 0 50 100 150 200 \
  --freq-values 2.9 3.0 3.1 \
  --theta-inc-values 0 10 20 \
  --mesh-mode structured --nx 12 --ny 12
```

### 6.2 Automated Regression Testing

Incorporate validation into CI pipelines:

```python
# Example test script
import subprocess
import json

# Run validation
subprocess.run(["python", "validation/bempp/run_impedance_validation_matrix.py", 
                "--mesh-mode", "structured", "--nx", "6", "--ny", "6"])

# Load and check results
with open("data/impedance_validation_matrix_summary.json") as f:
    data = json.load(f)
    
for case in data["cases"]:
    assert case["delta_theta_main_deg"] < 1.0, f"Main-beam angle mismatch: {case}"
    assert case["delta_D_main_db"] < 0.5, f"Main-beam level mismatch: {case}"
```

### 6.3 Publication-Ready Reports

Generate comprehensive validation reports:

```bash
# Generate all artifacts
python validation/bempp/run_impedance_validation_matrix.py \
  --convention-profile paper_default \
  --output-tag v1.0

# Combined report includes:
# - Summary table (LaTeX format)
# - Beam-centric metrics
# - Diagnostic plots
# - Convention reconciliation analysis
```

---

## 7) Code Mapping

### 7.1 Primary Scripts

- **PEC validation**: `validation/bempp/run_pec_cross_validation.py`, `compare_pec_to_julia.py`
- **Impedance validation**: `validation/bempp/run_impedance_cross_validation.py`, `compare_impedance_to_julia.py`
- **Matrix validation**: `validation/bempp/run_impedance_validation_matrix.py`
- **Convention analysis**: `validation/bempp/sweep_impedance_conventions.py`
- **Diagnostic plots**: `validation/bempp/plot_impedance_comparison.py`
- **Operator-aligned benchmark**: `validation/bempp/run_impedance_operator_aligned_benchmark.py`

### 7.2 Julia Reference Generation

- **PEC reference**: `validation/paper/run_beam_steering_case.jl`
- **Impedance reference**: `validation/bempp/run_impedance_case_julia_reference.jl`
- **Paper aggregation**: `validation/paper/generate_consistency_report.jl`

### 7.3 Output Artifacts

All outputs in `data/` directory with consistent naming:
- `bempp_*_farfield.csv`: Bempp far-field patterns
- `julia_*_farfield.csv`: Julia far-field patterns  
- `*_cross_validation_report.*`: Summary reports (JSON, MD)
- `*_validation_matrix_summary.*`: Matrix results (CSV, JSON, MD)

---

## 8) Exercises

### 8.1 Basic Level

1. **Run PEC cross-validation**:
   - Execute `run_pec_cross_validation.py` and `compare_pec_to_julia.py`
   - Verify main-beam agreement: $|\Delta\theta_{\text{main}}| < 0.1^\circ$, $|\Delta D_{\text{main}}| < 0.1$ dB
   - Inspect generated report files

2. **Single impedance case**:
   - Run validation for $Z_s = i100\,\Omega$ at 3.0 GHz
   - Compare results to paper values (row 5 of seven-case matrix)
   - Generate diagnostic plots and interpret differences

### 8.2 Intermediate Level

3. **Seven-case matrix reproduction**:
   - Run the complete validation matrix
   - Compare your results to Table 2 in the paper
   - Identify any discrepancies > 0.1 dB and propose explanations

4. **Convention sweep analysis**:
   - Run `sweep_impedance_conventions.py` for a single case
   - Identify which convention variant gives best agreement
   - Document the impact of each convention parameter

### 8.3 Advanced Level

5. **Extended validation matrix**:
   - Design and run a 12-case matrix with varied parameters
   - Analyze trends: How do discrepancies scale with $|Z_s|$, frequency, incidence angle?
   - Propose acceptance thresholds for each metric

6. **Operator-aligned investigation**:
   - Run the operator-aligned benchmark for a problematic case
   - Compare element-center currents and phases
   - Identify whether discrepancies originate from far-field computation or current solution

---

## 9) Chapter Checklist

Before considering external validation complete, ensure you can:

- [ ] Run PEC cross-validation with excellent agreement ($|\Delta D_{\text{main}}| < 0.1$ dB)
- [ ] Reproduce the seven-case matrix results within 0.1 dB of paper values
- [ ] Interpret beam-centric metrics and distinguish important from negligible differences
- [ ] Diagnose and reconcile convention mismatches
- [ ] Generate comprehensive validation reports
- [ ] Understand limitations of cross-solver comparisons for impedance-loaded structures

---

## 10) Further Reading

- **Bempp-cl documentation**: Betcke & Scroggs, *Bempp-cl: A fast Python-based boundary element library* (2021)
- **Cross-solver validation in CEM**: Peterson et al., *Comparison of CEM codes for canonical problems* (2005)
- **Beam-centric metrics for antenna validation**: Balanis, *Antenna Theory: Analysis and Design* (2016)
- **Uncertainty quantification in simulation**: Roache, *Verification and Validation in Computational Science and Engineering* (1998)
- **Paper reference**: `bare_jrnl.tex`, Section IV (seven-case Bempp-cl matrix, Table 2)
