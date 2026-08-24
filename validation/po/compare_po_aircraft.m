% compare_po_aircraft.m — Compare Julia PO with POFacets 4.5 on aircraft
%
% Produces a diagnostic comparison at 0.3 GHz. The two inputs must represent
% the same geometry and units before their RCS curves are comparable.

clear; close all; clc;

% Resolve external inputs without embedding machine-specific paths.
pof_dir = getenv('POFACETS_DIR');
if isempty(pof_dir)
    error(['POFACETS_DIR is not set. Set it to the POFacets 4.5 directory ' ...
           'and rerun this script.']);
end
if exist(pof_dir, 'dir') ~= 7
    error('POFACETS_DIR does not name a directory: %s', pof_dir);
end
addpath(pof_dir);
if exist('facetRCS', 'file') ~= 2
    error(['facetRCS.m was not found after adding POFACETS_DIR=%s. ' ...
           'Point POFACETS_DIR at the POFacets 4.5 directory and rerun.'], ...
          pof_dir);
end

airplane_mat = getenv('POFACETS_AIRCRAFT_MAT');
if isempty(airplane_mat)
    airplane_mat = fullfile(pof_dir, 'CAD Library Pofacets', 'airplane.mat');
end
if exist(airplane_mat, 'file') ~= 2
    error(['POFacets aircraft geometry not found at %s. Set ' ...
           'POFACETS_AIRCRAFT_MAT to the required airplane.mat path and rerun.'], ...
          airplane_mat);
end
S = load(airplane_mat);
if ~isfield(S, 'coord') || ~isfield(S, 'facet')
    error(['POFacets geometry %s must contain coord and facet arrays. ' ...
           'Select a compatible airplane.mat file.'], airplane_mat);
end
coord = double(S.coord);
facet = double(S.facet);
if size(coord, 2) < 3 || size(facet, 2) < 5 || isempty(coord) || isempty(facet)
    error(['POFacets geometry %s must contain a nonempty coord array with at ' ...
           'least three columns and a facet array with at least five columns.'], ...
          airplane_mat);
end
if any(~isfinite(coord(:, 1:3)), 'all') || any(~isfinite(facet(:, 1:5)), 'all')
    error('POFacets geometry %s contains NaN or Inf values.', airplane_mat);
end
facet_indices = facet(:, 1:3);
if any(facet_indices ~= fix(facet_indices), 'all') || ...
        any(facet_indices < 1, 'all') || ...
        any(facet_indices > size(coord, 1), 'all')
    error('POFacets facet indices in %s must be integers within coord.', airplane_mat);
end

out_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
julia_phi0 = fullfile(out_dir, 'julia_po_aircraft_0p3_phi0.csv');
julia_phi90 = fullfile(out_dir, 'julia_po_aircraft_0p3_phi90.csv');
if ~isfile(julia_phi0) || ~isfile(julia_phi90)
    error(['Julia PO inputs are incomplete. Expected %s and %s. Run ' ...
           'validation/po/generate_julia_po_aircraft.jl with the matching ' ...
           'DMOM_AIRCRAFT_OBJ, then rerun this comparison.'], ...
          julia_phi0, julia_phi90);
end

fprintf('========================================\n');
fprintf('PO Comparison: POFacets vs DiffMoM data\n');
fprintf('========================================\n');
fprintf('POFacets directory: %s\n', pof_dir);
fprintf('POFacets geometry: %s\n', airplane_mat);
fprintf('Airplane.mat: %d vertices, %d facets\n', size(coord,1), size(facet,1));
fprintf(['Interpret the curve comparison only if the Julia run used the same ' ...
         'geometry and metre units.\n']);

% Parameters matching Julia examples
freq_ghz = 0.3;
C0 = 3e8;
wave = C0 / (freq_ghz * 1e9);
bk = 2*pi / wave;
eta0 = 376.730313668;

% Incidence: wave propagating in -z (from +z toward -z)
% In POFacets convention, θ_i specifies the SOURCE direction, NOT propagation.
% Source at +z → θ_i=0°.  k̂_prop = -D0i = [0,0,-1] → matches Julia k_vec=[0,0,-k].
% θ-pol at θ_i=0° gives e0=[1,0,0] = x-pol, matching Julia pol=[1,0,0].
itheta_deg = 0.0;
iphi_deg = 0.0;
i_pol = 1;  % θ-polarized → e0 = [1, 0, 0] at θ_i=0°

% Observation grid: 1° resolution at φ=0° and φ=90°
Ntheta = 180;
phi_cuts = [0.0, 90.0];
Nphi = length(phi_cuts);

% POFacets parameters
rsmethod = 1;  % use Rs (PEC: Rs=0)
iflag = 0;     % enable illumination test
Lt = 1e-5;
Nt = 5;
corr = 0.0;
stdv = 0.0;

corel = corr / wave;
delsq = stdv^2;
cfac1 = exp(-4*bk^2*delsq);
cfac2 = 4*pi*(bk*corel)^2*delsq;

% Build facet normals and areas (same as run_pofacets_bistatic_batch.m)
node1 = facet(:,1);
node2 = facet(:,2);
node3 = facet(:,3);
ilum = facet(:,4);
Rs = facet(:,5);

ntria = size(facet,1);
vind = [node1 node2 node3];

x = coord(:,1);
y = coord(:,2);
z = coord(:,3);
r = [x y z];

N = zeros(ntria,3);
Area = zeros(ntria,1);
alpha = zeros(ntria,1);
beta = zeros(ntria,1);

for i = 1:ntria
    A = r(vind(i,2),:) - r(vind(i,1),:);
    B = r(vind(i,3),:) - r(vind(i,2),:);
    C = r(vind(i,1),:) - r(vind(i,3),:);
    nvec = -cross(B,A);
    d1 = norm(A); d2 = norm(B); d3 = norm(C);
    ss = 0.5 * (d1 + d2 + d3);
    Area(i) = sqrt(max(ss*(ss-d1)*(ss-d2)*(ss-d3), 0));
    Nn = norm(nvec);
    if Nn == 0
        N(i,:) = [0 0 1];
    else
        N(i,:) = nvec / Nn;
    end
    beta(i) = acos(max(min(N(i,3),1),-1));
    alpha(i) = atan2(N(i,2),N(i,1));
end

% Incidence direction and polarization
rad = pi/180;
ithetar = itheta_deg * rad;
iphir = iphi_deg * rad;

if i_pol == 1
    Et = 1 + 1i*0;
    Ep = 0 + 1i*0;
else
    Et = 0 + 1i*0;
    Ep = 1 + 1i*0;
end

cpi = cos(iphir); spi = sin(iphir);
sti = sin(ithetar); cti = cos(ithetar);
uui = cti*cpi; vvi = cti*spi; wwi = -sti;
ui = sti*cpi; vi = sti*spi; wi = cti;

e0 = [uui*Et - spi*Ep, vvi*Et + cpi*Ep, wwi*Et];

% Illumination test
Ri = [ui vi wi];
illuminated = zeros(ntria,1);
for m = 1:ntria
    nidotk = N(m,:) * transpose(Ri);
    if (ilum(m) == 1 && nidotk >= 0) || ilum(m) == 0
        illuminated(m) = 1;
    end
end

fprintf('\nIllumination:\n');
fprintf('  Total facets: %d\n', ntria);
fprintf('  Illuminated: %d (%.1f%%)\n', sum(illuminated), 100*sum(illuminated)/ntria);

% Compute RCS at observation grid
theta_vec = ((1:Ntheta) - 0.5) * pi / Ntheta;

results = struct();

for ip = 1:Nphi
    phi_deg = phi_cuts(ip);
    phr = phi_deg * rad;

    fprintf('\n=== φ = %.0f° cut ===\n', phi_deg);

    sigma_tot = zeros(Ntheta, 1);
    theta_deg_out = zeros(Ntheta, 1);

    for it = 1:Ntheta
        thr = theta_vec(it);

        sumt = 0 + 1i*0;
        sump = 0 + 1i*0;
        sumdt = 0;
        sumdp = 0;
        RCpar = 0;
        RCperp = 0;

        for m = 1:ntria
            [Ets, Etd, Eps, Epd] = facetRCS(thr, phr, ithetar, iphir, ...
                N(m,:), ilum(m), iflag, alpha(m), beta(m), Rs(m), Area(m), ...
                x, y, z, vind(m,:), e0, Nt, Lt, cfac2, corel, wave, ...
                0, 0, 0, rsmethod, RCpar, RCperp);
            sumt = sumt + Ets;
            sump = sump + Eps;
            sumdt = sumdt + abs(Etd);
            sumdp = sumdp + abs(Epd);
        end

        sig_t = 4*pi * (abs(sumt)^2) / wave^2;
        sig_p = 4*pi * (abs(sump)^2) / wave^2;
        sig_tot = sig_t + sig_p;

        theta_deg_out(it) = thr / rad;
        sigma_tot(it) = sig_tot;
    end

    % Store results
    field_name = sprintf('phi%d', round(phi_deg));
    results.(field_name).theta_deg = theta_deg_out;
    results.(field_name).sigma_m2 = sigma_tot;
    results.(field_name).sigma_dBsm = 10*log10(max(sigma_tot, 1e-30));

    % Backscatter (θ=0°, opposite to propagation -z → scatter toward +z)
    [~, bs_idx] = min(abs(theta_deg_out - 0.0));
    bs_sigma = sigma_tot(bs_idx);
    bs_dB = 10*log10(max(bs_sigma, 1e-30));

    fprintf('  Backscatter RCS: %.2f dBsm (θ=%.1f°)\n', bs_dB, theta_deg_out(bs_idx));
    fprintf('  Peak RCS: %.2f dBsm\n', max(results.(field_name).sigma_dBsm));
end

% Save results
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Save phi=0 cut
T0 = table(results.phi0.theta_deg, results.phi0.sigma_m2, results.phi0.sigma_dBsm, ...
    'VariableNames', {'theta_deg', 'sigma_m2', 'sigma_dBsm'});
out_file_0 = fullfile(out_dir, 'pofacets45_aircraft_0p3_phi0.csv');
writetable(T0, out_file_0);
fprintf('\nSaved: %s\n', out_file_0);

% Save phi=90 cut
T90 = table(results.phi90.theta_deg, results.phi90.sigma_m2, results.phi90.sigma_dBsm, ...
    'VariableNames', {'theta_deg', 'sigma_m2', 'sigma_dBsm'});
out_file_90 = fullfile(out_dir, 'pofacets45_aircraft_0p3_phi90.csv');
writetable(T90, out_file_90);
fprintf('Saved: %s\n', out_file_90);

% Plot diagnostic comparison with the preflighted Julia results.
figure('Position', [100, 100, 1200, 500]);

% φ=0° cut
subplot(1,2,1);
plot(results.phi0.theta_deg, results.phi0.sigma_dBsm, 'b-', 'LineWidth', 2, 'DisplayName', 'POFacets 4.5');
hold on;
J0 = readtable(julia_phi0);
required_columns = {'theta_deg', 'sigma_dBsm'};
if ~all(ismember(required_columns, J0.Properties.VariableNames)) || ...
        isempty(J0) || any(~isfinite(J0.theta_deg)) || any(~isfinite(J0.sigma_dBsm))
    error('Julia PO input %s must contain finite theta_deg and sigma_dBsm columns.', ...
          julia_phi0);
end
plot(J0.theta_deg, J0.sigma_dBsm, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Julia PO');
sigma_pof_interp = interp1(results.phi0.theta_deg, results.phi0.sigma_dBsm, ...
                           J0.theta_deg, 'linear', 'extrap');
rmse_phi0 = sqrt(mean((J0.sigma_dBsm - sigma_pof_interp).^2));
if ~isfinite(rmse_phi0)
    error('φ=0° diagnostic RMSE is not finite; inspect %s.', julia_phi0);
end
fprintf('\nφ=0° diagnostic RMSE: %.3f dB\n', rmse_phi0);
legend('Location', 'best');
grid on;
xlabel('θ (deg)');
ylabel('Bistatic RCS (dBsm)');
title(sprintf('Aircraft φ=0° — %.1f GHz', freq_ghz));
xlim([0 180]);

% φ=90° cut
subplot(1,2,2);
plot(results.phi90.theta_deg, results.phi90.sigma_dBsm, 'b-', 'LineWidth', 2, 'DisplayName', 'POFacets 4.5');
hold on;
J90 = readtable(julia_phi90);
if ~all(ismember(required_columns, J90.Properties.VariableNames)) || ...
        isempty(J90) || any(~isfinite(J90.theta_deg)) || any(~isfinite(J90.sigma_dBsm))
    error('Julia PO input %s must contain finite theta_deg and sigma_dBsm columns.', ...
          julia_phi90);
end
plot(J90.theta_deg, J90.sigma_dBsm, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Julia PO');
sigma_pof_interp = interp1(results.phi90.theta_deg, results.phi90.sigma_dBsm, ...
                           J90.theta_deg, 'linear', 'extrap');
rmse_phi90 = sqrt(mean((J90.sigma_dBsm - sigma_pof_interp).^2));
if ~isfinite(rmse_phi90)
    error('φ=90° diagnostic RMSE is not finite; inspect %s.', julia_phi90);
end
fprintf('φ=90° diagnostic RMSE: %.3f dB\n', rmse_phi90);
legend('Location', 'best');
grid on;
xlabel('θ (deg)');
ylabel('Bistatic RCS (dBsm)');
title(sprintf('Aircraft φ=90° — %.1f GHz', freq_ghz));
xlim([0 180]);

sgtitle('POFacets 4.5 vs Julia PO — Aircraft 0.3 GHz');

fig_file = fullfile(out_dir, 'po_comparison_aircraft.png');
saveas(gcf, fig_file);
fprintf('\nSaved plot: %s\n', fig_file);

fprintf('\n========================================\n');
fprintf('PO diagnostic comparison complete. Outputs: %s\n', out_dir);
fprintf('No acceptance threshold is applied; verify that both runs used the same geometry.\n');
fprintf('========================================\n');
