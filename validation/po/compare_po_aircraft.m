% compare_po_aircraft.m — Compare Julia PO with POFacets 4.5 on aircraft
%
% Validates DiffMoM.jl PO implementation against POFacets 4.5
% using airplane.mat geometry at 0.3 GHz (14λ wingspan)

clear; close all; clc;

% Add POFacets to path
pof_dir = '/Users/jake/Library/CloudStorage/OneDrive-Personal/tests/testutdp/pofacets4.5';
addpath(pof_dir);

% Load airplane.mat
airplane_mat = fullfile(pof_dir, 'CAD Library Pofacets', 'airplane.mat');
S = load(airplane_mat);
coord = double(S.coord);
facet = double(S.facet);

fprintf('========================================\n');
fprintf('PO Comparison: airplane.mat vs demo_aircraft.obj\n');
fprintf('========================================\n');
fprintf('Airplane.mat: %d vertices, %d facets\n', size(coord,1), size(facet,1));

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
    nidotk = N(m,:) * Ri.';
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
out_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
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

% Plot comparison (if Julia results exist)
julia_phi0 = fullfile(out_dir, 'julia_po_aircraft_0p3_phi0.csv');
julia_phi90 = fullfile(out_dir, 'julia_po_aircraft_0p3_phi90.csv');

figure('Position', [100, 100, 1200, 500]);

% φ=0° cut
subplot(1,2,1);
plot(results.phi0.theta_deg, results.phi0.sigma_dBsm, 'b-', 'LineWidth', 2, 'DisplayName', 'POFacets 4.5');
hold on;
if isfile(julia_phi0)
    J0 = readtable(julia_phi0);
    plot(J0.theta_deg, J0.sigma_dBsm, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Julia PO');

    % Compute RMSE
    % Interpolate to common grid
    sigma_pof_interp = interp1(results.phi0.theta_deg, results.phi0.sigma_dBsm, J0.theta_deg, 'linear', 'extrap');
    rmse = sqrt(mean((J0.sigma_dBsm - sigma_pof_interp).^2));
    fprintf('\nφ=0° RMSE: %.3f dB\n', rmse);
    legend('Location', 'best');
else
    legend('Location', 'best');
    fprintf('\nJulia results not found at: %s\n', julia_phi0);
end
grid on;
xlabel('θ (deg)');
ylabel('Bistatic RCS (dBsm)');
title(sprintf('Aircraft φ=0° — %.1f GHz', freq_ghz));
xlim([0 180]);

% φ=90° cut
subplot(1,2,2);
plot(results.phi90.theta_deg, results.phi90.sigma_dBsm, 'b-', 'LineWidth', 2, 'DisplayName', 'POFacets 4.5');
hold on;
if isfile(julia_phi90)
    J90 = readtable(julia_phi90);
    plot(J90.theta_deg, J90.sigma_dBsm, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Julia PO');

    sigma_pof_interp = interp1(results.phi90.theta_deg, results.phi90.sigma_dBsm, J90.theta_deg, 'linear', 'extrap');
    rmse = sqrt(mean((J90.sigma_dBsm - sigma_pof_interp).^2));
    fprintf('φ=90° RMSE: %.3f dB\n', rmse);
    legend('Location', 'best');
else
    legend('Location', 'best');
    fprintf('\nJulia results not found at: %s\n', julia_phi90);
end
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
fprintf('Done.\n');
fprintf('========================================\n');
