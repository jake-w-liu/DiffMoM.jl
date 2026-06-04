% compare_geometries.m — Check if demo_aircraft.obj == airplane.mat

clear; close all; clc;

% Load airplane.mat
pof_dir = '/Users/jake/Library/CloudStorage/OneDrive-Personal/tests/testutdp/pofacets4.5';
airplane_mat = fullfile(pof_dir, 'CAD Library Pofacets', 'airplane.mat');
S = load(airplane_mat);
coord_mat = double(S.coord);
facet_mat = double(S.facet);

fprintf('Airplane.mat:\n');
fprintf('  Vertices: %d\n', size(coord_mat,1));
fprintf('  Facets: %d\n', size(facet_mat,1));
fprintf('  Vertex range: x[%.2f, %.2f], y[%.2f, %.2f], z[%.2f, %.2f]\n', ...
    min(coord_mat(:,1)), max(coord_mat(:,1)), ...
    min(coord_mat(:,2)), max(coord_mat(:,2)), ...
    min(coord_mat(:,3)), max(coord_mat(:,3)));

% Load demo_aircraft.obj
repo_root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
obj_file = fullfile(repo_root, 'examples', 'demo_aircraft.obj');
fid = fopen(obj_file, 'r');
vertices_obj = [];
faces_obj = [];

while ~feof(fid)
    line = fgetl(fid);
    if startsWith(line, 'v ')
        parts = strsplit(strtrim(line(3:end)));
        vertices_obj = [vertices_obj; str2double(parts(1:3))];
    elseif startsWith(line, 'f ')
        parts = strsplit(strtrim(line(3:end)));
        face = [];
        for i = 1:length(parts)
            idx_str = strsplit(parts{i}, '/');
            face = [face, str2double(idx_str{1})];
        end
        faces_obj = [faces_obj; face];
    end
end
fclose(fid);

fprintf('\nDemo_aircraft.obj:\n');
fprintf('  Vertices: %d\n', size(vertices_obj,1));
fprintf('  Faces: %d\n', size(faces_obj,1));
fprintf('  Vertex range: x[%.2f, %.2f], y[%.2f, %.2f], z[%.2f, %.2f]\n', ...
    min(vertices_obj(:,1)), max(vertices_obj(:,1)), ...
    min(vertices_obj(:,2)), max(vertices_obj(:,2)), ...
    min(vertices_obj(:,3)), max(vertices_obj(:,3)));

% Check if vertices match
if isequal(size(coord_mat), size(vertices_obj))
    fprintf('\n✓ Vertex counts match\n');

    % Check coordinate match (with tolerance)
    max_diff = max(abs(coord_mat - vertices_obj), [], 'all');
    fprintf('  Max vertex coordinate difference: %.6f m\n', max_diff);

    if max_diff < 1e-6
        fprintf('  ✓ Vertices identical\n');
    else
        fprintf('  ✗ Vertices differ!\n');
    end
else
    fprintf('\n✗ Vertex counts differ!\n');
end

% Check if facets match
if isequal(size(facet_mat(:,1:3)), size(faces_obj))
    fprintf('\n✓ Facet counts match\n');

    % Check connectivity (1-indexed in both)
    mat_faces = facet_mat(:,1:3);

    % Direct match
    direct_match = isequal(mat_faces, faces_obj);
    if direct_match
        fprintf('  ✓ Facet connectivity identical\n');
    else
        % Check winding order (might be flipped)
        flipped_obj = faces_obj(:,[1,3,2]);  % swap columns 2<->3
        flipped_match = isequal(mat_faces, flipped_obj);

        if flipped_match
            fprintf('  ⚠ Facet winding FLIPPED (columns 2<->3)\n');
        else
            % Check row permutation
            mat_sorted = sortrows(sort(mat_faces, 2));
            obj_sorted = sortrows(sort(faces_obj, 2));
            if isequal(mat_sorted, obj_sorted)
                fprintf('  ⚠ Facet connectivity same but REORDERED\n');
            else
                fprintf('  ✗ Facet connectivity completely different!\n');

                % Show first 5 facets
                fprintf('\n  First 5 facets (airplane.mat):\n');
                disp(mat_faces(1:5,:));
                fprintf('  First 5 facets (demo_aircraft.obj):\n');
                disp(faces_obj(1:5,:));
            end
        end
    end
else
    fprintf('\n✗ Facet counts differ!\n');
end

% Compute normals and check orientation
fprintf('\nNormal vector check (first 10 facets):\n');
fprintf('  Facet   airplane.mat normal           demo_aircraft.obj normal\n');
fprintf('  ------  ---------------------------   ---------------------------\n');

for i = 1:min(10, size(facet_mat,1))
    % airplane.mat normal
    v1_mat = coord_mat(facet_mat(i,1),:);
    v2_mat = coord_mat(facet_mat(i,2),:);
    v3_mat = coord_mat(facet_mat(i,3),:);
    A_mat = v2_mat - v1_mat;
    B_mat = v3_mat - v2_mat;
    n_mat = -cross(B_mat, A_mat);  % POFacets convention
    n_mat = n_mat / norm(n_mat);

    % demo_aircraft.obj normal
    v1_obj = vertices_obj(faces_obj(i,1),:);
    v2_obj = vertices_obj(faces_obj(i,2),:);
    v3_obj = vertices_obj(faces_obj(i,3),:);
    n_obj = cross(v2_obj - v1_obj, v3_obj - v1_obj);  % Julia convention
    n_obj = n_obj / norm(n_obj);

    % Compare
    dot_prod = dot(n_mat, n_obj);
    match = abs(dot_prod - 1.0) < 0.01;
    flip = abs(dot_prod + 1.0) < 0.01;

    status = '?';
    if match
        status = '✓';
    elseif flip
        status = '✗ FLIPPED';
    end

    fprintf('  %3d    [%6.3f %6.3f %6.3f]      [%6.3f %6.3f %6.3f]  %s\n', ...
        i, n_mat(1), n_mat(2), n_mat(3), n_obj(1), n_obj(2), n_obj(3), status);
end

fprintf('\nDone.\n');
