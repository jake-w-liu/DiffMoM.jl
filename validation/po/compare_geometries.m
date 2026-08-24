% compare_geometries.m — Compare the POFacets and DiffMoM aircraft meshes.
%
% Required inputs:
%   POFACETS_AIRCRAFT_MAT=/absolute/path/to/airplane.mat
%   DMOM_AIRCRAFT_OBJ=/absolute/path/to/aircraft.obj
%
% POFACETS_AIRCRAFT_MAT may be omitted when POFACETS_DIR points to a
% POFacets installation containing CAD Library Pofacets/airplane.mat.

clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..', '..');

airplane_mat = getenv('POFACETS_AIRCRAFT_MAT');
if isempty(airplane_mat)
    pof_dir = getenv('POFACETS_DIR');
    if isempty(pof_dir)
        error(['Set POFACETS_AIRCRAFT_MAT to airplane.mat, or set ' ...
               'POFACETS_DIR to the POFacets 4.5 directory, then rerun.']);
    end
    airplane_mat = fullfile(pof_dir, 'CAD Library Pofacets', 'airplane.mat');
end
if exist(airplane_mat, 'file') ~= 2
    error(['POFacets aircraft geometry not found at %s. Set ' ...
           'POFACETS_AIRCRAFT_MAT to the required airplane.mat and rerun.'], ...
          airplane_mat);
end

obj_file = getenv('DMOM_AIRCRAFT_OBJ');
if isempty(obj_file)
    obj_file = fullfile(repo_root, 'examples', 'demo_aircraft.obj');
end
if exist(obj_file, 'file') ~= 2
    error(['DiffMoM aircraft geometry not found at %s. Set ' ...
           'DMOM_AIRCRAFT_OBJ to the required triangular OBJ and rerun.'], ...
          obj_file);
end

S = load(airplane_mat);
if ~isfield(S, 'coord') || ~isfield(S, 'facet')
    error(['POFacets geometry %s must contain coord and facet arrays. ' ...
           'Select a compatible airplane.mat file.'], airplane_mat);
end
coord_mat = double(S.coord);
facet_data = double(S.facet);
if size(coord_mat, 2) < 3 || size(facet_data, 2) < 3
    error('POFacets coord and facet arrays must each have at least three columns.');
end
coord_mat = coord_mat(:, 1:3);
facet_mat = facet_data(:, 1:3);
if isempty(coord_mat) || isempty(facet_mat) || ...
        any(~isfinite(coord_mat), 'all') || any(~isfinite(facet_mat), 'all')
    error('POFacets coord and facet arrays must be nonempty and finite.');
end
if any(facet_mat ~= fix(facet_mat), 'all') || ...
        any(facet_mat < 1, 'all') || any(facet_mat > size(coord_mat, 1), 'all')
    error('POFacets facet indices must be integers within the coord array.');
end

[vertices_obj, faces_obj] = read_triangular_obj(obj_file);

fprintf('Geometry comparison inputs\n');
fprintf('  POFacets MAT: %s\n', airplane_mat);
fprintf('  DiffMoM OBJ:  %s\n', obj_file);
fprintf('  Coordinate tolerance: 1e-6 m\n\n');
fprintf('POFacets: %d vertices, %d facets\n', ...
        size(coord_mat, 1), size(facet_mat, 1));
fprintf('DiffMoM:  %d vertices, %d facets\n', ...
        size(vertices_obj, 1), size(faces_obj, 1));

vertex_count_match = size(coord_mat, 1) == size(vertices_obj, 1);
if vertex_count_match
    max_coordinate_difference = max(abs(coord_mat - vertices_obj), [], 'all');
    coordinates_match = max_coordinate_difference <= 1e-6;
    fprintf('Maximum same-index coordinate difference: %.6g m\n', ...
            max_coordinate_difference);
else
    coordinates_match = false;
    fprintf('Coordinate comparison skipped because vertex counts differ.\n');
end

facet_count_match = size(facet_mat, 1) == size(faces_obj, 1);
if facet_count_match
    pof_oriented = sortrows(canonical_oriented_faces(facet_mat));
    obj_oriented = sortrows(canonical_oriented_faces(faces_obj));
    oriented_connectivity_match = isequal(pof_oriented, obj_oriented);
    unoriented_connectivity_match = isequal(
        sortrows(sort(facet_mat, 2)), sortrows(sort(faces_obj, 2)));
else
    oriented_connectivity_match = false;
    unoriented_connectivity_match = false;
end

fprintf('Vertex count: %s\n', status_word(vertex_count_match));
fprintf('Same-index coordinates: %s\n', status_word(coordinates_match));
fprintf('Facet count: %s\n', status_word(facet_count_match));
fprintf('Oriented triangle connectivity (row-order independent): %s\n', ...
        status_word(oriented_connectivity_match));
if unoriented_connectivity_match && ~oriented_connectivity_match
    fprintf(['Triangle vertex sets match, but one or more windings differ. ' ...
             'Repair orientation before comparing PO results.\n']);
end

overall_match = coordinates_match && oriented_connectivity_match;
if ~overall_match
    error(['Geometry comparison failed. Use meshes with the same vertex order, ' ...
           'coordinates (within 1e-6 m), triangle sets, and winding before ' ...
           'comparing solver output.']);
end

fprintf('\nPASS: coordinates and oriented triangle connectivity match.\n');

function [vertices, faces] = read_triangular_obj(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Could not open OBJ geometry at %s for reading.', path);
    end
    cleanup = onCleanup(@() fclose(fid));
    vertices = zeros(0, 3);
    faces = zeros(0, 3);
    line_number = 0;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        line_number = line_number + 1;
        trimmed = strtrim(line);
        if isempty(trimmed) || startsWith(trimmed, '#')
            continue;
        end
        if ~isempty(regexp(trimmed, '^v\s+', 'once'))
            values = transpose(sscanf(trimmed(2:end), '%f'));
            if numel(values) < 3 || any(~isfinite(values(1:3)))
                error('OBJ %s has an invalid vertex at line %d.', path, line_number);
            end
            vertices(end + 1, :) = values(1:3); %#ok<AGROW>
        elseif ~isempty(regexp(trimmed, '^f\s+', 'once'))
            tokens = regexp(strtrim(trimmed(2:end)), '\s+', 'split');
            if numel(tokens) ~= 3
                error(['OBJ %s has a non-triangular face at line %d. ' ...
                       'Triangulate the mesh before comparison.'], path, line_number);
            end
            face = zeros(1, 3);
            for index = 1:3
                fields = strsplit(tokens{index}, '/');
                vertex_index = str2double(fields{1});
                if ~isfinite(vertex_index) || vertex_index ~= fix(vertex_index) || ...
                        vertex_index <= 0
                    error(['OBJ %s has an unsupported vertex index at line %d. ' ...
                           'Use positive integer face indices.'], path, line_number);
                end
                face(index) = vertex_index;
            end
            faces(end + 1, :) = face; %#ok<AGROW>
        end
    end
    clear cleanup;
    if isempty(vertices) || isempty(faces)
        error('OBJ %s must contain at least one vertex and one triangular face.', path);
    end
    if any(faces > size(vertices, 1), 'all')
        error('OBJ %s contains a face index beyond its %d vertices.', ...
              path, size(vertices, 1));
    end
end

function canonical = canonical_oriented_faces(faces)
    canonical = zeros(size(faces));
    for row = 1:size(faces, 1)
        rotations = [faces(row, :); faces(row, [2, 3, 1]); faces(row, [3, 1, 2])];
        rotations = sortrows(rotations);
        canonical(row, :) = rotations(1, :);
    end
end

function word = status_word(condition)
    if condition
        word = 'MATCH';
    else
        word = 'DIFFER';
    end
end
