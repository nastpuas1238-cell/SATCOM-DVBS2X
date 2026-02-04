function verify_ldpc(outdir, matfile, report_file)
%VERIFY_LDPC  Cross-check DVB-S2 BCH and LDPC results from Python in MATLAB.
%
% Usage (from repo root):
%   verify_ldpc()                            % uses verification/out/vectors.mat
%   verify_ldpc('verification/out')          % custom output folder
%   verify_ldpc('verification/out','my.mat') % explicit MAT file
%
% The MAT file is produced by verification/dump_vectors.py and contains:
%   raw_bits         - original input bits (<= Kbch)
%   padded_kbch      - padded (unp scrambled) Kbch bits
%   scrambled_kbch   - BB-scrambled Kbch bits (BCH input)
%   bch_codeword     - Nbch bits
%   ldpc_codeword    - Nldpc bits
%   meta             - struct with fields: fecframe, rate, Kbch, Nbch, kldpc, nldpc, mat_path
%
% This script performs:
%   1) BCH check using the ETSI generator polynomial (no toolbox dependency).
%   2) LDPC syndrome check using the parity table in meta.mat_path (PT_* key).
%
% It prints PASS/FAIL for each stage and returns true/false overall.

% Default: look in the same folder as this script (no subdirectories needed)
script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outdir)
    outdir = script_dir;
end
if nargin < 2 || isempty(matfile)
    matfile = fullfile(outdir, 'vectors.mat');
end
if nargin < 3 || isempty(report_file)
    report_file = fullfile(outdir, 'verify_report.txt');
end
% Fallbacks if the file is still missing
if ~isfile(matfile)
    alt1 = fullfile(pwd, 'vectors.mat');
    alt2 = fullfile(script_dir, 'vectors.mat');
    if isfile(alt1)
        matfile = alt1;
    elseif isfile(alt2)
        matfile = alt2;
    else
        error('MAT file not found. Tried:\n  %s\n  %s\n  %s', matfile, alt1, alt2);
    end
end

S = load(matfile);
required = {'scrambled_kbch','bch_codeword','ldpc_codeword','meta'};
for i = 1:numel(required)
    if ~isfield(S, required{i})
        error('Missing field "%s" in %s', required{i}, matfile);
    end
end

y_scr = logical(S.scrambled_kbch(:));
c_bch = logical(S.bch_codeword(:));
c_ldpc = logical(S.ldpc_codeword(:));
meta = S.meta;

fecframe = string(meta.fecframe);
rate     = string(meta.rate);

fprintf('\n--- MATLAB cross-check ---\n');
fprintf('fecframe = %s, rate = %s\n', fecframe, rate);

% Prepare report writer
fid = fopen(report_file, 'w');
if fid < 0
    warning('Could not open report file %s for writing. Skipping file log.', report_file);
end
logline(fid, 'MATFILE', matfile);
logline(fid, 'MAT_PATH_RESOLVED', mat_path_if_exists(matfile));
logline(fid, 'FECFRAME', fecframe);
logline(fid, 'RATE', rate);
logline(fid, 'RAW_BITS_LEN', numel(S.raw_bits));
logline(fid, 'SCRAMBLED_KBCH_LEN', numel(y_scr));
logline(fid, 'BCH_LEN', numel(c_bch));
logline(fid, 'LDPC_LEN', numel(c_ldpc));

% Save bitstreams for inspection
raw_bits_path       = fullfile(outdir, 'raw_bits_dump.txt');
scrambled_path      = fullfile(outdir, 'scrambled_kbch_dump.txt');
bch_path            = fullfile(outdir, 'bch_codeword_dump.txt');
ldpc_path           = fullfile(outdir, 'ldpc_codeword_dump.txt');
write_bits(raw_bits_path, S.raw_bits);
write_bits(scrambled_path, y_scr);
write_bits(bch_path, c_bch);
write_bits(ldpc_path, c_ldpc);
logline(fid, 'RAW_BITS_FILE', raw_bits_path);
logline(fid, 'SCRAMBLED_FILE', scrambled_path);
logline(fid, 'BCH_FILE', bch_path);
logline(fid, 'LDPC_FILE', ldpc_path);

% ---------------- BCH check ----------------
[Kbch, Nbch, t] = bch_params(fecframe, rate);
if numel(y_scr) ~= Kbch
    error('scrambled_kbch length mismatch: got %d, expected %d', numel(y_scr), Kbch);
end
if numel(c_bch) ~= Nbch
    error('bch_codeword length mismatch: got %d, expected %d', numel(c_bch), Nbch);
end

g = dvbs2_bch_generator(fecframe, t); % generator polynomial row vector over GF(2)
[~, rem_bch] = gfdeconv(double(c_bch).', double(g), 2);
bch_ok = ~any(rem_bch); % all-zero remainder => valid codeword
fprintf('BCH parity check : %s\n', tf(bch_ok));
logline(fid, 'BCH_PARITY', tf(bch_ok));
logline(fid, 'BCH_LENGTH_MATCH', tf(numel(c_bch) == Nbch));

% ---------------- LDPC check ----------------
mat_path = meta.mat_path;
if ~isfile(mat_path)
    local_mat = fullfile(outdir, 'dvbs2xLDPCParityMatrices.mat');
    alt_script = fullfile(script_dir, 'dvbs2xLDPCParityMatrices.mat');
    if isfile(local_mat)
        mat_path = local_mat;
    elseif isfile(alt_script)
        mat_path = alt_script;
    else
        error('Parity matrix .mat not found. Tried:\n  %s\n  %s\n  %s', mat_path, local_mat, alt_script);
    end
end

expected_m = meta.nldpc - meta.kldpc;
H = load_parity_matrix(mat_path, fecframe, rate, numel(c_ldpc), expected_m);
syn = mod(double(H) * double(c_ldpc(:)), 2);
ldpc_ok = all(syn == 0);
fprintf('LDPC syndrome    : %s (rows=%d, n=%d)\n', tf(ldpc_ok), size(H,1), size(H,2));
logline(fid, 'LDPC_SYNDROME', tf(ldpc_ok));
logline(fid, 'LDPC_SHAPE', sprintf('%d x %d', size(H,1), size(H,2)));
logline(fid, 'LDPC_LENGTH_MATCH', tf(numel(c_ldpc) == meta.nldpc));

ok = bch_ok && ldpc_ok;
if ok
    fprintf('Overall          : PASS ✅\n\n');
else
    fprintf('Overall          : FAIL ❌\n\n');
end

% ---------------- Optional built-in cross-checks (if available) ----------------
% BCH built-in (Communications Toolbox): dvbs2bchEncode
if exist('dvbs2bchEncode', 'file') == 2
    logline(fid, 'BCH_BUILTIN_AVAILABLE', 'YES');
    try
        c_bch_ref = dvbs2bchEncode(y_scr, fecframe, rate);
        bch_diff = nnz(c_bch_ref(:) ~= c_bch(:));
        logline(fid, 'BCH_BUILTIN_DIFF', num2str(bch_diff));
        logline(fid, 'BCH_BUILTIN_EQUAL', tf(bch_diff == 0));
        fprintf('BCH builtin diff: %d bits\n', bch_diff);
    catch ME
        logline(fid, 'BCH_BUILTIN_ERROR', ME.message);
        fprintf('BCH builtin error: %s\n', ME.message);
    end
else
    logline(fid, 'BCH_BUILTIN_AVAILABLE', 'NO');
end

% LDPC built-in: dvbs2ldpcEncode (if present)
if exist('dvbs2ldpcEncode', 'file') == 2
    logline(fid, 'LDPC_BUILTIN_AVAILABLE', 'YES');
    try
        c_ldpc_ref = dvbs2ldpcEncode(c_bch, fecframe, rate);
        ldpc_diff = nnz(c_ldpc_ref(:) ~= c_ldpc(:));
        logline(fid, 'LDPC_BUILTIN_DIFF', num2str(ldpc_diff));
        logline(fid, 'LDPC_BUILTIN_EQUAL', tf(ldpc_diff == 0));
        fprintf('LDPC builtin diff: %d bits\n', ldpc_diff);
    catch ME
        logline(fid, 'LDPC_BUILTIN_ERROR', ME.message);
        fprintf('LDPC builtin error: %s\n', ME.message);
    end
else
    logline(fid, 'LDPC_BUILTIN_AVAILABLE', 'NO');
end

% Final summary in report
logline(fid, 'SUMMARY_BCH_EQUALS_MATLAB', tf(bch_ok));
logline(fid, 'SUMMARY_LDPC_EQUALS_MATLAB', tf(ldpc_ok));
logline(fid, 'SUMMARY_OVERALL', tf(ok));

if fid >= 0
    fclose(fid);
    fprintf('Report written to %s\n', report_file);
end
end

% -------------------------------------------------------------------------
function [Kbch, Nbch, t] = bch_params(fecframe, rate)
table = containers.Map;
table("normal_1/4")  = [16008 16200 12];
table("normal_1/3")  = [21408 21600 12];
table("normal_2/5")  = [25728 25920 12];
table("normal_1/2")  = [32208 32400 12];
table("normal_3/5")  = [38688 38880 12];
table("normal_2/3")  = [43040 43200 10];
table("normal_3/4")  = [48408 48600 12];
table("normal_4/5")  = [51648 51840 12];
table("normal_5/6")  = [53840 54000 10];
table("normal_8/9")  = [57472 57600 8];
table("normal_9/10") = [58192 58320 8];
table("short_1/4")   = [3072 3240 12];
table("short_1/3")   = [5232 5400 12];
table("short_2/5")   = [6312 6480 12];
table("short_1/2")   = [7032 7200 12];
table("short_3/5")   = [9552 9720 12];
table("short_2/3")   = [10632 10800 12];
table("short_3/4")   = [11712 11880 12];
table("short_4/5")   = [12432 12600 12];
table("short_5/6")   = [13152 13320 12];
table("short_8/9")   = [14232 14400 12];

key = lower(strcat(fecframe, "_", rate));
if ~isKey(table, key)
    error('Unsupported (fecframe, rate): %s', key);
end
vals = table(key);
Kbch = vals(1); Nbch = vals(2); t = vals(3);
end

% -------------------------------------------------------------------------
function g = dvbs2_bch_generator(fecframe, t)
% Multiply first t polynomials from ETSI Tables 6a/6b (exponent form).
if strcmpi(fecframe, "normal")
    G = {
        [0 2 3 5 16], ...
        [0 1 4 5 6 8 16], ...
        [0 2 3 4 5 7 8 9 10 11 16], ...
        [0 2 4 6 9 11 12 14 16], ...
        [0 1 2 3 5 8 9 10 11 12 16], ...
        [0 2 4 5 7 8 9 10 12 13 14 15 16], ...
        [0 2 5 6 8 9 10 11 13 15 16], ...
        [0 1 2 5 6 8 9 12 13 14 16], ...
        [0 5 7 9 10 11 16], ...
        [0 1 2 5 7 8 10 12 13 14 16], ...
        [0 2 3 5 9 11 12 13 16], ...
        [0 1 5 6 7 9 11 12 16] ...
    };
else
    G = {
        [0 1 3 5 14], ...
        [0 6 8 11 14], ...
        [0 1 2 6 9 10 14], ...
        [0 4 7 8 10 12 14], ...
        [0 2 4 6 8 9 11 13 14], ...
        [0 3 7 8 9 13 14], ...
        [0 2 5 6 7 10 11 13 14], ...
        [0 5 8 9 10 11 14], ...
        [0 1 2 3 9 10 14], ...
        [0 3 6 9 11 12 14], ...
        [0 4 11 12 14], ...
        [0 1 2 3 5 6 7 8 10 13 14] ...
    };
end

g = 1;
for i = 1:t
    g = gfconv(g, exps_to_poly(G{i}), 2);
end
g = logical(mod(g, 2));
end

% -------------------------------------------------------------------------
function p = exps_to_poly(exps)
% exps -> polynomial vector, MSB = highest degree
d = max(exps);
p = zeros(1, d + 1);
p(d - exps + 1) = 1;
end

% -------------------------------------------------------------------------
function H = load_parity_matrix(mat_path, fecframe, rate, expected_n, expected_m)
    if ~isfile(mat_path)
        error('Parity matrix .mat not found: %s', mat_path);
    end
    mat = load(mat_path);
    rate_key = replace(rate, "/", "_");
    % try common naming first (PT_1_2_S / PT_3_5_N, etc.)
    suffix = upper(fecframe(1)); % 'N' or 'S'
    % Gather all PT_* fields
    fn = fieldnames(mat);
    pt_fields = fn(startsWith(fn, "PT_", 'IgnoreCase', true));
    if isempty(pt_fields)
        error('No PT_* fields found in %s', mat_path);
    end

    % Compute n,m for each PT_* and filter by expected_n/expected_m
    matches = {};
    ns = [];
    ms = [];
    for i = 1:numel(pt_fields)
        f = pt_fields{i};
        pairs = mat.(f);
        if size(pairs,2) ~= 2, continue; end
        n_val = max(pairs(:,2));
        m_val = max(pairs(:,1));
        ns(end+1) = n_val; %#ok<AGROW>
        ms(end+1) = m_val; %#ok<AGROW>
        if ~isempty(expected_n) && n_val == expected_n ...
                && (~exist('expected_m','var') || isempty(expected_m) || m_val == expected_m)
            matches{end+1} = f; %#ok<AGROW>
        end
    end

    if isempty(matches)
        % try relaxing m match but same n
        if ~isempty(expected_n)
            for i = 1:numel(pt_fields)
                f = pt_fields{i};
                pairs = mat.(f);
                if size(pairs,2) ~= 2, continue; end
                if max(pairs(:,2)) == expected_n
                    matches{end+1} = f; %#ok<AGROW>
                end
            end
        end
    end

    if isempty(matches)
        available = unique([ns(:) ms(:)], 'rows');
        error(['No parity table with n=%d, m=%d found. Available (n,m) rows: %s. ' ...
               'Check that you are using SHORT vs NORMAL correctly.'], ...
               expected_n, expected_m, mat2str(available));
    end

    chosen = matches{1};
    pairs = mat.(chosen);
    rows = double(pairs(:, 1));
    cols = double(pairs(:, 2));
    % Parity tables should be 1-based. If zero-based values appear, shift.
    if min(rows) == 0 || min(cols) == 0
        rows = rows + 1;
        cols = cols + 1;
    end
    M = max(rows);
    N = max(cols);
    H = sparse(rows, cols, true, M, N);
end

% -------------------------------------------------------------------------
function out = tf(flag)
out = "PASS";
if ~flag
    out = "FAIL";
end
end

% -------------------------------------------------------------------------
function logline(fid, key, val)
if fid < 0
    return
end
if isstring(val) || ischar(val)
    fprintf(fid, "%s: %s\n", key, string(val));
else
    fprintf(fid, "%s: %s\n", key, mat2str(val));
end
end

% -------------------------------------------------------------------------
function p = mat_path_if_exists(pathstr)
if isfile(pathstr)
    p = which(pathstr);
else
    p = "NOT_FOUND";
end
end

% -------------------------------------------------------------------------
function write_bits(path, bits)
    try
        fid = fopen(path, 'w');
        if fid < 0, return; end
        fprintf(fid, '%s', char('0' + reshape(uint8(bits(:)), 1, [])));
        fclose(fid);
    catch
        % silently ignore write errors
    end
end
