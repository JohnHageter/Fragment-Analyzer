"""ABIF/FSA file reader for fragment analysis data."""

import struct
import numpy as np
from pathlib import Path

# GeneScan 600 LIZ size standard fragment sizes (bp)
GS600_LIZ_SIZES = [
    20, 40, 60, 80, 100, 114, 120, 140, 160, 180, 200, 214, 220, 240, 260,
    280, 300, 314, 320, 340, 360, 380, 400, 414, 420, 440, 460, 480, 500,
    514, 520, 540, 560, 580, 600
]

GS600_LIZ_60_600_SIZES = [
    60, 80, 100, 114, 120, 140, 160, 180, 200, 214, 220, 240, 260, 280, 300,
    314, 320, 340, 360, 380, 400, 414, 420, 440, 460, 480, 500, 514, 520,
    540, 560, 580, 600
]

# Catalogue of common size ladders keyed by display name
KNOWN_LADDERS = {
    'Auto-detect from file': None,
    'GS 600 LIZ (60-600)': GS600_LIZ_60_600_SIZES,
    'GS 600 LIZ (20-600)': GS600_LIZ_SIZES,
    'GS 500 LIZ': [
        35, 50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450, 490, 500
    ],
    'GS 400HD ROX': [
        50, 60, 90, 100, 120, 150, 160, 180, 190, 200, 220, 240, 260, 280,
        290, 300, 320, 340, 360, 380, 400
    ],
    'GS 350 ROX': [35, 50, 75, 100, 150, 200, 250, 300, 350],
    'Custom': None,  # user-entered sizes
}

CHANNEL_NAMES = {1: '6-FAM', 2: 'VIC', 3: 'NED', 4: 'PET'}
CHANNEL_COLORS = {
    1: '#1565C0',
    2: '#2E7D32',
    3: '#E65100',
    4: '#B71C1C',
    5: '#6A1B9A',
}


def _doublet_score(scan_gaps, size_gaps, threshold):
    """Count (14 bp, 6 bp) doublet pairs where both scan gaps are small.

    GS600 LIZ contains evenly-spaced (14, 6) bp consecutive gap pairs at
    every 100 bp boundary (100→114→120, 200→214→220, …).  In CE scan space
    these show up as TWO CONSECUTIVE SMALL SCAN GAPS.  The correct size window
    is the one whose expected doublet positions align with observed consecutive
    small scan gaps.
    """
    score = 0
    n = len(scan_gaps)
    for i in range(len(size_gaps) - 1):
        if size_gaps[i] == 14 and size_gaps[i + 1] == 6:
            if i + 1 < n and scan_gaps[i] <= threshold and scan_gaps[i + 1] <= threshold:
                score += 1
    return score


def _bp_at_scan0_linear(candidates, size_window):
    """Extrapolate a linear scan→bp fit back to scan 0.

    CE runs physically start with no DNA at the detector (scan 0 ≈ time 0),
    so bp(scan=0) should be ≤ 0 for the correct window.  Windows that start
    later in the ladder (e.g. 180 bp vs 80 bp) shift all fitted bp values up
    by one ladder period (100 bp), making bp(0) positive — physically wrong.
    A linear fit is used deliberately: polynomial extrapolation to scan 0 is
    unstable, while the linear fit gives a stable, monotone estimate whose
    sign reliably separates windows one period apart.
    """
    x = candidates.astype(float)
    y = np.asarray(size_window, dtype=float)
    if len(x) < 2:
        return 0.0
    slope = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
    intercept = y.mean() - slope * x.mean()
    return intercept  # bp at scan=0


def _loglog_r2(peak_window, size_window):
    """R² of a linear fit of ln(scan) vs ln(bp) — used for peak-window sliding."""
    ln_s = np.log(peak_window.astype(float))
    ln_b = np.log(np.asarray(size_window, dtype=float))
    if len(ln_s) < 2:
        return -np.inf
    # Closed-form simple regression (avoids np.polyfit's LAPACK call)
    db = ln_b - ln_b.mean()
    denom = float(np.dot(db, db))
    slope = float(np.dot(db, ln_s - ln_s.mean()) / denom) if denom > 0 else 0.0
    residuals = (ln_s - ln_s.mean()) - slope * db
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.var(ln_s) * len(ln_s))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def _select_ladder_peaks(candidates, std_arr, signal):
    """Return (peaks, matched_sizes) using the best-fitting size window.

    Two cases:
      n_cand <= n_std  — slide the *size* window: small-fragment peaks often
                         fall below threshold; find the n_cand consecutive
                         sizes that best match the detected peaks.
      n_cand > n_std   — slide the *peak* window: extra candidates present;
                         find the n_std consecutive peaks that fit the full
                         ladder sizes best.

    For the size-window case, scoring uses two criteria in order:
      1. Doublet fingerprint (primary): GS600 LIZ has (14 bp, 6 bp) consecutive
         gap pairs at every 100-bp boundary.  These appear as TWO CONSECUTIVE
         SMALL SCAN GAPS.  The window that has more expected doublet positions
         matching observed consecutive small scan gaps is preferred.
      2. Physics extrapolation (tiebreaker): extrapolate the fitted bp(scan)
         back to scan = 0.  At time zero no DNA has reached the detector, so
         bp(0) must be ≤ 0 for the correct window.  Windows that start later
         in the ladder shift all bp values up by one ladder period, giving a
         positive bp(0) — physically wrong.  We pick the window with the
         smallest (most negative) bp(0).

    This combination correctly resolves the degeneracy inherent in GS600 LIZ:
    its gap pattern repeats every 6 positions, so windows one period apart
    (e.g. 80 bp vs 180 bp) produce identical Pearson correlations and identical
    log-log R² values.  The doublet + bp(0) approach is immune to that.
    """
    n_std  = len(std_arr)
    n_cand = len(candidates)

    if n_cand < 4:
        return None, None

    if n_cand <= n_std:
        scan_gaps = np.diff(candidates.astype(float))
        med_gap   = float(np.median(scan_gaps)) if len(scan_gaps) > 0 else 1.0
        threshold = med_gap * 0.70  # below 70 % of median → "small gap"

        best_doublets = -1
        best_bp0      = np.inf
        best_start    = 0

        for i in range(n_std - n_cand + 1):
            std_window = std_arr[i : i + n_cand]
            size_gaps  = np.diff(std_window)
            doublets   = _doublet_score(scan_gaps, size_gaps, threshold)
            bp0        = _bp_at_scan0_linear(candidates, std_window)

            if doublets > best_doublets or (doublets == best_doublets and bp0 < best_bp0):
                best_doublets = doublets
                best_bp0      = bp0
                best_start    = i

        return candidates, list(std_arr[best_start : best_start + n_cand])

    # More candidates than sizes — pre-trim to n_std + 12 tallest so the
    # window search remains tractable while not discarding real peaks.
    if n_cand > n_std + 12:
        heights  = signal[candidates]
        keep_idx = np.sort(np.argsort(heights)[-(n_std + 12):])
        candidates = candidates[keep_idx]
        n_cand     = len(candidates)

    best_r2    = -np.inf
    best_start = 0
    for i in range(n_cand - n_std + 1):
        r2 = _loglog_r2(candidates[i : i + n_std], std_arr)
        if r2 > best_r2:
            best_r2    = r2
            best_start = i

    return candidates[best_start : best_start + n_std], list(std_arr)


class FSAFile:
    """Parse and hold data from an Applied BioSystems ABIF (.fsa) file."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.name = self.filepath.stem
        self._entries = {}
        self._parse()


    def _parse(self):
        with open(self.filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'ABIF':
                raise ValueError(f"Not an ABIF file: {self.filepath}")

            f.read(2)  # version

            # Root directory entry (28 bytes total, we're at offset 6)
            f.read(4)   # tag
            f.read(4)   # tag_num
            f.read(2)   # elem_type
            f.read(2)   # elem_size
            num_entries = struct.unpack('>i', f.read(4))[0]
            f.read(4)   # data_size
            dir_offset = struct.unpack('>i', f.read(4))[0]
            f.read(4)   # handle

            # Read directory
            f.seek(dir_offset)
            dir_records = []
            for _ in range(num_entries):
                tag = f.read(4).decode('ascii', errors='replace')
                tag_num = struct.unpack('>i', f.read(4))[0]
                elem_type = struct.unpack('>H', f.read(2))[0]
                elem_size = struct.unpack('>H', f.read(2))[0]
                n_elems = struct.unpack('>i', f.read(4))[0]
                data_size = struct.unpack('>i', f.read(4))[0]
                data_raw = f.read(4)
                f.read(4)  # handle
                dir_records.append((tag, tag_num, elem_type, n_elems, data_size, data_raw))

            # Read data for each entry
            for tag, tag_num, elem_type, n_elems, data_size, data_raw in dir_records:
                if data_size <= 4:
                    data = data_raw[:data_size]
                else:
                    offset = struct.unpack('>i', data_raw)[0]
                    f.seek(offset)
                    data = f.read(data_size)
                self._entries[(tag, tag_num)] = {
                    'type': elem_type,
                    'n': n_elems,
                    'data': data,
                }

        self._load_channels()
        self._load_metadata()

    def _entry(self, tag, tag_num=1):
        return self._entries.get((tag, tag_num))

    def _read_shorts(self, tag, tag_num=1):
        e = self._entry(tag, tag_num)
        if e is None or len(e['data']) == 0:
            return None
        return np.frombuffer(e['data'], dtype='>i2').astype(np.float32)

    def _read_pstring(self, tag, tag_num=1):
        """Read a Pascal-style or C-style string entry."""
        e = self._entry(tag, tag_num)
        if e is None:
            return ''
        d = e['data']
        if not d:
            return ''
        if e['type'] == 18:  # pString
            length = d[0]
            return d[1:1 + length].decode('ascii', errors='replace')
        if e['type'] == 19:  # cString
            return d.rstrip(b'\x00').decode('ascii', errors='replace')
        return d.decode('ascii', errors='replace').strip('\x00')


    def _load_channels(self):
        # DATA[9-12]: color-corrected (matrix-applied) channels 1-4
        # DATA[1-4]:  raw channel data 1-4
        # DATA[205]:  LIZ size-standard channel (after matrix correction)

        corrected = {}
        for i in range(9, 13):
            d = self._read_shorts('DATA', i)
            if d is not None and len(d) > 100:
                corrected[i - 8] = d  # remap indices → 1-4

        raw = {}
        for i in range(1, 5):
            d = self._read_shorts('DATA', i)
            if d is not None and len(d) > 100:
                raw[i] = d

        self.channels = corrected if corrected else raw
        self.raw_channels = raw

        # LIZ channel: prefer DATA[205] (matrix-corrected), fall back to DATA[105]
        liz = self._read_shorts('DATA', 205)
        if liz is None or liz.max() < 5:
            liz = self._read_shorts('DATA', 105)

        # Corrected PET (DATA[12]) — used as a fallback when LIZ energy has
        # stayed in the PET channel rather than being isolated by matrix
        # correction.  Matrix correction still removes most 6-FAM bleed-through
        # so the LIZ pull-up into PET at ~2000 RFU is the dominant signal.
        corrected_pet = corrected.get(4)
        self._liz_corrected_pet = corrected_pet

        # Select the best amplitude-quality LIZ source for both display and
        # size calling.  Using the same source here keeps the "LIZ" trace in
        # the GUI consistent with what the calibration algorithm uses.
        liz_max = float(liz.max()) if liz is not None else 0.0
        pet_max = float(corrected_pet.max()) if corrected_pet is not None else 0.0
        if corrected_pet is not None and (liz_max < 200 or
                                          (liz_max < 1000 and pet_max > liz_max * 3)):
            self.liz_channel = corrected_pet
        else:
            self.liz_channel = liz

        if self.channels:
            self.n_scans = len(next(iter(self.channels.values())))
        else:
            self.n_scans = 0

        self.scan_numbers = np.arange(self.n_scans, dtype=np.float32)



    # Metadata
    def _load_metadata(self):
        # SpNm (specimen name) is the per-sample identifier; SMPL is often generic ("Sample")
        sp_nm = self._read_pstring('SpNm', 1)
        smpl  = self._read_pstring('SMPL', 1)
        self.sample_name = sp_nm or (smpl if smpl.lower() != 'sample' else '') or self.name

        self.dye_names = {}
        for i in range(1, 6):
            name = self._read_pstring('DyeN', i).strip()
            if name:
                self.dye_names[i] = name

        self.std_file = self._read_pstring('StdF', 1)

        rund = self._entry('RUND', 1)
        self.run_date = ''
        if rund and len(rund['data']) == 4:
            d = rund['data']
            year = struct.unpack('>H', d[0:2])[0]
            self.run_date = f"{year}-{d[2]:02d}-{d[3]:02d}"


    #Public
    def get_channel_data(self, ch):
        """Return fluorescence array for sample channel 1-4, or None."""
        return self.channels.get(ch)

    def get_channel_name(self, ch):
        return self.dye_names.get(ch, CHANNEL_NAMES.get(ch, f'Ch{ch}'))


    #Size calling
    def get_ladder_channel(self, ch):
        """Return the trace to use as the size standard for channel 4 or 5.

        Channel 4 is the matrix-corrected 4th dye (DATA[12], raw DATA[4] as a
        fallback); channel 5 is the dedicated size-standard channel
        (DATA[205], raw DATA[105] as a fallback).
        """
        if ch == 4:
            d = self.channels.get(4)
            return d if d is not None else self.raw_channels.get(4)
        if ch == 5:
            d = self._read_shorts('DATA', 205)
            return d if d is not None else self._read_shorts('DATA', 105)
        raise ValueError(f"Ladder channel must be 4 or 5, got {ch!r}")

    def attempt_size_calling(self, std_sizes=None, ladder_channel=None):
        """Attempt scan-number → base-pair conversion using the LIZ channel.

        Parameters
        ----------
        std_sizes : list[int] | None
            Ordered list of fragment sizes (bp) for the size standard used.
            Pass None to auto-detect from the StdF annotation in the file.
        ladder_channel : int | None
            4 or 5 to force the channel carrying the size standard.  None
            picks between channel 5 and corrected channel 4 automatically.

        Returns an array of bp values with length n_scans, or None on failure.
        """
        from scipy.signal import find_peaks

        if ladder_channel is not None:
            liz = self.get_ladder_channel(ladder_channel)
            liz_corr_pet = None   # explicit choice: skip auto-fallback below
        else:
            liz = self.liz_channel
            liz_corr_pet = getattr(self, '_liz_corrected_pet', None)

        # Select the best LIZ signal source.
        # DATA[205] is the matrix-corrected LIZ channel.  When it is weak
        # (i.e. LIZ energy stayed in the PET channel rather than being
        # isolated by matrix correction), the corrected PET (DATA[12])
        # provides a cleaner signal: matrix correction removed most 6-FAM
        # bleed-through while LIZ pull-up into PET persists at ~2000 RFU.
        if liz_corr_pet is not None:
            liz_max = float(liz.max()) if liz is not None else 0.0
            pet_max = float(liz_corr_pet.max())
            # Prefer corrected PET when DATA[205] is too weak or much weaker
            # than corrected PET.
            if liz_max < 200 or (liz_max < 1000 and pet_max > liz_max * 3):
                liz = liz_corr_pet
            elif liz is not None:
                # DATA[205] passes the amplitude check, but might still be
                # dominated by a single injection artifact with no real ladder
                # peaks.  Do a quick candidate count; if fewer than 5 peaks
                # exist in DATA[205], fall back to corrected PET.
                _lc = np.clip(liz, 0, None)
                _ls = _lc.copy(); _ls[:500] = 0
                _min_h = max(float(_lc.max()) * 0.04, 3.0)
                _nc, _ = find_peaks(_ls, height=_min_h, distance=30,
                                    prominence=_min_h * 0.2)
                # A partial ladder in DATA[205] (e.g. 15 of 22 peaks) is
                # also worse than corrected PET: switch when PET shows
                # clearly more peaks.
                _pc = np.clip(liz_corr_pet, 0, None)
                _ps = _pc.copy(); _ps[:500] = 0
                _pmin_h = max(float(_pc.max()) * 0.04, 3.0)
                _np, _ = find_peaks(_ps, height=_pmin_h, distance=30,
                                    prominence=_pmin_h * 0.2)
                if len(_nc) < 5 or len(_np) >= len(_nc) + 5:
                    liz = liz_corr_pet

        if liz is None or len(liz) < 100:
            return None

        liz_clean = np.clip(liz, 0, None)

        # Skip the first 500 scans — pull-up from the 6-FAM primer-dimer
        # peak bleeds into LIZ at early scan positions and would be picked up
        # as a false ladder peak.  The smallest GS600 LIZ fragment (60 bp)
        # never appears before scan ~500 in practice.
        avoid_scans = 500
        liz_search = liz_clean.copy()
        liz_search[:avoid_scans] = 0

        # Permissive peak detection — selection happens below.
        # distance=30 avoids picking up two peaks within the same noise burst;
        # the narrowest real GS600 LIZ doublet (6 bp gap at small fragments)
        # gives at least ~40 scan separation, so real peaks are preserved.
        min_h = max(liz_clean.max() * 0.04, 3)
        candidates, _ = find_peaks(
            liz_search, height=min_h, distance=30, prominence=min_h * 0.2
        )

        if len(candidates) < 5:
            return None

        # Remove height outliers before ladder matching.
        # LIZ peaks are roughly uniform in amplitude, so peaks far outside
        # that range are artifacts:
        #   > 2× median → sample-channel bleed-through (very tall false peak)
        #   < 30% median → residual injection / primer-dimer artifact that
        #                  matrix correction didn't fully remove
        # Fragman's detect.ladder.R removes the tallest peak for the same reason.
        h = liz_clean[candidates]
        med = np.median(h)
        keep = (h >= med * 0.30) & (h <= med * 2)
        if keep.sum() < 5:
            return None
        
        # Ladder peaks are contiguous in time, so a tall peak sitting
        # *between* accepted peaks may be a sample peak overlapping a real
        # ladder peak.  Tall peaks outside that span (e.g. primer-dimer
        # bleed-through) are never kept.
        kept_idx = np.flatnonzero(keep)
        idx = np.arange(len(candidates))
        keep_wide = keep | ((idx > kept_idx[0]) & (idx < kept_idx[-1])
                            & (h > med * 2))
        cand_narrow = candidates[keep]
        cand_wide   = candidates[keep_wide]

        # Auto-detect ladder from StdF annotation when not supplied
        if std_sizes is None:
            if '60' in self.std_file and '600' in self.std_file:
                std_sizes = GS600_LIZ_60_600_SIZES
            else:
                std_sizes = GS600_LIZ_SIZES

        std_arr = np.array(std_sizes, dtype=float)

        # Re-admit the overlapped peaks only when the ladder is short of
        # peaks and doing so does not overshoot the expected count.
        if len(cand_narrow) < len(std_arr) and len(cand_wide) <= len(std_arr):
            candidates = cand_wide
        else:
            candidates = cand_narrow

        # Use correlation-based sliding-window search (mirrors Fragman's
        # "red" method) rather than height-based trimming, which fails when
        # a noise peak is taller than a real ladder peak.
        peaks, matched_sizes = _select_ladder_peaks(candidates, std_arr, liz_clean)

        if peaks is None or len(peaks) < 4:
            return None

        # Local Southern method: piecewise cubic spline through all
        # calibration points.  The spline interpolates exactly — at each
        # ladder peak scan the returned bp value equals the assigned ladder
        # size — so LIZ peaks in the PET channel align precisely with their
        # expected positions on a bp-axis plot.
        from scipy.interpolate import CubicSpline
        px = peaks.astype(float)
        try:
            cs        = CubicSpline(px, matched_sizes)
            bp_values = cs(self.scan_numbers.astype(float))
            bp_values = np.clip(bp_values, 0, 800).astype(np.float32)
            return bp_values
        except Exception:
            return None
