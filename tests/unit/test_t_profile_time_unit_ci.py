from __future__ import annotations

import numpy as np
import pytest

from kimeco.experiments.t_profile import TimeProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_csv(tmp_path, name, header, rows):
    """Write a small CSV file and return its path."""
    path = tmp_path / name
    lines = [','.join(header)]
    for row in rows:
        lines.append(','.join(str(v) for v in row))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(path)


def _make_time_profile(species, data, time_unit='s'):
    """Build a minimal TimeProfile without touching KMOLogger behaviour."""
    return TimeProfile(
        temp=500.0,
        pres=101325.0,
        composition={'O2': 1.0},
        data_file='data.csv',
        error_file='error.csv',
        sim_file='sim.inp',
        settings={},
        klog=None,          # type: ignore[arg-type]  # stored, never called
        species=species,
        data=np.asarray(data, dtype=float),
        error=np.zeros_like(np.asarray(data, dtype=float)),
        time_unit=time_unit,
    )


# ---------------------------------------------------------------------------
# 1) Valid header parsing + seconds normalization via read_data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'header, expected_unit, factor',
    [
        ('time', 's', 1.0),
        ('time[s]', 's', 1.0),
        ('TIME [ms]', 'ms', 1e-3),
        ('time[milliseconds]', 'milliseconds', 1e-3),
        ('time[millisecond]', 'millisecond', 1e-3),
        ('time[1e-3s]', '1e-3s', 1e-3),
        ('time[0.001 s]', '0.001 s', 1e-3),
        ('time[1E-3s]', '1E-3s', 1e-3),
        ('time[1e-3]', '1e-3', 1e-3),
        ('time[min]', 'min', 60.0),
        ('time[us]', 'us', 1e-6),
    ],
)
def test_read_data_valid_headers_scale_time_to_seconds(
        tmp_path, header, expected_unit, factor):
    raw_time = [1.0, 2.0, 3.0]
    sp_a = [10.0, 20.0, 30.0]
    sp_b = [0.5, 0.6, 0.7]
    rows = list(zip(raw_time, sp_a, sp_b))
    file = _write_csv(tmp_path, 'data.csv', [header, 'A', 'B'], rows)

    headers, matrix, time_unit = TimeProfile.read_data(file)

    # header[0] normalized to 'time', species headers untouched
    assert headers[0] == 'time'
    assert headers[1:] == ['A', 'B']
    # raw bracket content preserved as time_unit
    assert time_unit == expected_unit
    # time row converted to seconds
    np.testing.assert_allclose(matrix[0], np.array(raw_time) * factor)
    # species rows are NOT scaled
    np.testing.assert_allclose(matrix[1], sp_a)
    np.testing.assert_allclose(matrix[2], sp_b)


@pytest.mark.parametrize(
    'header, expected_is_time, expected_unit, expected_factor',
    [
        ('time', True, 's', 1.0),
        ('  time  ', True, 's', 1.0),
        ('Time', True, 's', 1.0),
        ('TIME', True, 's', 1.0),
        ('time[ ms ]', True, 'ms', 1e-3),
        ('TIME [ MS ]', True, 'MS', 1e-3),
        ('time[  1e-3  s ]', True, '1e-3  s', 1e-3),
        ('time[0.001 s]', True, '0.001 s', 1e-3),
        ('time[min]', True, 'min', 60.0),
        ('time[us]', True, 'us', 1e-6),
        ('foo', False, 's', 1.0),
    ],
)
def test_parse_time_header_whitespace_and_case_tolerance(
        header, expected_is_time, expected_unit, expected_factor):
    is_time, raw_unit, factor = TimeProfile._parse_time_header(header)
    assert is_time is expected_is_time
    assert raw_unit == expected_unit
    np.testing.assert_allclose(factor, expected_factor)


def test_parse_time_header_factor_only():
    is_time, raw_unit, factor = TimeProfile._parse_time_header('time[1e-3]')
    assert is_time is True
    assert raw_unit == '1e-3'
    np.testing.assert_allclose(factor, 1e-3)


def test_unit_to_seconds_known_units():
    np.testing.assert_allclose(TimeProfile._unit_to_seconds('ms'), 1e-3)
    np.testing.assert_allclose(
        TimeProfile._unit_to_seconds('millisecond'), 1e-3)
    np.testing.assert_allclose(TimeProfile._unit_to_seconds('s'), 1.0)
    np.testing.assert_allclose(TimeProfile._unit_to_seconds('min'), 60.0)
    np.testing.assert_allclose(TimeProfile._unit_to_seconds('us'), 1e-6)


# ---------------------------------------------------------------------------
# 2) Invalid / error paths
# ---------------------------------------------------------------------------
def test_read_data_first_column_not_time_raises_keyerror(tmp_path):
    file = _write_csv(
        tmp_path, 'data.csv', ['foo', 'A'], [(1.0, 2.0), (3.0, 4.0)])
    with pytest.raises(KeyError):
        TimeProfile.read_data(file)


def test_read_data_unknown_unit_raises_valueerror(tmp_path):
    file = _write_csv(
        tmp_path, 'data.csv', ['time[bananas]', 'A'], [(1.0, 2.0)])
    with pytest.raises(ValueError):
        TimeProfile.read_data(file)


def test_read_data_non_time_unit_raises_valueerror(tmp_path):
    file = _write_csv(
        tmp_path, 'data.csv', ['time[kg]', 'A'], [(1.0, 2.0)])
    with pytest.raises(ValueError):
        TimeProfile.read_data(file)


def test_read_data_empty_bracket_raises_valueerror(tmp_path):
    file = _write_csv(
        tmp_path, 'data.csv', ['time[]', 'A'], [(1.0, 2.0)])
    with pytest.raises(ValueError):
        TimeProfile.read_data(file)


def test_parse_time_header_empty_bracket_raises_valueerror():
    with pytest.raises(ValueError):
        TimeProfile._parse_time_header('time[]')


def test_parse_time_header_unknown_unit_raises_valueerror():
    with pytest.raises(ValueError):
        TimeProfile._parse_time_header('time[bananas]')


def test_parse_time_header_non_time_unit_raises_valueerror():
    with pytest.raises(ValueError):
        TimeProfile._parse_time_header('time[kg]')


def test_read_data_empty_header_raises_valueerror(tmp_path):
    path = tmp_path / 'empty.csv'
    path.write_text('', encoding='utf-8')
    with pytest.raises(ValueError):
        TimeProfile.read_data(str(path))


def test_read_data_non_float_value_raises_typeerror(tmp_path):
    file = _write_csv(
        tmp_path, 'data.csv', ['time', 'A'],
        [(1.0, 'not_a_number'), (2.0, 3.0)])
    with pytest.raises(TypeError):
        TimeProfile.read_data(file)


def test_read_data_header_only_no_rows_raises_valueerror(tmp_path):
    file = _write_csv(tmp_path, 'data.csv', ['time', 'A'], [])
    with pytest.raises(ValueError):
        TimeProfile.read_data(file)


def test_read_data_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        TimeProfile.read_data(str(tmp_path / 'does_not_exist.csv'))


# ---------------------------------------------------------------------------
# 3) exp.time property + time_unit attribute
# ---------------------------------------------------------------------------
def test_exp_time_property_returns_seconds_row():
    data = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]], dtype=float)
    exp = _make_time_profile(['A'], data, time_unit='ms')
    np.testing.assert_array_equal(exp.time, data[0])
    # property exposes row 0 of self.data (shares underlying memory)
    assert np.shares_memory(exp.time, exp.data)
    np.testing.assert_array_equal(exp.time, exp.data[0])


def test_time_unit_attribute_preserved():
    data = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    exp = _make_time_profile(['A'], data, time_unit='ms')
    assert exp.time_unit == 'ms'


def test_time_unit_defaults_to_seconds():
    data = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    exp = _make_time_profile(['A'], data)
    assert exp.time_unit == 's'


# ---------------------------------------------------------------------------
# 4) validate_pair
# ---------------------------------------------------------------------------
def test_validate_pair_same_unit_matches(tmp_path):
    rows = [(1.0, 5.0), (2.0, 6.0)]
    d = _write_csv(tmp_path, 'd.csv', ['time[ms]', 'A'], rows)
    e = _write_csv(tmp_path, 'e.csv', ['time[ms]', 'A'], rows)
    dh, dm, _ = TimeProfile.read_data(d)
    eh, em, _ = TimeProfile.read_data(e)
    TimeProfile.validate_pair(dh, dm, eh, em, d, e)  # no raise


def test_validate_pair_mixed_unit_equivalent_seconds_matches(tmp_path):
    # 1 ms == 1000 us -> same seconds grid
    d = _write_csv(tmp_path, 'd.csv', ['time[ms]', 'A'],
                   [(1.0, 5.0), (2.0, 6.0)])
    e = _write_csv(tmp_path, 'e.csv', ['time[us]', 'A'],
                   [(1000.0, 5.0), (2000.0, 6.0)])
    dh, dm, _ = TimeProfile.read_data(d)
    eh, em, _ = TimeProfile.read_data(e)
    TimeProfile.validate_pair(dh, dm, eh, em, d, e)  # no raise


def test_validate_pair_mixed_unit_non_matching_raises(tmp_path):
    d = _write_csv(tmp_path, 'd.csv', ['time[ms]', 'A'],
                   [(1.0, 5.0), (2.0, 6.0)])
    e = _write_csv(tmp_path, 'e.csv', ['time[us]', 'A'],
                   [(1.0, 5.0), (2.0, 6.0)])
    dh, dm, _ = TimeProfile.read_data(d)
    eh, em, _ = TimeProfile.read_data(e)
    with pytest.raises(ValueError):
        TimeProfile.validate_pair(dh, dm, eh, em, d, e)


def test_validate_pair_header_mismatch_raises(tmp_path):
    d = _write_csv(tmp_path, 'd.csv', ['time', 'A'], [(1.0, 5.0)])
    e = _write_csv(tmp_path, 'e.csv', ['time', 'B'], [(1.0, 5.0)])
    dh, dm, _ = TimeProfile.read_data(d)
    eh, em, _ = TimeProfile.read_data(e)
    with pytest.raises(ValueError):
        TimeProfile.validate_pair(dh, dm, eh, em, d, e)


def test_validate_pair_shape_mismatch_raises(tmp_path):
    d = _write_csv(tmp_path, 'd.csv', ['time', 'A'],
                   [(1.0, 5.0), (2.0, 6.0)])
    e = _write_csv(tmp_path, 'e.csv', ['time', 'A'],
                   [(1.0, 5.0), (2.0, 6.0), (3.0, 7.0)])
    dh, dm, _ = TimeProfile.read_data(d)
    eh, em, _ = TimeProfile.read_data(e)
    with pytest.raises(ValueError):
        TimeProfile.validate_pair(dh, dm, eh, em, d, e)


def test_validate_pair_allclose_tolerates_tiny_difference():
    headers = ['time', 'A']
    data = np.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]], dtype=float)
    error = data.copy()
    error[0] = error[0] + 1e-12
    TimeProfile.validate_pair(headers, data, headers, error,
                              'd.csv', 'e.csv')  # no raise


# ---------------------------------------------------------------------------
# 5) Backward compatibility
# ---------------------------------------------------------------------------
def test_read_data_plain_time_backward_compat(tmp_path):
    rows = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    file = _write_csv(tmp_path, 'data.csv', ['time', 'A'], rows)
    headers, matrix, time_unit = TimeProfile.read_data(file)
    assert headers == ['time', 'A']
    assert time_unit == 's'
    np.testing.assert_allclose(matrix[0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(matrix[1], [1.0, 2.0, 3.0])
