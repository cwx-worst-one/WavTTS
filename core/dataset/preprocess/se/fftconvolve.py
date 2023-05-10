from numbers import Number
import operator
import torch
import torch.fft as sp_fft
from scipy.fftpack import next_fast_len


def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int

    Parameters
    ----------
    x : value, or sequence of values, convertible to int
    name : str, optional
        Name of the argument being converted, only used in the error message

    Returns
    -------
    y : ``List[int]``
    """
    if isinstance(x, Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or "value"
        raise ValueError("{} must be a scalar or iterable of integers".format(name)) from e

    return x


def _init_nd_shape_and_axes(x, shape, axes):
    """Handles shape and axes arguments for nd transforms"""
    noshape = shape is None
    noaxes = axes is None

    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]

        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")

    if not noshape:
        shape = _iterable_of_int(shape, 'shape')

        if axes and len(axes) != len(shape):
            raise ValueError(
                "when given, axes and shape arguments" " have to be of the same length"
            )
        if noaxes:
            if len(shape) > x.ndim:
                raise ValueError("shape requires more axes than are present")
            axes = range(x.ndim - len(shape), x.ndim)

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]

    if any(s < 1 for s in shape):
        raise ValueError("invalid number of data points ({0}) specified".format(shape))

    return shape, axes


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """Determine if inputs arrays need to be swapped in `"valid"` mode.
    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.
    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.
    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode != 'valid':
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError(
            "For 'valid' mode, one must be at least " "as large as the other in every dimension"
        )

    return not ok1


def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    """Handle the axes argument for frequency-domain convolution.

    Returns the inputs and axes in a standard form, eliminating redundant axes,
    swapping the inputs if necessary, and checking for various potential
    errors.

    Parameters
    ----------
    in1 : array
        First input.
    in2 : array
        Second input.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the FFTs.
    sorted_axes : bool, optional
        If `True`, sort the axes.
        Default is `False`, do not sort.

    Returns
    -------
    in1 : array
        The first input, possible swapped with the second input.
    in2 : array
        The second input, possible swapped with the first input.
    axes : list of ints
        Axes over which to compute the FFTs.

    """
    s1 = in1.shape
    s2 = in2.shape
    noaxes = axes is None

    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)

    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")

    # Axes of length 1 can rely on broadcasting rules for multipy,
    # no fft needed.
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

    if sorted_axes:
        axes.sort()

    if not all(
        s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1 for a in range(in1.ndim) if a not in axes
    ):
        raise ValueError("incompatible shapes for in1 and in2:" " {0} and {1}".format(s1, s2))

    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # Convolution is commutative; order doesn't have any effect on output.
        in1, in2 = in2, in1

    return in1, in2, axes


def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    if not len(axes):
        return in1 * in2

    complex_result = False

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [next_fast_len(shape[a]) for a in axes]
    else:
        fshape = shape

    fft, ifft = sp_fft.rfftn, sp_fft.irfftn

    sp1 = fft(in1, fshape, dim=axes)
    sp2 = fft(in2, fshape, dim=axes)

    ret = ifft(sp1 * sp2, fshape, dim=axes)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = torch.tensor(newshape)
    currshape = torch.tensor(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Calculate the convolution result shape based on the `mode` argument.
    Returns the result sliced to the correct size for the given mode.
    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.
    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.
    """
    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid)
    else:
        raise ValueError("acceptable mode flags are 'valid'," " 'same', or 'full'")


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    """
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return None

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)
