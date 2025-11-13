"""
This module implements the Residue function and related tools for working
with residues.
"""

from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.bessel import besselj
from sympy.concrete.summations import Sum
from sympy.core.numbers import oo
from sympy.utilities.timeutils import timethis


class SeriesCoefficient(Function):
    nargs = 3

    @classmethod
    def eval(cls, expr, sym, n):
        expr = sympify(expr)
        sym = sympify(sym)
        n = sympify(n)
        if n.is_Integer:
            if n.is_nonnegative:
                return expr.series(sym, 0, int(n) + 1).removeO().coeff(sym, int(n))
            return S.Zero


def _residue_exp_laurent(expr, x):
    if expr.func is not exp:
        return None
    arg = expr.exp.expand()
    coeff_inv = S.Zero
    coeff_x = S.Zero
    const = S.Zero
    pos_poly = S.Zero
    for term in Add.make_args(arg):
        coeff, exponent = term.as_coeff_exponent(x)
        if exponent == 0:
            const += coeff
        elif exponent == 1:
            coeff_x += coeff
        elif exponent == -1:
            coeff_inv += coeff
        elif exponent.is_Integer and exponent >= 2:
            pos_poly += coeff * x**exponent
        else:
            return None
    if coeff_inv == 0:
        return None
    prefactor = exp(const)
    if pos_poly == 0:
        if coeff_x == 0:
            return exp(const) * coeff_inv
        beta = sqrt(-coeff_x*coeff_inv)
        return exp(const) * (-beta/coeff_x) * besselj(1, 2*beta)

    analytic = exp(pos_poly)
    m = Dummy('m', integer=True, nonnegative=True)
    coeff_series = SeriesCoefficient(analytic, x, m)
    if coeff_x == 0:
        term = coeff_series * coeff_inv**(m + 1) / factorial(m + 1)
    else:
        beta = sqrt(-coeff_x*coeff_inv)
        ratio = beta/coeff_x
        term = coeff_series * (-1)**(m + 1) * ratio**(m + 1) * besselj(m + 1, 2*beta)
    return prefactor * Sum(term, (m, 0, oo))


@timethis('residue')
def residue(expr, x, x0):
    """
    Finds the residue of ``expr`` at the point x=x0.

    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series
    expansion about ``x=x0``.

    Examples
    ========

    >>> from sympy import Symbol, residue, sin
    >>> x = Symbol("x")
    >>> residue(1/x, x, 0)
    1
    >>> residue(1/x**2, x, 0)
    0
    >>> residue(2/sin(x), x, 0)
    2

    This function is essential for the Residue Theorem [1].

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Residue_theorem
    """
    # The current implementation uses series expansion to
    # calculate it. A more general implementation is explained in
    # the section 5.6 of the Bronstein's book {M. Bronstein:
    # Symbolic Integration I, Springer Verlag (2005)}. For purely
    # rational functions, the algorithm is much easier. See
    # sections 2.4, 2.5, and 2.7 (this section actually gives an
    # algorithm for computing any Laurent series coefficient for
    # a rational function). The theory in section 2.4 will help to
    # understand why the resultant works in the general algorithm.
    # For the definition of a resultant, see section 1.4 (and any
    # previous sections for more review).

    from sympy.series.order import Order
    from sympy.simplify.radsimp import collect
    expr = sympify(expr)
    if x0 != 0:
        expr = expr.subs(x, x + x0)
    special = _residue_exp_laurent(expr, x)
    if special is not None:
        return special

    for n in (0, 1, 2, 4, 8, 16, 32):
        s = expr.nseries(x, n=n)
        if not s.has(Order) or s.getn() >= 0:
            break
    s = collect(s.removeO(), x)
    if s.is_Add:
        args = s.args
    else:
        args = [s]
    res = S.Zero
    for arg in args:
        c, m = arg.as_coeff_mul(x)
        m = Mul(*m)
        if not (m in (S.One, x) or (m.is_Pow and m.exp.is_Integer)):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1/x:
            res += c
    return res
