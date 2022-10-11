from math import sqrt
import re
import dis
import numbers
import os
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp

import numba.parfors.parfor
from numba import (njit, prange, set_num_threads, get_num_threads, typeof)
from numba.core import (types, utils, typing, errors, ir, rewrites,
                        typed_passes, inline_closurecall, config, compiler, cpu)
from numba.extending import (overload_method, register_model,
                             typeof_impl, unbox, NativeValue, models)
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
                            get_definition, is_getitem, is_setitem,
                            index_var_of_get_setitem)
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.bytecode import ByteCodeIter
from numba.core.compiler import (compile_isolated, Flags, CompilerBase,
                                 DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
                      override_env_config, linux_only, tag,
                      skip_parfors_unsupported, _32bit, needs_blas,
                      needs_lapack, disabled_test, skip_unless_scipy,
                      needs_subprocess)
from numba.core.extending import register_jitable
import cmath
import unittest
import dpctl

# NOTE: Each parfors test class is run in separate subprocess, this is to reduce
# memory pressure in CI settings. The environment variable "SUBPROC_TEST" is
# used to determine whether a test is skipped or not, such that if you want to
# run any parfors test directly this environment variable can be set. The
# subprocesses running the test classes set this environment variable as the new
# process starts which enables the tests within the process. The decorator
# @needs_subprocess is used to ensure the appropriate test skips are made.


@skip_parfors_unsupported
class TestParforsRunner(TestCase):

    _numba_parallel_test_ = False

    # Each test class can run for 30 minutes before time out. Extend this to an
    # hour on aarch64 (some public CI systems were timing out).
    _TIMEOUT = 1800 if platform.machine() != 'aarch64' else 3600

    """This is the test runner for all the parfors tests, it runs them in
    subprocesses as described above. The convention for the test method naming
    is: `test_<TestClass>` where <TestClass> is the name of the test class in
    this module.
    """
    def runner(self):
        themod = self.__module__
        test_clazz_name = self.id().split('.')[-1].split('_')[-1]
        # don't specify a given test, it's an entire class that needs running
        self.subprocess_test_runner(test_module=themod,
                                    test_class=test_clazz_name,
                                    timeout=self._TIMEOUT)

    def test_TestParforBasic(self):
        self.runner()

    def test_TestParforNumericalMisc(self):
        self.runner()

    def test_TestParforNumPy(self):
        self.runner()

    def test_TestParfors(self):
        self.runner()

    def test_TestParforsBitMask(self):
        self.runner()

    def test_TestParforsDiagnostics(self):
        self.runner()

    def test_TestParforsLeaks(self):
        self.runner()

    def test_TestParforsMisc(self):
        self.runner()

    def test_TestParforsOptions(self):
        self.runner()

    def test_TestParforsSlice(self):
        self.runner()

    def test_TestParforsVectorizer(self):
        self.runner()

    def test_TestPrangeBasic(self):
        self.runner()

    def test_TestPrangeSpecific(self):
        self.runner()


x86_only = unittest.skipIf(platform.machine() not in ('i386', 'x86_64'), 'x86 only test')

_GLOBAL_INT_FOR_TESTING1 = 17
_GLOBAL_INT_FOR_TESTING2 = 5

TestNamedTuple = namedtuple('TestNamedTuple', ('part0', 'part1'))

def null_comparer(a, b):
    """
    Used with check_arq_equality to indicate that we do not care
    whether the value of the parameter at the end of the function
    has a particular value.
    """
    pass


@needs_subprocess
class TestParforsBase(TestCase):
    """
    Base class for testing parfors.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and parfor njit'd functions.
    """

    _numba_parallel_test_ = False

    def __init__(self, *args, **kwargs):
        enable_parallel = kwargs.pop('enable_parallel', True)

        # flags for njit()
        self.cflags = Flags()
        self.cflags.nrt = True

        # flags for njit(parallel=True)
        self.pflags = Flags()
        self.pflags.auto_parallel = cpu.ParallelOptions(enable_parallel)
        self.pflags.nrt = True

        # flags for njit(parallel=True, fastmath=True)
        self.fast_pflags = Flags()
        self.fast_pflags.auto_parallel = cpu.ParallelOptions(enable_parallel)
        self.fast_pflags.nrt = True
        self.fast_pflags.fastmath = cpu.FastMathOptions(True)

        self.device = dpctl.select_default_device()
        # print("Using device ...")
        self.device.print_device_info()

        super(TestParforsBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_isolated(func, sig, flags=flags)

    def compile_parallel(self, func, sig):
        return self._compile_this(func, sig, flags=self.pflags)

    def compile_parallel_fastmath(self, func, sig):
        return self._compile_this(func, sig, flags=self.fast_pflags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])
        # print("---------- sig ----------")
        # print(sig)

        # compile the prange injected function
        cpfunc = self.compile_parallel(pyfunc, sig)

        # compile a standard njit of the original function
        cfunc = self.compile_njit(pyfunc, sig)

        return cfunc, cpfunc

    def check_parfors_vs_others(self, pyfunc, cfunc, cpfunc, *args, **kwargs):
        """
        Checks python, njit and parfor impls produce the same result.

        Arguments:
            pyfunc - the python function to test
            cfunc - CompilerResult from njit of pyfunc
            cpfunc - CompilerResult from njit(parallel=True) of pyfunc
            args - arguments for the function being tested
        Keyword Arguments:
            scheduler_type - 'signed', 'unsigned' or None, default is None.
                           Supply in cases where the presence of a specific
                           scheduler is to be asserted.
            fastmath_pcres - a fastmath parallel compile result, if supplied
                             will be run to make sure the result is correct
            check_arg_equality - some functions need to check that a
                                 parameter is modified rather than a certain
                                 value returned.  If this keyword argument
                                 is supplied, it should be a list of
                                 comparison functions such that the i'th
                                 function in the list is used to compare the
                                 i'th parameter of the njit and parallel=True
                                 functions against the i'th parameter of the
                                 standard Python function, asserting if they
                                 differ.  The length of this list must be equal
                                 to the number of parameters to the function.
                                 The null comparator is available for use
                                 when you do not desire to test if some
                                 particular parameter is changed.
            Remaining kwargs are passed to np.testing.assert_almost_equal
        """
        scheduler_type = kwargs.pop('scheduler_type', None)
        check_fastmath = kwargs.pop('check_fastmath', None)
        fastmath_pcres = kwargs.pop('fastmath_pcres', None)
        check_scheduling = kwargs.pop('check_scheduling', True)
        check_args_for_equality = kwargs.pop('check_arg_equality', None)

        def copy_args(*args):
            if not args:
                return tuple()
            new_args = []
            for x in args:
                if isinstance(x, np.ndarray):
                    new_args.append(x.copy('k'))
                elif isinstance(x, np.number):
                    new_args.append(x.copy())
                elif isinstance(x, numbers.Number):
                    new_args.append(x)
                elif x is None:
                    new_args.append(x)
                elif isinstance(x, tuple):
                    new_args.append(copy.deepcopy(x))
                elif isinstance(x, list):
                    new_args.append(x[:])
                else:
                    raise ValueError('Unsupported argument type encountered')
            return tuple(new_args)

        # python result
        print("*args:", *args)
        py_args = copy_args(*args)
        py_expected = pyfunc(*py_args)
        print("py_expected:", py_expected)

        # njit result
        njit_args = copy_args(*args)
        njit_output = cfunc.entry_point(*njit_args)
        print("njit_output:", njit_output)

        # parfor result
        parfor_args = copy_args(*args)
        parfor_output = cpfunc.entry_point(*parfor_args)
        print("parfor_output:", parfor_output)

        # numba_dpex_result
        # TODO: what to do when we need to have a data parallel operations?
        numba_dpex_args = copy_args(*args)
        with dpctl.device_context(self.device):
            numba_dpex_output = cfunc.entry_point(*numba_dpex_args)
            # numba_dpex_output = cpfunc.entry_point(*numba_dpex_args) # do we need to test both?
        print("numba_dpex_output:", numba_dpex_output)

        if check_args_for_equality is None:
            np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
            np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)
            np.testing.assert_almost_equal(numba_dpex_output, py_expected, **kwargs)
            self.assertEqual(type(njit_output), type(parfor_output))
            self.assertEqual(type(parfor_output), type(numba_dpex_output))
        else:
            assert(len(py_args) == len(check_args_for_equality))
            for pyarg, njitarg, parforarg, numbadpexarg, argcomp in zip(
                py_args, njit_args, parfor_args, numba_dpex_args, check_args_for_equality):
                argcomp(njitarg, pyarg, **kwargs)
                argcomp(parforarg, pyarg, **kwargs)
                argcomp(numbadpexarg, pyarg, **kwargs)

        if check_scheduling:
            self.check_scheduling(cpfunc, scheduler_type)

        # if requested check fastmath variant
        if fastmath_pcres is not None:
            parfor_fastmath_output = fastmath_pcres.entry_point(*copy_args(*args))
            np.testing.assert_almost_equal(parfor_fastmath_output, py_expected,
                                           **kwargs)

    def check(self, pyfunc, *args, **kwargs):
        """Checks that pyfunc compiles for *args under parallel=True and njit
        and asserts that all version execute and produce the same result"""
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def check_variants(self, impl, arg_gen, **kwargs):
        """Run self.check(impl, ...) on array data generated from arg_gen.
        """
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                self.check(impl, *args, **kwargs)

    def count_parfors_variants(self, impl, arg_gen, **kwargs):
        """Run self.countParfors(impl, ...) on array types generated from
        arg_gen.
        """
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                argtys = tuple(map(typeof, args))
                # At least one parfors
                self.assertGreaterEqual(countParfors(impl, argtys), 1)

    def check_scheduling(self, cres, scheduler_type):
        # make sure parfor set up scheduling
        scheduler_str = '@do_scheduling'
        if scheduler_type is not None:
            if scheduler_type in ['signed', 'unsigned']:
                scheduler_str += '_' + scheduler_type
            else:
                msg = "Unknown scheduler_type specified: %s"
                raise ValueError(msg % scheduler_type)

        self.assertIn(scheduler_str, cres.library.get_llvm_str())

    def gen_linspace(self, n, ct):
        """Make *ct* sample 1D arrays of length *n* using np.linspace().
        """
        def gen():
            yield np.linspace(0, 1, n)
            yield np.linspace(2, 1, n)
            yield np.linspace(1, 2, n)

        src = cycle(gen())
        return [next(src) for i in range(ct)]

    def gen_linspace_variants(self, ct):
        """Make 1D, 2D, 3D variants of the data in C and F orders
        """
        # 1D
        yield self.gen_linspace(10, ct=ct)

        # 2D
        arr2ds = [x.reshape((2, 3))
                  for x in self.gen_linspace(n=2 * 3, ct=ct)]
        yield arr2ds
        # Fortran order
        yield [np.asfortranarray(x) for x in arr2ds]

        # 3D
        arr3ds = [x.reshape((2, 3, 4))
                  for x in self.gen_linspace(n=2 * 3 * 4, ct=ct)]
        yield arr3ds
        # Fortran order
        yield [np.asfortranarray(x) for x in arr3ds]

    def _filter_mod(self, mod, magicstr, checkstr=None):
        """ helper function to filter out modules by name"""
        filt = [x for x in mod if magicstr in x.name]
        if checkstr is not None:
            for x in filt:
                assert checkstr in str(x)
        return filt

    def _get_gufunc_modules(self, cres, magicstr, checkstr=None):
        """ gets the gufunc LLVM Modules"""
        _modules = [x for x in cres.library._codegen._engine._ee._modules]
        return self._filter_mod(_modules, magicstr, checkstr=checkstr)

    def _get_gufunc_info(self, cres, fn):
        """ helper for gufunc IR/asm generation"""
        # get the gufunc modules
        magicstr = '__numba_parfor_gufunc'
        gufunc_mods = self._get_gufunc_modules(cres, magicstr)
        x = dict()
        for mod in gufunc_mods:
            x[mod.name] = fn(mod)
        return x

    def _get_gufunc_ir(self, cres):
        """
        Returns the IR of the gufuncs used as parfor kernels
        as a dict mapping the gufunc name to its IR.

        Arguments:
         cres - a CompileResult from `njit(parallel=True, ...)`
        """
        return self._get_gufunc_info(cres, str)

    def _get_gufunc_asm(self, cres):
        """
        Returns the assembly of the gufuncs used as parfor kernels
        as a dict mapping the gufunc name to its assembly.

        Arguments:
         cres - a CompileResult from `njit(parallel=True, ...)`
        """
        tm = cres.library._codegen._tm
        def emit_asm(mod):
            return str(tm.emit_assembly(mod))
        return self._get_gufunc_info(cres, emit_asm)

    def assert_fastmath(self, pyfunc, sig):
        """
        Asserts that the fastmath flag has some effect in that suitable
        instructions are now labelled as `fast`. Whether LLVM can actually do
        anything to optimise better now the derestrictions are supplied is
        another matter!

        Arguments:
         pyfunc - a function that contains operations with parallel semantics
         sig - the type signature of pyfunc
        """

        cres = self.compile_parallel_fastmath(pyfunc, sig)
        _ir = self._get_gufunc_ir(cres)

        def _get_fast_instructions(ir):
            splitted = ir.splitlines()
            fast_inst = []
            for x in splitted:
                m = re.search(r'\bfast\b', x)  # \b for wholeword
                if m is not None:
                    fast_inst.append(x)
            return fast_inst

        def _assert_fast(instrs):
            ops = ('fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp')
            for inst in instrs:
                count = 0
                for op in ops:
                    match = op + ' fast'
                    if match in inst:
                        count += 1
                self.assertTrue(count > 0)

        for name, guir in _ir.items():
            inst = _get_fast_instructions(guir)
            _assert_fast(inst)


def blackscholes_impl(sptprice, strike, rate, volatility, timev):
    # blackscholes example
    logterm = np.log(sptprice / strike)
    powterm = 0.5 * volatility * volatility
    den = volatility * np.sqrt(timev)
    d1 = (((rate + powterm) * timev) + logterm) / den
    d2 = d1 - den
    NofXd1 = 0.5 + 0.5 * 2.0 * d1
    NofXd2 = 0.5 + 0.5 * 2.0 * d2
    futureValue = strike * np.exp(- rate * timev)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    put = call - futureValue + sptprice
    return put


def lr_impl(Y, X, w, iterations):
    # logistic regression example
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
    return w

def example_kmeans_test(A, numCenter, numIter, init_centroids):
    centroids = init_centroids
    N, D = A.shape

    for l in range(numIter):
        dist = np.array([[sqrt(np.sum((A[i,:]-centroids[j,:])**2))
                                for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i,:].argmin() for i in range(N)])

        centroids = np.array([[np.sum(A[labels==i, j])/np.sum(labels==i)
                                 for j in range(D)] for i in range(numCenter)])

    return centroids

def get_optimized_numba_ir(test_func, args, **kws):
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx, 'cpu')
    test_ir = compiler.run_frontend(test_func)
    if kws:
        options = cpu.ParallelOptions(kws)
    else:
        options = cpu.ParallelOptions(True)

    tp = TestPipeline(typingctx, targetctx, args, test_ir)

    with cpu_target.nested_context(typingctx, targetctx):
        typingctx.refresh()
        targetctx.refresh()

        inline_pass = inline_closurecall.InlineClosureCallPass(tp.state.func_ir,
                                                               options,
                                                               typed=True)
        inline_pass.run()

        rewrites.rewrite_registry.apply('before-inference', tp.state)

        tp.state.typemap, tp.state.return_type, tp.state.calltypes, _ = \
        typed_passes.type_inference_stage(tp.state.typingctx,
            tp.state.targetctx, tp.state.func_ir, tp.state.args, None)

        type_annotations.TypeAnnotation(
            func_ir=tp.state.func_ir,
            typemap=tp.state.typemap,
            calltypes=tp.state.calltypes,
            lifted=(),
            lifted_from=None,
            args=tp.state.args,
            return_type=tp.state.return_type,
            html_output=config.HTML)

        diagnostics = numba.parfors.parfor.ParforDiagnostics()

        preparfor_pass = numba.parfors.parfor.PreParforPass(
            tp.state.func_ir, tp.state.typemap, tp.state.calltypes,
            tp.state.typingctx, tp.state.targetctx, options,
            swapped=diagnostics.replaced_fns)
        preparfor_pass.run()

        rewrites.rewrite_registry.apply('after-inference', tp.state)

        flags = compiler.Flags()
        parfor_pass = numba.parfors.parfor.ParforPass(
            tp.state.func_ir, tp.state.typemap, tp.state.calltypes,
            tp.state.return_type, tp.state.typingctx, tp.state.targetctx,
            options, flags, tp.state.metadata, diagnostics=diagnostics)
        parfor_pass.run()
        test_ir._definitions = build_definitions(test_ir.blocks)

    return test_ir, tp

def countParfors(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0

    for label, block in test_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                ret_count += 1

    return ret_count


def countArrays(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    return _count_arrays_inner(test_ir.blocks, tp.state.typemap)

def get_init_block_size(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    blocks = test_ir.blocks

    ret_count = 0

    for label, block in blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                ret_count += len(inst.init_block.body)

    return ret_count

def _count_arrays_inner(blocks, typemap):
    ret_count = 0
    arr_set = set()

    for label, block in blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                parfor_blocks = inst.loop_body.copy()
                parfor_blocks[0] = inst.init_block
                ret_count += _count_arrays_inner(parfor_blocks, typemap)
            if (isinstance(inst, ir.Assign)
                    and isinstance(typemap[inst.target.name],
                                    types.ArrayCompatible)):
                arr_set.add(inst.target.name)

    ret_count += len(arr_set)
    return ret_count

def countArrayAllocs(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0

    for block in test_ir.blocks.values():
        ret_count += _count_array_allocs_inner(test_ir, block)

    return ret_count

def _count_array_allocs_inner(func_ir, block):
    ret_count = 0
    for inst in block.body:
        if isinstance(inst, numba.parfors.parfor.Parfor):
            ret_count += _count_array_allocs_inner(func_ir, inst.init_block)
            for b in inst.loop_body.values():
                ret_count += _count_array_allocs_inner(func_ir, b)

        if (isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr)
                and inst.value.op == 'call'
                and (guard(find_callname, func_ir, inst.value) == ('empty', 'numpy')
                or guard(find_callname, func_ir, inst.value)
                    == ('empty_inferred', 'numba.np.unsafe.ndarray'))):
            ret_count += 1

    return ret_count

def countNonParforArrayAccesses(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    return _count_non_parfor_array_accesses_inner(test_ir, test_ir.blocks,
                                                  tp.state.typemap)

def _count_non_parfor_array_accesses_inner(f_ir, blocks, typemap, parfor_indices=None):
    ret_count = 0
    if parfor_indices is None:
        parfor_indices = set()

    for label, block in blocks.items():
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                parfor_indices.add(stmt.index_var.name)
                parfor_blocks = stmt.loop_body.copy()
                parfor_blocks[0] = stmt.init_block
                ret_count += _count_non_parfor_array_accesses_inner(
                    f_ir, parfor_blocks, typemap, parfor_indices)

            # getitem
            elif (is_getitem(stmt) and isinstance(typemap[stmt.value.value.name],
                        types.ArrayCompatible) and not _uses_indices(
                        f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1

            # setitem
            elif (is_setitem(stmt) and isinstance(typemap[stmt.target.name],
                    types.ArrayCompatible) and not _uses_indices(
                    f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1

            # find parfor_index aliases
            elif (isinstance(stmt, ir.Assign) and
                  isinstance(stmt.value, ir.Var) and
                  stmt.value.name in parfor_indices):
                parfor_indices.add(stmt.target.name)

    return ret_count

def _uses_indices(f_ir, index, index_set):
    if index.name in index_set:
        return True

    ind_def = guard(get_definition, f_ir, index)
    if isinstance(ind_def, ir.Expr) and ind_def.op == 'build_tuple':
        varnames = set(v.name for v in ind_def.items)
        return len(varnames & index_set) != 0

    return False


class TestPipeline(object):
    def __init__(self, typingctx, targetctx, args, test_ir):
        self.state = compiler.StateDict()
        self.state.typingctx = typingctx
        self.state.targetctx = targetctx
        self.state.args = args
        self.state.func_ir = test_ir
        self.state.typemap = None
        self.state.return_type = None
        self.state.calltypes = None
        self.state.metadata = {}


@skip_parfors_unsupported
class TestParforBasic(TestParforsBase):
    """Smoke tests for the parfors transforms. These tests check the most basic
    functionality"""

    def __init__(self, *args, **kwargs):
        TestParforsBase.__init__(self, *args, **kwargs)
        # these are used in the mass of simple tests
        m = np.reshape(np.arange(12.), (3, 4))
        self.simple_args = [np.arange(3.), np.arange(4.), m, m.T]

    def test_simple01(self):
        def test_impl():
            return np.ones(())
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    def test_simple02(self):
        def test_impl():
            return np.ones((1,))
        self.check(test_impl)

    def test_simple03(self):
        def test_impl():
            return np.ones((1, 2))
        self.check(test_impl)

    def test_simple04(self):
        def test_impl():
            return np.ones(1)
        self.check(test_impl)

    def test_simple07(self):
        def test_impl():
            return np.ones((1, 2), dtype=np.complex128)
        self.check(test_impl)

    def test_simple08(self):
        def test_impl():
            return np.ones((1, 2)) + np.ones((1, 2))
        self.check(test_impl)

    def test_simple09(self):
        def test_impl():
            return np.ones((1, 1))
        self.check(test_impl)

    def test_simple10(self):
        def test_impl():
            return np.ones((0, 0))
        self.check(test_impl)

    def test_simple11(self):
        def test_impl():
            return np.ones((10, 10)) + 1.
        self.check(test_impl)

    def test_simple12(self):
        def test_impl():
            return np.ones((10, 10)) + np.complex128(1.)
        self.check(test_impl)

    def test_simple13(self):
        def test_impl():
            return np.complex128(1.)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    def test_simple14(self):
        def test_impl():
            return np.ones((10, 10))[0::20]
        self.check(test_impl)

    def test_simple15(self):
        def test_impl(v1, v2, m1, m2):
            return v1 + v1
        self.check(test_impl, *self.simple_args)

    def test_simple16(self):
        def test_impl(v1, v2, m1, m2):
            return m1 + m1
        self.check(test_impl, *self.simple_args)

    def test_simple17(self):
        def test_impl(v1, v2, m1, m2):
            return m2 + v1
        self.check(test_impl, *self.simple_args)

    # for blas and lapack functions, we need scipy
    # TODO: update README.md to conda install scipy
    @needs_lapack
    def test_simple18(self):
        def test_impl(v1, v2, m1, m2):
            return m1.T + np.linalg.svd(m2)[1]
        self.check(test_impl, *self.simple_args)

    @needs_blas
    def test_simple19(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, v2)
        self.check(test_impl, *self.simple_args)

    @needs_blas
    def test_simple20(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, m2)
        # gemm is left to BLAS
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, *self.simple_args)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @needs_blas
    def test_simple21(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(v1, v1)
        self.check(test_impl, *self.simple_args)

    def test_simple22(self):
        def test_impl(v1, v2, m1, m2):
            return np.sum(v1 + v1)
        self.check(test_impl, *self.simple_args)

    def test_simple23(self):
        def test_impl(v1, v2, m1, m2):
            x = 2 * v1
            y = 2 * v1
            return 4 * np.sum(x**2 + y**2 < 1) / 10
        self.check(test_impl, *self.simple_args)

    def test_simple24(self):
        def test_impl():
            n = 20
            A = np.ones((n, n))
            b = np.arange(n)
            return np.sum(A[:, b])
        self.check(test_impl)

    # TODO: disabled_tests are not tested yet
    @disabled_test
    def test_simple_operator_15(self):
        """same as corresponding test_simple_<n> case but using operator.add"""
        def test_impl(v1, v2, m1, m2):
            return operator.add(v1, v1)

        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_16(self):
        def test_impl(v1, v2, m1, m2):
            return operator.add(m1, m1)

        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_17(self):
        def test_impl(v1, v2, m1, m2):
            return operator.add(m2, v1)

        self.check(test_impl, *self.simple_args)

    # TDOD: this fails, need to fix, issue 7201
    def test_inplace_alias(self):
        # issue7201
        def test_impl(a):
            a += 1
            a[:] = 3

        def comparer(a, b):
            np.testing.assert_equal(a, b)

        x = np.ones(1)
        self.check(test_impl, x, check_arg_equality=[comparer])

# TODO: get rid of main
if __name__ == "__main__":
    tpfb = TestParforBasic()
    tpfb.test_simple01()
    tpfb.test_simple02()
    tpfb.test_simple03()
    tpfb.test_simple04()
    tpfb.test_simple07()
    tpfb.test_simple08()
    tpfb.test_simple09()
    tpfb.test_simple10()
    tpfb.test_simple11()
    tpfb.test_simple12()
    tpfb.test_simple13()
    tpfb.test_simple14()
    tpfb.test_simple15()
    tpfb.test_simple16()
    tpfb.test_simple17()
    tpfb.test_simple18() # Needs scipy 1.0+
    tpfb.test_simple19()
    tpfb.test_simple20()
    tpfb.test_simple21()
    tpfb.test_simple22()
    tpfb.test_simple23()
    tpfb.test_simple24()
    # tpfb.test_inplace_alias() # fails
    
    
    
    
    
    
    
    
    
