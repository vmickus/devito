from devito import Eq
from devito.ir.clusters import Toposort
from devito.ir.equations import ClusterizedEq
from devito.ir.iet import Call, Expression, List, find_affine_trees
from devito.ir.iet.visitors import FindSymbols, Transformer
from devito.logger import warning
from devito.operator import Operator
from devito.ops import ops_configuration
from devito.ops.types import OpsBlock
from devito.ops.transformer import (create_ops_dat, create_ops_memory_fetch,
                                    create_ops_memory_set, opsit)
from devito.ops.utils import namespace
from devito.passes.clusters import Lift, fuse, scalarize, eliminate_arrays, rewrite
from devito.passes.iet import DataManager, iet_pass
from devito.symbolics import Literal
from devito.tools import filter_sorted, flatten, generator, timed_pass

__all__ = ['OPSOperator']


class OPSOperator(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    _default_headers = Operator._default_headers + ['#define restrict __restrict']
    _use_ops = True

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        # TODO: this is currently identical to CPU64NoopOperator._specialize_clusters,
        # but it will have to change

        # To create temporaries
        counter = generator()
        template = lambda: "r%d" % counter()

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = Toposort().process(clusters)
        clusters = fuse(clusters)

        # Flop reduction via the DSE
        clusters = rewrite(clusters, template, **kwargs)

        # Lifting
        clusters = Lift().process(clusters)

        # Lifting may create fusion opportunities, which in turn may enable
        # further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters, template)
        clusters = scalarize(clusters, template)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        try:
            # Create OPS kernels
            make_ops_kernels(graph)
        except OPSException:
            # No kernel can be off-loaded, use the default compiler
            OPSOperator._use_ops = False

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph

    @classmethod
    def _build(cls, expressions, **kwargs):

        op = super(OPSOperator, cls)._build(expressions, **kwargs)

        if OPSOperator._use_ops:
            op._compiler = ops_configuration['compiler'].copy()

        return op

    @property
    def hcode(self):
        return ''.join(str(kernel.root) for kernel in self._func_table.values())

    def _jit_compile(self):

        if OPSOperator._use_ops:
            self._includes.append('%s.h' % self._soname)
            if self._lib is None:
                self._compiler.jit_compile(self._soname, str(self.ccode), str(self.hcode))
        else:
            self._compiler.jit_compile(self._soname, str(self.ccode))


@iet_pass
def make_ops_kernels(iet):
    warning("The OPS backend is still work-in-progress")

    affine_trees = find_affine_trees(iet).items()

    # If there is no affine trees, then there is no loop to be optimized using OPS.
    if not affine_trees:
        raise OPSException("No off-loadable code found.")

    ops_init = Call(namespace['ops_init'], [0, 0, 2])
    ops_partition = Call(namespace['ops_partition'], Literal('""'))
    ops_exit = Call(namespace['ops_exit'])

    # Extract all symbols that need to be converted to ops_dat
    dims = []
    to_dat = set()
    for _, tree in affine_trees:
        dims.append(len(tree[0].dimensions))
        symbols = set(FindSymbols('symbolics').visit(tree[0].root))
        symbols -= set(FindSymbols('defines').visit(tree[0].root))
        to_dat |= symbols

    # Create the OPS block for this problem
    ops_block = OpsBlock('block')
    ops_block_init = Expression(ClusterizedEq(Eq(
        ops_block,
        namespace['ops_decl_block'](
            dims[0],
            Literal('"block"')
        )
    )))

    # To ensure deterministic code generation we order the datasets to
    # be generated (since a set is an unordered collection)
    to_dat = filter_sorted(to_dat)

    name_to_ops_dat = {}
    pre_time_loop = []
    for f in to_dat:
        if f.is_Constant:
            continue

        pre_time_loop.extend(list(create_ops_dat(f, name_to_ops_dat, ops_block)))

    # Generate ops kernels for each offloadable iteration tree
    mapper = {}
    ffuncs = []
    for n, (_, tree) in enumerate(affine_trees):
        pre_loop, ops_kernel, ops_par_loop_call, accessibles_info = opsit(
            tree, n, name_to_ops_dat, ops_block, dims[0]
        )

        pre_time_loop.extend(pre_loop)
        ffuncs.append(ops_kernel)

        # Memory set calls
        memory_set_calls = [create_ops_memory_set(
            f, name_to_ops_dat, accessibles_info
        ) for f in to_dat if not f.is_Constant]
        # Memory fetch calls
        memory_fetch_calls = [create_ops_memory_fetch(
            f, name_to_ops_dat, accessibles_info
        ) for f in to_dat if not f.is_Constant]

        mapper[tree[0].root] = List(
            body=(memory_set_calls, ops_par_loop_call, memory_fetch_calls))
        mapper.update({i.root: mapper.get(i.root) for i in tree})  # Drop trees

    iet = Transformer(mapper).visit(iet)

    assert (d == dims[0] for d in dims), \
        "The OPS backend currently assumes that all kernels \
        have the same number of dimensions"

    iet = iet._rebuild(body=flatten([ops_init, ops_block_init, pre_time_loop,
                                     ops_partition, iet.body,
                                     ops_exit]))

    return iet, {'includes': ['stdio.h', 'ops_seq.h'],
                 'ffuncs': ffuncs,
                 'headers': [namespace['ops_define_dimension'](dims[0])]}


class OPSException(Exception):
    pass
