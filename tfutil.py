import os
import sys
import inspect
import importlib
import imp
import numpy as np
from collections import OrderedDict
import tensorflow as tf

#----------------------------------------------------------------------------

def run(*args, **kwargs):
    return tf.get_default_session().run(*args, **kwargs)

def is_tf_expression(x):
    return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)

def shape_to_list(shape):
    return [dim.value for dim in shape]

def flatten(x):
    with tf.name_scope('Flatten'):
        return tf.reshape(x, [-1])

def log2(x):
    with tf.name_scope('Log2'):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))

def exp2(x):
    with tf.name_scope('Exp2'):
        return tf.exp(x * np.float32(np.log(2.0)))

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

def absolute_name_scope(scope):
    return tf.name_scope(scope + '/')


def set_vars(var_to_value_dict):
    ops = []
    feed_dict = {}
    for var, value in var_to_value_dict.items():
        assert is_tf_expression(var)
        try:
            setter = tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/setter:0'))
        except KeyError:
            with absolute_name_scope(var.name.split(':')[0]):
                with tf.control_dependencies(None):
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, 'new_value'), name='setter') # create new setter
        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value
    run(ops, feed_dict)

#----------------------------------------------------------------------------

def import_module(module_or_obj_name):
    parts = module_or_obj_name.split('.')
    parts[0] = {'np': 'numpy', 'tf': 'tensorflow'}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:i]))
            relative_obj_name = '.'.join(parts[i:])
            return module, relative_obj_name
        except ImportError:
            pass
    raise ImportError(module_or_obj_name)

def find_obj_in_module(module, relative_obj_name):
    obj = module
    for part in relative_obj_name.split('.'):
        obj = getattr(obj, part)
    return obj

def import_obj(obj_name):
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)

def call_func_by_name(*args, func=None, **kwargs):
    assert func is not None
    return import_obj(func)(*args, **kwargs)


#----------------------------------------------------------------------------
# Network abstraction.

network_import_handlers = []
_network_import_modules = []

class Network:
    def __init__(self,
        name=None,
        func=None,
        **static_kwargs):

        self._init_fields()
        self.name = name
        self.static_kwargs = dict(static_kwargs)

        # Init build func.
        module, self._build_func_name = import_module(func)
        self._build_module_src = inspect.getsource(module)
        self._build_func = find_obj_in_module(module, self._build_func_name)

        # Init graph.
        self._init_graph()
        self.reset_vars()

    def _init_fields(self):
        self.name               = None
        self.scope              = None
        self.static_kwargs      = dict()
        self.num_inputs         = 0
        self.num_outputs        = 0
        self.input_shapes       = [[]]
        self.output_shapes      = [[]]
        self.input_shape        = []
        self.output_shape       = []
        self.input_templates    = []
        self.output_templates   = []
        self.input_names        = []
        self.output_names       = []
        self.vars               = OrderedDict()
        self.trainables         = OrderedDict()
        self._build_func        = None
        self._build_func_name   = None
        self._build_module_src  = None
        self._run_cache         = dict()
        
    def _init_graph(self):
        self.input_names = []
        for param in inspect.signature(self._build_func).parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                self.input_names.append(param.name)
        self.num_inputs = len(self.input_names)
        assert self.num_inputs >= 1

        if self.name is None:
            self.name = self._build_func_name
        self.scope = tf.get_default_graph().unique_name(self.name.replace('/', '_'), mark_as_used=False)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            assert tf.get_variable_scope().name == self.scope
            with absolute_name_scope(self.scope):
                with tf.control_dependencies(None):
                    self.input_templates = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    out_expr = self._build_func(*self.input_templates, is_template_graph=True, **self.static_kwargs)
            
        assert is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        self.output_templates = [out_expr] if is_tf_expression(out_expr) else list(out_expr)
        self.output_names = [t.name.split('/')[-1].split(':')[0] for t in self.output_templates]
        self.num_outputs = len(self.output_templates)
        assert self.num_outputs >= 1
        
        self.input_shapes   = [shape_to_list(t.shape) for t in self.input_templates]
        self.output_shapes  = [shape_to_list(t.shape) for t in self.output_templates]
        self.input_shape    = self.input_shapes[0]
        self.output_shape   = self.output_shapes[0]
        self.vars           = OrderedDict([(self.get_var_localname(var), var) for var in tf.global_variables(self.scope + '/')])
        self.trainables     = OrderedDict([(self.get_var_localname(var), var) for var in tf.trainable_variables(self.scope + '/')])

    def reset_vars(self):
        run([var.initializer for var in self.vars.values()])

    def reset_trainables(self):
        run([var.initializer for var in self.trainables.values()])

    def get_output_for(self, *in_expr, return_as_list=False, **dynamic_kwargs):
        assert len(in_expr) == self.num_inputs
        all_kwargs = dict(self.static_kwargs)
        all_kwargs.update(dynamic_kwargs)
        with tf.variable_scope(self.scope, reuse=True):
            assert tf.get_variable_scope().name == self.scope
            named_inputs = [tf.identity(expr, name=name) for expr, name in zip(in_expr, self.input_names)]
            out_expr = self._build_func(*named_inputs, **all_kwargs)
        assert is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        if return_as_list:
            out_expr = [out_expr] if is_tf_expression(out_expr) else list(out_expr)
        return out_expr

    def get_var_localname(self, var_or_globalname):
        assert is_tf_expression(var_or_globalname) or isinstance(var_or_globalname, str)
        globalname = var_or_globalname if isinstance(var_or_globalname, str) else var_or_globalname.name
        assert globalname.startswith(self.scope + '/')
        localname = globalname[len(self.scope) + 1:]
        localname = localname.split(':')[0]
        return localname

    def find_var(self, var_or_localname):
        assert is_tf_expression(var_or_localname) or isinstance(var_or_localname, str)
        return self.vars[var_or_localname] if isinstance(var_or_localname, str) else var_or_localname

    def get_var(self, var_or_localname):
        return self.find_var(var_or_localname).eval()
        
    def set_var(self, var_or_localname, new_value):
        return set_vars({self.find_var(var_or_localname): new_value})

    # Pickle export.
    def __getstate__(self):
        return {
            'version':          2,
            'name':             self.name,
            'static_kwargs':    self.static_kwargs,
            'build_module_src': self._build_module_src,
            'build_func_name':  self._build_func_name,
            'variables':        list(zip(self.vars.keys(), run(list(self.vars.values()))))}

    # Pickle import.
    def __setstate__(self, state):
        self._init_fields()

        for handler in network_import_handlers:
            state = handler(state)

        assert state['version'] == 2
        self.name = state['name']
        self.static_kwargs = state['static_kwargs']
        self._build_module_src = state['build_module_src']
        self._build_func_name = state['build_func_name']
        
        module = imp.new_module('_tfutil_network_import_module_%d' % len(_network_import_modules))
        exec(self._build_module_src, module.__dict__)
        self._build_func = find_obj_in_module(module, self._build_func_name)
        _network_import_modules.append(module) # avoid gc

        self._init_graph()
        self.reset_vars()
        set_vars({self.find_var(name): value for name, value in state['variables']})

    def clone(self, name=None):
        net = object.__new__(Network)
        net._init_fields()
        net.name = name if name is not None else self.name
        net.static_kwargs = dict(self.static_kwargs)
        net._build_module_src = self._build_module_src
        net._build_func_name = self._build_func_name
        net._build_func = self._build_func
        net._init_graph()
        net.copy_vars_from(self)
        return net

    def copy_vars_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.vars.keys()})
        set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    def copy_trainables_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.trainables.keys()})
        set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    def convert(self, name=None, func=None, **static_kwargs):
        net = Network(name, func, **static_kwargs)
        net.copy_vars_from(self)
        return net

    def setup_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        assert isinstance(src_net, Network)
        with absolute_name_scope(self.scope):
            with tf.name_scope('MovingAvg'):
                ops = []
                for name, var in self.vars.items():
                    if name in src_net.vars:
                        cur_beta = beta if name in self.trainables else beta_nontrainable
                        new_value = lerp(src_net.vars[name], var, cur_beta)
                        ops.append(var.assign(new_value))
                return tf.group(*ops)

    def run(self, *in_arrays,
        return_as_list  = False,
        print_progress  = False,
        minibatch_size  = None,
        num_gpus        = 1,
        out_mul         = 1.0,
        out_add         = 0.0,
        out_shrink      = 1,
        out_dtype       = None,
        **dynamic_kwargs):

        assert len(in_arrays) == self.num_inputs
        num_items = in_arrays[0].shape[0]
        if minibatch_size is None:
            minibatch_size = num_items
        key = str([list(sorted(dynamic_kwargs.items())), num_gpus, out_mul, out_add, out_shrink, out_dtype])

        # Build graph.
        if key not in self._run_cache:
            with absolute_name_scope(self.scope + '/Run'), tf.control_dependencies(None):
                in_split = list(zip(*[tf.split(x, num_gpus) for x in self.input_templates]))
                out_split = []
                for gpu in range(num_gpus):
                    with tf.device('/gpu:%d' % gpu):
                        out_expr = self.get_output_for(*in_split[gpu], return_as_list=True, **dynamic_kwargs)
                        if out_mul != 1.0:
                            out_expr = [x * out_mul for x in out_expr]
                        if out_add != 0.0:
                            out_expr = [x + out_add for x in out_expr]
                        if out_shrink > 1:
                            ksize = [1, 1, out_shrink, out_shrink]
                            out_expr = [tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') for x in out_expr]
                        if out_dtype is not None:
                            if tf.as_dtype(out_dtype).is_integer:
                                out_expr = [tf.round(x) for x in out_expr]
                            out_expr = [tf.saturate_cast(x, out_dtype) for x in out_expr]
                        out_split.append(out_expr)
                self._run_cache[key] = [tf.concat(outputs, axis=0) for outputs in zip(*out_split)]

        # Run minibatches.
        out_expr = self._run_cache[key]
        out_arrays = [np.empty([num_items] + shape_to_list(expr.shape)[1:], expr.dtype.name) for expr in out_expr]
        for mb_begin in range(0, num_items, minibatch_size):
            if print_progress:
                print('\r%d / %d' % (mb_begin, num_items), end='')
            mb_end = min(mb_begin + minibatch_size, num_items)
            mb_in = [src[mb_begin : mb_end] for src in in_arrays]
            mb_out = tf.get_default_session().run(out_expr, dict(zip(self.input_templates, mb_in)))
            for dst, src in zip(out_arrays, mb_out):
                dst[mb_begin : mb_end] = src

        # Done.
        if print_progress:
            print('\r%d / %d' % (num_items, num_items))
        if not return_as_list:
            out_arrays = out_arrays[0] if len(out_arrays) == 1 else tuple(out_arrays)
        return out_arrays

    def list_layers(self):
        patterns_to_ignore = ['/Setter', '/new_value', '/Shape', '/strided_slice', '/Cast', '/concat']
        all_ops = tf.get_default_graph().get_operations()
        all_ops = [op for op in all_ops if not any(p in op.name for p in patterns_to_ignore)]
        layers = []

        def recurse(scope, parent_ops, level):
            prefix = scope + '/'
            ops = [op for op in parent_ops if op.name == scope or op.name.startswith(prefix)]

            # Does not contain leaf nodes => expand immediate children.
            if level == 0 or all('/' in op.name[len(prefix):] for op in ops):
                visited = set()
                for op in ops:
                    suffix = op.name[len(prefix):]
                    if '/' in suffix:
                        suffix = suffix[:suffix.index('/')]
                    if suffix not in visited:
                        recurse(prefix + suffix, ops, level + 1)
                        visited.add(suffix)

            else:
                layer_name = scope[len(self.scope)+1:]
                layer_output = ops[-1].outputs[0]
                layer_trainables = [op.outputs[0] for op in ops if op.type.startswith('Variable') and self.get_var_localname(op.name) in self.trainables]
                layers.append((layer_name, layer_output, layer_trainables))

        recurse(self.scope, all_ops, 0)
        return layers

    def print_layers(self, title=None, hide_layers_with_no_params=False):
        if title is None: title = self.name
        print()
        print('%-28s%-12s%-24s%-24s' % (title, 'Params', 'OutputShape', 'WeightShape'))
        print('%-28s%-12s%-24s%-24s' % (('---',) * 4))

        total_params = 0
        for layer_name, layer_output, layer_trainables in self.list_layers():
            weights = [var for var in layer_trainables if var.name.endswith('/weight:0')]
            num_params = sum(np.prod(shape_to_list(var.shape)) for var in layer_trainables)
            total_params += num_params
            if hide_layers_with_no_params and num_params == 0:
                continue

            print('%-28s%-12s%-24s%-24s' % (
                layer_name,
                num_params if num_params else '-',
                layer_output.shape,
                weights[0].shape if len(weights) == 1 else '-'))

        print('%-28s%-12s%-24s%-24s' % (('---',) * 4))
        print('%-28s%-12s%-24s%-24s' % ('Total', total_params, '', ''))
        print()

    def setup_weight_histograms(self, title=None):
        if title is None: title = self.name
        with tf.name_scope(None), tf.device(None), tf.control_dependencies(None):
            for localname, var in self.trainables.items():
                if '/' in localname:
                    p = localname.split('/')
                    name = title + '_' + p[-1] + '/' + '_'.join(p[:-1])
                else:
                    name = title + '_toplevel/' + localname
                tf.summary.histogram(name, var)

#----------------------------------------------------------------------------
