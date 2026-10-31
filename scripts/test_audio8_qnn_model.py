"""Static/protobuf tests only. Deliberately never import an inference runtime."""
import ast
from pathlib import Path
import sys
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n, TensorProto as T
from audio8_qnn_model import convert


def fixture(kind='fast'):
    layers,capacity=(4,10) if kind=='fast' else (24,2048)
    sequence=1 if kind=='fast' else 'sequence'
    inputs=[h.make_tensor_value_info('input_pos',T.INT64,[sequence]),
            h.make_tensor_value_info('delta',T.FLOAT16,['batch',2,sequence,2]),
            h.make_tensor_value_info('indices',T.INT64,['batch']),
            h.make_tensor_value_info('x',T.FLOAT16,[1,1,32])]
    nodes=[h.make_node('Gather',['input_pos','zero'],['position'],axis=0),
           h.make_node('Add',['position','one'],['end']),
           h.make_node('Range',['zero','end','one'],['positions'],name='/Range'),
           h.make_node('Range',['zero','position','one'],['prefix']),
           h.make_node('GatherBlockQuantized',['qw','indices','scale','zp'],['lookup'],
               domain='com.microsoft',block_size=32,gather_axis=0,quantize_axis=1),
           h.make_node('MatMulNBits',['x','nbw','nbs','nbz'],['projection'],
               domain='com.microsoft',K=32,N=2,bits=4,block_size=128,accuracy_level=4)]
    outputs=[h.make_tensor_value_info('positions',T.INT64,['valid']),
             h.make_tensor_value_info('lookup',T.FLOAT16,['batch',32]),
             h.make_tensor_value_info('projection',T.FLOAT16,[1,1,2])]
    for layer in range(layers):
        for kv in ('key','value'):
            name=f'cache_{kv}_{layer}'
            inputs.append(h.make_tensor_value_info(name,T.FLOAT16,['batch',2,capacity,2]))
            nodes.extend([h.make_node('Gather',[name,'prefix'],[name+'_prefix'],axis=2),
                          h.make_node('Concat',[name+'_prefix','delta'],[name+'_used'],axis=2)])
            outputs.append(h.make_tensor_value_info(name+'_used',T.FLOAT16,['batch',2,'valid',2]))
    tensors=[n.from_array(np.array(0,dtype=np.int64),'zero'),n.from_array(np.array(1,dtype=np.int64),'one'),
             h.make_tensor('qw',T.UINT4,[2,32],bytes([0x98])*32,raw=True),
             n.from_array(np.array([[.25],[.5]],dtype=np.float16),'scale'),
             h.make_tensor('zp',T.UINT4,[2,1],bytes([0x88]),raw=True),
             n.from_array(np.full((2,1,64),0x98,dtype=np.uint8),'nbw'),
             n.from_array(np.ones((2,1),dtype=np.float16),'nbs'),
             n.from_array(np.full((2,1),0x88,dtype=np.uint8),'nbz')]
    return h.make_model(h.make_graph(nodes,'fixture',inputs,outputs,tensors),
                        opset_imports=[h.make_opsetid('',17),h.make_opsetid('com.microsoft',1)],ir_version=10)


class ConversionTests(unittest.TestCase):
    def test_no_inference_runtime_dependency(self):
        source=Path(__file__).with_name('audio8_qnn_model.py').read_text()
        for node in ast.walk(ast.parse(source)):
            names=[]
            if isinstance(node,ast.Import): names=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom): names=[node.module or '']
            self.assertFalse(any(name.startswith('onnxruntime') for name in names))
        self.assertFalse(any(name.startswith('onnxruntime') for name in sys.modules))

    def test_static_contracts_and_int4_weights(self):
        for kind,layers in [('fast',4),('slow',24)]:
            original=fixture(kind)
            weights=next(t for t in original.graph.initializer if t.name=='nbw').SerializeToString()
            model=convert(original,kind)
            self.assertEqual(sum(n.op_type=='Where' for n in model.graph.node),layers*2)
            self.assertEqual(sum(n.op_type=='Range' for n in model.graph.node),0)
            self.assertEqual(sum(n.op_type=='GatherBlockQuantized' for n in model.graph.node),0)
            self.assertEqual(sum(n.op_type=='MatMulNBits' for n in model.graph.node),1)
            self.assertEqual(next(t for t in model.graph.initializer if t.name=='nbw').SerializeToString(),weights)
            self.assertTrue(all(d.HasField('dim_value') for t in model.graph.input for d in t.type.tensor_type.shape.dim))
            table=n.to_array(next(t for t in model.graph.initializer if t.name=='qw_dequant_fp16'))
            np.testing.assert_array_equal(table[0],np.tile(np.array([0,.25],dtype=np.float16),16))
            np.testing.assert_array_equal(table[1],np.tile(np.array([0,.5],dtype=np.float16),16))

    def test_invalid_contract_is_rejected(self):
        with self.assertRaises(ValueError): convert(fixture(),'unknown')
        model=fixture()
        node=next(n for n in model.graph.node if n.op_type=='GatherBlockQuantized')
        for attr in node.attribute:
            if attr.name=='block_size': attr.i=64
        with self.assertRaises(ValueError): convert(model,'fast')

    def test_fixed_attention_mask_math(self):
        rng=np.random.default_rng(42)
        keys=rng.normal(size=(10,4)); values=rng.normal(size=(10,4))
        q=rng.normal(size=(4,)); new_key=rng.normal(size=(4,)); new_value=rng.normal(size=(4,))
        for position in (0,1,3,9):
            prefix_k=np.concatenate([keys[:position],new_key[None]])
            prefix_v=np.concatenate([values[:position],new_value[None]])
            logits=prefix_k@q; probs=np.exp(logits-logits.max()); expected=probs@prefix_v/probs.sum()
            fixed_k=keys.copy();fixed_v=values.copy()
            fixed_k[position]=new_key;fixed_v[position]=new_value
            logits=fixed_k@q; logits[np.arange(10)>position]=-np.inf
            probs=np.exp(logits-logits.max()); actual=probs@fixed_v/probs.sum()
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)

if __name__=='__main__': unittest.main(verbosity=2)
