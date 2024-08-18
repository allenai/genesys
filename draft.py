graph(%self.1 : __torch__.builtins.GABCustom,
      %X.1 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=0, device=cpu)):
  %mlp : __torch__.builtins.SwiGluMLP = prim::GetAttr[name="mlp"](%self.1)
  %ffn_norm : __torch__.builtins.___torch_mangle_7.RMSNorm = prim::GetAttr[name="ffn_norm"](%self.1)
  %seq_modeling_block : __torch__.builtins.TTTLinear = prim::GetAttr[name="seq_modeling_block"](%self.1)
  %seq_norm : __torch__.builtins.___torch_mangle_6.RMSNorm = prim::GetAttr[name="seq_norm"](%self.1)
  %conv.3 : __torch__.builtins.Conv = prim::GetAttr[name="conv"](%self.1)
  %conv : __torch__.torch.nn.modules.conv.Conv1d = prim::GetAttr[name="conv"](%conv.3)
  %conv.1 : __torch__.builtins.Conv = prim::GetAttr[name="conv"](%self.1)
  %norm : __torch__.builtins.RMSNorm = prim::GetAttr[name="norm"](%conv.1)
  %101 : int = prim::Constant[value=1]() # <string>:471:0
  %102 : int = aten::size(%X.1, %101) # <string>:471:0
  %103 : Long(device=cpu) = prim::NumToTensor(%102)
  %107 : Scalar = aten::ScalarImplicit(%103)
  %108 : int = prim::Constant[value=0]() # <string>:471:0
  %109 : int = prim::Constant[value=4]() # <string>:471:0
  %110 : NoneType = prim::Constant()
  %111 : Device = prim::Constant[value="cpu"]() # <string>:471:0
  %112 : bool = prim::Constant[value=0]() # <string>:471:0
  %113 : Long(100, strides=[1], requires_grad=0, device=cpu) = aten::arange(%108, %107, %109, %110, %111, %112) # <string>:471:0
  %114 : int = prim::Constant[value=0]() # <string>:472:0
  %position_ids.1 : Long(1, 100, strides=[100, 1], requires_grad=0, device=cpu) = aten::unsqueeze(%113, %114) # <string>:472:0
  %119 : int = prim::Constant[value=1]() # <string>:144:0
  %120 : int = aten::size(%X.1, %119) # <string>:144:0
  %seq_len.1 : Long(device=cpu) = prim::NumToTensor(%120)
  %166 : int = aten::Int(%seq_len.1)
  %2614 : (Tensor, Tensor) = prim::CallMethod[name="forward"](%norm, %X.1)
  %2600 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu), %2601 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=0, device=cpu) = prim::TupleUnpack(%2614)
  %148 : int = prim::Constant[value=1]() # <string>:146:0
  %149 : int = prim::Constant[value=2]() # <string>:146:0
  %input.1 : Float(2, 128, 100, strides=[12800, 1, 128], requires_grad=1, device=cpu) = aten::transpose(%2600, %148, %149) # <string>:146:0
  %2615 : Tensor = prim::CallMethod[name="forward"](%conv, %input.1)
  %167 : int = prim::Constant[value=2]() # <string>:148:0
  %168 : int = prim::Constant[value=0]() # <string>:148:0
  %169 : int = prim::Constant[value=1]() # <string>:148:0
  %hidden_states.9 : Float(2, 128, 100, strides=[13184, 103, 1], requires_grad=1, device=cpu) = aten::slice(%2615, %167, %168, %166, %169) # <string>:148:0
  %171 : int = prim::Constant[value=1]() # <string>:154:0
  %172 : int = prim::Constant[value=2]() # <string>:154:0
  %hidden_states.11 : Float(2, 100, 128, strides=[13184, 1, 103], requires_grad=1, device=cpu) = aten::transpose(%hidden_states.9, %171, %172) # <string>:154:0
  %174 : int = prim::Constant[value=1]() # <string>:475:0
  %hidden_states.13 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu) = aten::add(%2601, %hidden_states.11, %174) # <string>:475:0
  %2616 : (Tensor, Tensor) = prim::CallMethod[name="forward"](%seq_norm, %hidden_states.13)
  %2604 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu), %2605 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu) = prim::TupleUnpack(%2616)
  %2617 : Tensor = prim::CallMethod[name="forward"](%seq_modeling_block, %2604, %position_ids.1)
  %2463 : int = prim::Constant[value=1]() # <string>:479:0
  %hidden_states.25 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu) = aten::add(%2605, %2617, %2463) # <string>:479:0
  %2618 : (Tensor, Tensor) = prim::CallMethod[name="forward"](%ffn_norm, %hidden_states.25)
  %2612 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu), %2613 : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu) = prim::TupleUnpack(%2618)
  %2619 : Tensor = prim::CallMethod[name="forward"](%mlp, %2612)
  %2496 : int = prim::Constant[value=1]() # <string>:483:0
  %Y : Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu) = aten::add(%2613, %2619, %2496) # <string>:483:0
  %2519 : () = prim::TupleConstruct()
  %2520 : (Float(2, 100, 128, strides=[12800, 128, 1], requires_grad=1, device=cpu), ()) = prim::TupleConstruct(%Y, %2519)
  return (%2520)