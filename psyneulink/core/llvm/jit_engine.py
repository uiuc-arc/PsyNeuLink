# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

from llvmlite import binding
import os

__all__ = ['cpu_jit_engine']

# Compiler binding
__initialized = False
def _binding_initialize():
    global __initialized
    if not __initialized:
        binding.initialize()
        __initialized = True


def _cpu_jit_constructor():
    _binding_initialize()    

    # PassManagerBuilder can be shared
    __pass_manager_builder = binding.PassManagerBuilder()
    __pass_manager_builder.inlining_threshold = 99999  # Inline all function calls
    __pass_manager_builder.loop_vectorize = True
    __pass_manager_builder.slp_vectorize = True
    __pass_manager_builder.opt_level = 3  # Most aggressive optimizations

    # native == currently running CPU. ASM printer includes opcode emission
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    __cpu_features = binding.get_host_cpu_features().flatten()
    __cpu_name = binding.get_host_cpu_name()

    # Create compilation target, use default triple
    __cpu_target = binding.Target.from_default_triple()
    __cpu_target_machine = __cpu_target.create_target_machine(cpu=__cpu_name, features=__cpu_features, opt=3)

    __cpu_pass_manager = binding.ModulePassManager()
    __cpu_target_machine.add_analysis_passes(__cpu_pass_manager)
    __pass_manager_builder.populate(__cpu_pass_manager)


    # And an execution engine with an empty backing module
    # TODO: why is empty backing mod necessary?
    # TODO: It looks like backing_mod is just another compiled module.
    #       Can we use it to avoid recompiling builtins?
    #       Would cross module calls work? and for GPUs?
    __backing_mod = binding.parse_assembly("")

    __cpu_jit_engine = binding.create_mcjit_compiler(__backing_mod, __cpu_target_machine)
    return __cpu_jit_engine, __cpu_pass_manager, __cpu_target_machine

_dumpenv = os.environ.get("PNL_LLVM_DUMP")

class jit_engine:
    def __init__(self):
        self._jit_engine = None
        self._jit_pass_manager = None
        self._target_machine = None
        self.__mod = None

    def opt_and_add_bin_module(self, module):
        self._pass_manager.run(module)
        if _dumpenv is not None and _dumpenv.find("opt") != -1:
            print(module)

        # This prints generated x86 assembly
        if _dumpenv is not None and _dumpenv.find("isa") != -1:
            print("ISA assembly:")
            print(self._target_machine.emit_assembly(self.__mod))

        self._engine.add_module(module)
        self._engine.finalize_object()

    def _remove_bin_module(self, module):
        self._engine.remove_module(module)

    def opt_and_append_bin_module(self, module):
        if self.__mod is None:
            self.__mod = module
        else:
            self._remove_bin_module(self.__mod)
            self.__mod.link_in(module)

        self.opt_and_add_bin_module(self.__mod)

    @property
    def _engine(self):
        if self._jit_engine is None:
            self._init()

        return self._jit_engine

    @property
    def _pass_manager(self):
        if self._jit_pass_manager is None:
            self._init()

        return self._jit_pass_manager


class cpu_jit_engine(jit_engine):

    def __init__(self, object_cache = None):
        super().__init__()
        self._object_cache = object_cache

    def _init(self):
        assert self._jit_engine is None
        assert self._jit_pass_manager is None
        assert self._target_machine is None

        self._jit_engine, self._jit_pass_manager, self._target_machine = _cpu_jit_constructor()
        if self._object_cache is not None:
             self._jit_engine.set_object_cache(self._object_cache)
