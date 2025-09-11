/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_PARENT_OPERATIONS_BUILDERS_H
#define FK_PARENT_OPERATIONS_BUILDERS_H

#include <fused_kernel/core/execution_model/operation_model/parent_operations_exec.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>

namespace fk {
    // PARENT OPERATIONS
    // PARENT COMPUTE OPERATIONS

#define UNARY_BUILD(NAME, InputT, OutputT, IS_FOP)                      \
  UNARY_ALIAS(InputT, OutputT, IS_FOP)                                  \
  using InstantiableType = Unary<NAME>;                                 \
  FK_HOST_FUSE InstantiableType build() { return InstantiableType{}; }

#define TEMPLATE_UNARY_BUILD(NAME, TEMPLATE_PARAMS, InputT, OutputT, IS_FOP)  \
    UNARY_ALIAS(InputT, OutputT, IS_FOP)                                      \
    using InstantiableType = Unary<NAME<DEPAREN(TEMPLATE_PARAMS)>>;           \
    FK_HOST_FUSE InstantiableType build() { return InstantiableType{}; }

#define DECLARE_ODATA_BUILD \
FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return InstantiableType{opData}; }
#define DECLARE_PARAMS_BUILD DECLARE_ODATA_BUILD \
FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return InstantiableType{{params}}; }
#define DECLARE_PARAMS_BACKIOP_BUILD DECLARE_ODATA_BUILD \
FK_HOST_FUSE InstantiableType build(const ParamsType &params, const BackIOp &backIOp) { \
  return InstantiableType{{params, backIOp}}; \
}

#define BINARY_BUILD(NAME, InputT, ParamsT, OutputT, IS_FOP)                \
  BINARY_OPERATION(NAME, InputT, ParamsT, OutputT, IS_FOP)                  \
  using InstantiableType = Binary<NAME>;                                    \
  DECLARE_PARAMS_BUILD

#define TEMPLATE_BINARY_BUILD(NAME, TEMPLATE_PARAMS, InputT, ParamsT, OutputT, IS_FOP) \
  TEMPLATE_BINARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, OutputT, IS_FOP)   \
  using InstantiableType = Binary<NAME<DEPAREN(TEMPLATE_PARAMS)>>;                     \
  DECLARE_PARAMS_BUILD

#define TERNARY_BUILD(NAME, InputT, ParamsT, BackIOpT, OutputT)  \
  TERNARY_ALIAS(InputT, ParamsT, BackIOpT, OutputT)              \
  using InstantiableType = Ternary<NAME>;                        \
  DECLARE_PARAMS_BACKIOP_BUILD

#define TEMPLATE_TERNARY_BUILD(NAME, TEMPLATE_PARAMS, InputT, ParamsT, BackIOpT, OutputT) \
    TERNARY_ALIAS(InputT, ParamsT, BackIOpT, OutputT)                                     \
    using InstantiableType = Ternary<NAME<DEPAREN(TEMPLATE_PARAMS)>>;                     \
    DECLARE_PARAMS_BACKIOP_BUILD

#define READ_BUILD(NAME, ReadDataT, ParamsT, OutputT, IS_FOP, TF_) \
  READ_ALIAS(ReadDataT, ParamsT, OutputT, IS_FOP, TF_)             \
  using InstantiableType = Read<NAME>;                             \
  DECLARE_PARAMS_BUILD

#define TEMPLATE_READ_BUILD(NAME, TEMPLATE_PARAMS, ReadDataT, ParamsT, OutputT, IS_FOP, TF_)  \
  READ_ALIAS(ReadDataT, ParamsT, OutputT, IS_FOP, TF_)                                        \
  using InstantiableType = Read<NAME<DEPAREN(TEMPLATE_PARAMS)>>;                              \
  DECLARE_PARAMS_BUILD

#define READBACK_BUILD(NAME, ReadDataT, ParamsT, BackIOpT, OutputT, IS_FOP, TF_) \
  READBACK_ALIAS(ReadDataT, ParamsT, BackIOpT, OutputT, IS_FOP, TF_)             \
  using InstantiableType = ReadBack<NAME>;                                       \
  DECLARE_PARAMS_BACKIOP_BUILD

#define TEMPLATE_READBACK_BUILD(NAME, TEMPLATE_PARAMS, ReadDataT, ParamsT, BackIOpT, OutputT, IS_FOP, TF_) \
  READBACK_ALIAS(ReadDataT, ParamsT, BackIOpT, OutputT, IS_FOP, TF_)                                       \
  using InstantiableType = ReadBack<NAME<DEPAREN(TEMPLATE_PARAMS)>>;                                       \
  DECLARE_PARAMS_BACKIOP_BUILD

} // namespace fk

#endif // FK_PARENT_OPERATIONS_BUILDERS_H