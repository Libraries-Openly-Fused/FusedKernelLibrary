/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_CONVOLUTION2D_H
#define FK_CONVOLUTION2D_H

#include <fused_kernel/core/execution_model/operation_model/batch_operations.h>

namespace fk {

    struct MultiplyAddType;
    struct DilateType;
    struct ErodeType;

    // Convolution2D Kernels
    template <uint H_, uint W_, typename T>
    struct MultiplyAdd {
        using Type = MultiplyAddType;
        template <typename T>
        static constexpr bool is = std::is_same_v<T, Type>;
        static constexpr uint H{H_};
        static constexpr uint W{W_};
        using KType = T;
        T k[H][W];
    };

    template <uint H_, uint W_>
    struct Dilate{
        using Type = DilateType;
        template <typename T>
        static constexpr bool is = std::is_same_v<T, Type>;
        static constexpr uint H{H_};
        static constexpr uint W{W_};
    };

    template <uint H_, uint W_>
    struct Erode {
        using Type = ErodeType;
        template <typename T>
        static constexpr bool is = std::is_same_v<T, Type>;
        static constexpr uint H{H_};
        static constexpr uint W{W_};
    };


    template <typename P, typename BackIOp_>
    struct NaiveConvolution2DComplete {
        static_assert((P::W >= 3) && (P::H >= 3) &&
                      (!cxp::is_even::f(P::W) && !cxp::is_even::f(P::H)),
                      "Mask width and height minimum value supported is 3 and only odd values.");
        private:
            template <typename ParamsType, typename Enabler = void>
            struct ComputeOutputType;

            template <typename ParamsType>
            struct ComputeOutputType<ParamsType, std::enable_if_t<ParamsType::template is<MultiplyAddType>>> {
                using type = decltype(std::declval<typename BackIOp_::Operation::OutputType>()
                                        * std::declval<typename ParamsType::KType>());
            };

            template <typename ParamsType>
            struct ComputeOutputType<ParamsType, std::enable_if_t<!ParamsType::template is<MultiplyAddType>>> {
                using type = typename BackIOp_::Operation::OutputType;
            };

            using ComputedOutputType = typename ComputeOutputType<P>::type;
            using SelfType = NaiveConvolution2DComplete<P, BackIOp_>;
            using Parent =
                ReadBackOperation<typename BackIOp_::Operation::ReadDataType, P, BackIOp_,
                                  ComputedOutputType, SelfType>;
        public:
            FK_STATIC_STRUCT(NaiveConvolution2DComplete, SelfType)
            DECLARE_READBACK_PARENT

            FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
                return BackIOp::Operation::num_elems_x(thread, opData.backIOp);
            }
            FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
                return BackIOp::Operation::num_elems_y(thread, opData.backIOp);
            }
            FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
                return BackIOp::Operation::num_elems_z(thread, opData.backIOp);
            }

        private:
            FK_HOST_DEVICE_FUSE OutputType accumFirstVal() {
                if constexpr (P::template is<MultiplyAddType>) {
                    return 0;
                } else if constexpr (P::template is<DilateType>) {
                    return cxp::minValue<OutputType>;
                } else {
                    return cxp::maxValue<OutputType>;
                }
            }
        
        public:
            FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
                constexpr int2 radius = { (ParamsType::W - 1) >> 1, (ParamsType::H - 1) >> 1 };
                OutputType accum = accumFirstVal();
                for (int y=0; y < P::H; ++y) {
                    for (int x=0; x < P::W; ++x) {
                        const int2 offset = {x - radius.x, y - radius.y};
                        const Point newThreadIdx(cxp::max::f(thread.x + offset.x, 0), cxp::max::f(thread.y + offset.y, 0), thread.z);
                        if constexpr (P::template is<MultiplyAddType>) {
                            accum += BackIOp::Operation::exec(newThreadIdx, backIOp) * params.k[y][x];
                        } else if constexpr (P::template is<DilateType>) {
                            accum = cxp::max::f(BackIOp::Operation::exec(newThreadIdx, backIOp), accum);
                        } else {
                            static_assert(P::template is<ErodeType>,
                                "Wrong parameters passed to NaiveConvolution2DComplete");
                            accum = cxp::min::f(BackIOp::Operation::exec(newThreadIdx, backIOp), accum);
                        }
                    }
                }
                return accum;
            }
    };

    template <typename P = NullType>
    struct Conv2D {
        static_assert((P::W >= 3) && (P::H >= 3) &&
                      (!cxp::is_even::f(P::W) && !cxp::is_even::f(P::H)),
                      "Mask width and height minimum value supported is 3 and only odd values.");
        private:
            using SelfType = Conv2D<P>;
            using Parent =
                IncompleteReadBackOperation<NullType, P, NullType,
                                            NullType, SelfType>;
        public:
            FK_STATIC_STRUCT(Conv2D, SelfType)
            DECLARE_INCOMPLETEREADBACK_PARENT

            template <typename NewBackIOp>
            FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const InstantiableType& selfIOp) {
                return NaiveConvolution2DComplete<P, NewBackIOp>::build(selfIOp.params, backIOp);
            }
    };
    
    template <>
    struct Conv2D<NullType> {
        template <typename ParamsType>
        FK_HOST_FUSE auto build(const ParamsType& params) {
            return Conv2D<ParamsType>::build(params, NullType{});
        }
        template <typename ParamsType, typename BackIOp>
        FK_HOST_FUSE auto build(const ParamsType& params, const BackIOp& backIOp) {
            return NaiveConvolution2DComplete<ParamsType, BackIOp>::build(params, backIOp);
        }
    };

}

#endif // FK_CONVOLUTION2D_H