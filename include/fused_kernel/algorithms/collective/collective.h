/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_COLLECTIVE
#define FK_COLLECTIVE

/* Cooperative (cross-thread) building blocks for tiled / compute-bound
 * algorithms: warp- and block-level reductions parameterised by an
 * associative combine functor, and statically-shaped Tile views with
 * composable layout/swizzle policies — all in the FKL composition style. */
#include <fused_kernel/algorithms/collective/reduce.h>
#include <fused_kernel/algorithms/collective/tile.h>
#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/algorithms/collective/copy.h>
#include <fused_kernel/algorithms/collective/mainloop.h>
#include <fused_kernel/algorithms/collective/epilogue.h>
#include <fused_kernel/algorithms/collective/register_tile.h>

#endif // FK_COLLECTIVE
