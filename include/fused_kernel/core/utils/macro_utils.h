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

#ifndef FK_MACRO_UTILS_H
#define FK_MACRO_UTILS_H

// Standard concatenation and stringification
#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define STRINGIFY_INNER(x) #x
#define STRINGIFY(x) STRINGIFY_INNER(x)

// Removes a single set of parentheses: DEPAREN((a,b)) -> a,b
#define DEPAREN_IMPL(...) __VA_ARGS__
#define DEPAREN(x) DEPAREN_IMPL x

// =========================================================================
// 2. Variadic Argument Concatenation (the magic part)
//
// This takes multiple arguments (e.g., a, b, c) and creates a single
// token with underscores (a_b_c). It supports 1 to 10 identifiers.
// =========================================================================

// Macros to concatenate a specific number of arguments with underscores
#define CAT_WITH_UNDERSCORE_1(a) a
#define CAT_WITH_UNDERSCORE_2(a, b) a##_##b
#define CAT_WITH_UNDERSCORE_3(a, b, c) a##_##b##_##c
#define CAT_WITH_UNDERSCORE_4(a, b, c, d) a##_##b##_##c##_##d
#define CAT_WITH_UNDERSCORE_5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define CAT_WITH_UNDERSCORE_6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f
#define CAT_WITH_UNDERSCORE_7(a, b, c, d, e, f, g) a##_##b##_##c##_##d##_##e##_##f##_##g
#define CAT_WITH_UNDERSCORE_8(a, b, c, d, e, f, g, h) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h
#define CAT_WITH_UNDERSCORE_9(a, b, c, d, e, f, g, h, i) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h##_##i
#define CAT_WITH_UNDERSCORE_10(a, b, c, d, e, f, g, h, i, j) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h##_##i##_##j

// Macros to expand template arguments by removing parentheses from each argument
#define TEMPLATE_ARGS_1(a) DEPAREN(a)
#define TEMPLATE_ARGS_2(a, b) DEPAREN(a), DEPAREN(b)
#define TEMPLATE_ARGS_3(a, b, c) DEPAREN(a), DEPAREN(b), DEPAREN(c)
#define TEMPLATE_ARGS_4(a, b, c, d) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d)
#define TEMPLATE_ARGS_5(a, b, c, d, e) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d), DEPAREN(e)
#define TEMPLATE_ARGS_6(a, b, c, d, e, f) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d), DEPAREN(e), DEPAREN(f)
#define TEMPLATE_ARGS_7(a, b, c, d, e, f, g) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d), DEPAREN(e), DEPAREN(f), DEPAREN(g)
#define TEMPLATE_ARGS_8(a, b, c, d, e, f, g, h) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d), DEPAREN(e), DEPAREN(f), DEPAREN(g), DEPAREN(h)
#define TEMPLATE_ARGS_9(a, b, c, d, e, f, g, h, i) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d), DEPAREN(e), DEPAREN(f), DEPAREN(g), DEPAREN(h), DEPAREN(i)
#define TEMPLATE_ARGS_10(a, b, c, d, e, f, g, h, i, j) DEPAREN(a), DEPAREN(b), DEPAREN(c), DEPAREN(d), DEPAREN(e), DEPAREN(f), DEPAREN(g), DEPAREN(h), DEPAREN(i), DEPAREN(j)

// This helper simply forces one more round of macro expansion.
#define EXPAND(x) x

// The helper that selects the 11th argument from a list.
#define GET_11TH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

// The reverse sequence of numbers, now including 0 for the zero-argument case.
#define REVERSE_SEQ_10() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

// The robust implementation of COUNT_VARARGS
// The extra _IMPL layer ensures __VA_ARGS__ is expanded correctly before counting.
#define COUNT_VARARGS_IMPL(...) EXPAND(GET_11TH_ARG(__VA_ARGS__))
#define COUNT_VARARGS(...) EXPAND(COUNT_VARARGS_IMPL(__VA_ARGS__, REVERSE_SEQ_10()))

// Dispatches to the correct CAT_WITH_UNDERSCORE_N macro based on the arg count
#define VA_CONCAT_DISPATCHER(count, ...) EXPAND(CONCAT(CAT_WITH_UNDERSCORE_, count)(__VA_ARGS__))
#define VA_CONCAT(...) VA_CONCAT_DISPATCHER(COUNT_VARARGS(__VA_ARGS__), __VA_ARGS__)

// Dispatches to the correct TEMPLATE_ARGS_N macro based on the arg count
#define VA_TEMPLATE_ARGS_DISPATCHER(count, ...) EXPAND(CONCAT(TEMPLATE_ARGS_, count)(__VA_ARGS__))
#define VA_TEMPLATE_ARGS(...) VA_TEMPLATE_ARGS_DISPATCHER(COUNT_VARARGS(__VA_ARGS__), __VA_ARGS__)

#endif // FK_MACRO_UTILS_H