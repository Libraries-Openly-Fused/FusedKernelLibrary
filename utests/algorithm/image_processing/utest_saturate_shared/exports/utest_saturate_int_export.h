
#ifndef UTEST_SATURATE_INT_EXPORT_H
#define UTEST_SATURATE_INT_EXPORT_H

#ifdef UTEST_SATURATE_INT_STATIC_DEFINE
#  define UTEST_SATURATE_INT_EXPORT
#  define UTEST_SATURATE_INT_NO_EXPORT
#else
#  ifndef UTEST_SATURATE_INT_EXPORT
#    ifdef utest_saturate_int_EXPORTS
        /* We are building this library */
#      define UTEST_SATURATE_INT_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define UTEST_SATURATE_INT_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef UTEST_SATURATE_INT_NO_EXPORT
#    define UTEST_SATURATE_INT_NO_EXPORT 
#  endif
#endif

#ifndef UTEST_SATURATE_INT_DEPRECATED
#  define UTEST_SATURATE_INT_DEPRECATED __declspec(deprecated)
#endif

#ifndef UTEST_SATURATE_INT_DEPRECATED_EXPORT
#  define UTEST_SATURATE_INT_DEPRECATED_EXPORT UTEST_SATURATE_INT_EXPORT UTEST_SATURATE_INT_DEPRECATED
#endif

#ifndef UTEST_SATURATE_INT_DEPRECATED_NO_EXPORT
#  define UTEST_SATURATE_INT_DEPRECATED_NO_EXPORT UTEST_SATURATE_INT_NO_EXPORT UTEST_SATURATE_INT_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef UTEST_SATURATE_INT_NO_DEPRECATED
#    define UTEST_SATURATE_INT_NO_DEPRECATED
#  endif
#endif

#endif /* UTEST_SATURATE_INT_EXPORT_H */
