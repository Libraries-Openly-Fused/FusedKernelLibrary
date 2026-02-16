
#ifndef UTEST_SATURATE_ULONG_CPP_EXPORT_H
#define UTEST_SATURATE_ULONG_CPP_EXPORT_H

#ifdef UTEST_SATURATE_ULONG_CPP_STATIC_DEFINE
#  define UTEST_SATURATE_ULONG_CPP_EXPORT
#  define UTEST_SATURATE_ULONG_CPP_NO_EXPORT
#else
#  ifndef UTEST_SATURATE_ULONG_CPP_EXPORT
#    ifdef utest_saturate_ulong_cpp_EXPORTS
        /* We are building this library */
#      define UTEST_SATURATE_ULONG_CPP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define UTEST_SATURATE_ULONG_CPP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef UTEST_SATURATE_ULONG_CPP_NO_EXPORT
#    define UTEST_SATURATE_ULONG_CPP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef UTEST_SATURATE_ULONG_CPP_DEPRECATED
#  define UTEST_SATURATE_ULONG_CPP_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef UTEST_SATURATE_ULONG_CPP_DEPRECATED_EXPORT
#  define UTEST_SATURATE_ULONG_CPP_DEPRECATED_EXPORT UTEST_SATURATE_ULONG_CPP_EXPORT UTEST_SATURATE_ULONG_CPP_DEPRECATED
#endif

#ifndef UTEST_SATURATE_ULONG_CPP_DEPRECATED_NO_EXPORT
#  define UTEST_SATURATE_ULONG_CPP_DEPRECATED_NO_EXPORT UTEST_SATURATE_ULONG_CPP_NO_EXPORT UTEST_SATURATE_ULONG_CPP_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef UTEST_SATURATE_ULONG_CPP_NO_DEPRECATED
#    define UTEST_SATURATE_ULONG_CPP_NO_DEPRECATED
#  endif
#endif

#endif /* UTEST_SATURATE_ULONG_CPP_EXPORT_H */
