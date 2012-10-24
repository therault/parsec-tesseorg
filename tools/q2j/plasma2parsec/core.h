#define QUARK_CORE_zaxpy(quark, task_flags, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_zaxpy_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha), VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             INOUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma CORE_zaxpy A B

#define QUARK_CORE_zbrdalg(quark, task_flags, uplo, N, NB, A, V, TAU, i, j, m, grsiz, BAND, PCOL, ACOL, MCOL) {\
    QUARK_Insert_Task((quark), CORE_zbrdalg_quark,   (task_flags),\
        sizeof(int),               &(uplo),               VALUE,\
        sizeof(int),                  &(N),               VALUE,\
        sizeof(int),                 &(NB),               VALUE,\
        sizeof(PLASMA_desc),           (A),               NODEP,\
        sizeof(PLASMA_Complex64_t),    (V),               NODEP,\
        sizeof(PLASMA_Complex64_t),    (TAU),               NODEP,\
        sizeof(int),                  &(i),               VALUE,\
        sizeof(int),                  &(j),               VALUE,\
        sizeof(int),                  &(m),               VALUE,\
        sizeof(int),              &(grsiz),               VALUE,\
        sizeof(int),                (PCOL),               INPUT,\
        sizeof(int),                (ACOL),               INPUT,\
        sizeof(int),                (MCOL),              OUTPUT | LOCALITY,\
        0);}
#pragma CORE_zbrdalg PCOL ACOL

#define QUARK_CORE_zgelqt(quark, task_flags, m, n, ib, nb, A, lda, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zgelqt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma CORE_zgelqt A T

#define QUARK_CORE_zgemm(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zgemm A B C

#define QUARK_CORE_zgemm2(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT | LOCALITY | GATHERV,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zgemm2 A B

#define QUARK_CORE_zgemm_f2(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc, fake1, szefake1, flag1, fake2, szefake2, flag2) {\
    QUARK_Insert_Task((quark), CORE_zgemm_f2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1),             flag1,\
        sizeof(PLASMA_Complex64_t)*szefake2, (fake2),             flag2,\
        0);}
#pragma CORE_zgemm_f2 A B C

#define QUARK_CORE_zgemm_p2(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_p2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t*),         (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*ldc*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zgemm_p2 A B C

#define QUARK_CORE_zgemm_p3(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_p3_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*ldb*nb,   (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t*),         (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zgemm_p3 A B C

#define QUARK_CORE_zgemm_p2f1(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc, fake1, szefake1, flag1) {\
    QUARK_Insert_Task((quark), CORE_zgemm_p2f1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t*),         (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*ldc*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1),             flag1,\
        0);}
#pragma CORE_zgemm_p2f1 A B C

#define QUARK_CORE_zgeqrt(quark, task_flags, m, n, ib, nb, A, lda, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zgeqrt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma CORE_zgeqrt A T

#define QUARK_CORE_zgessm(quark, task_flags, m, n, k, ib, nb, IPIV, L, ldl, A, lda) {\
    QUARK_Insert_Task((quark), CORE_zgessm_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(int)*nb,                      (IPIV),          INPUT,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (L),             INPUT | QUARK_REGION_L,\
        sizeof(int),                        &(ldl),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        0);}
#pragma CORE_zgessm IPIV A

#define QUARK_CORE_zgetrf(quark, task_flags, m, n, nb, A, lda, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT | LOCALITY,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_sequence*),           &(sequence),      VALUE,\
        sizeof(PLASMA_request*),            &(request),       VALUE,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma CORE_zgetrf IPIV

#define QUARK_CORE_zgetrf_incpiv(quark, task_flags, m, n, ib, nb, A, lda, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_incpiv_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(int),                        &(ib),            VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_sequence*),           &(sequence),      VALUE,\
        sizeof(PLASMA_request*),            &(request),       VALUE,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma CORE_zgetrf_incpiv A IPIV

#define QUARK_CORE_zgetrf_reclap(quark, task_flags, m, n, nb, A, lda, IPIV, sequence, request, check_info, iinfo, nbthread) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_reclap_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_sequence*),           &(sequence),      VALUE,\
        sizeof(PLASMA_request*),            &(request),       VALUE,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        sizeof(int),                        &(nbthread),      VALUE,\
        0);}
#pragma CORE_zgetrf_reclap A IPIV

#define QUARK_CORE_zgetrf_rectil(quark, task_flags, A, Amn, size, IPIV, sequence, request, check_info, iinfo, nbthread) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_rectil_quark, (task_flags),\
        sizeof(PLASMA_desc),                &(A),             VALUE,\
        sizeof(PLASMA_Complex64_t)*size,     (Amn),               INOUT,\
        sizeof(int)*A.n,                     (IPIV),              OUTPUT,\
        sizeof(PLASMA_sequence*),           &(sequence),      VALUE,\
        sizeof(PLASMA_request*),            &(request),       VALUE,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        sizeof(int),                        &(nbthread),      VALUE,\
        0);}
#pragma CORE_zgetrf_rectil Amn IPIV

#define QUARK_CORE_zgetrip(quark, task_flags, m, n, A, szeA) {\
    QUARK_Insert_Task((quark), CORE_zgetrip_quark, (task_flags),\
        sizeof(int),                     &(m),   VALUE,\
        sizeof(int),                     &(n),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA, (A),        INOUT,\
        sizeof(PLASMA_Complex64_t)*szeA, (NULL),     SCRATCH,\
        0);}
#pragma CORE_zgetrip A

#define QUARK_CORE_zgetrip_f1(quark, task_flags, m, n, A, szeA, fake, szeF, paramF) {\
    QUARK_Insert_Task(\
        quark, (CORE_zgetrip_f1_quark), task_flags,\
        sizeof(int),                     &(m),   VALUE,\
        sizeof(int),                     &(n),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA, (A),        INOUT,\
        sizeof(PLASMA_Complex64_t)*szeA, (NULL),     SCRATCH,\
        sizeof(PLASMA_Complex64_t)*szeF, (fake),     paramF,\
        0);}
#pragma CORE_zgetrip_f1 A

#define QUARK_CORE_zgetrip_f2(quark, task_flags, m, n, A, szeA, fake1, szeF1, paramF1, fake2, szeF2, paramF2) {\
    QUARK_Insert_Task(\
        quark, (CORE_zgetrip_f2_quark), task_flags,\
        sizeof(int),                     &(m),   VALUE,\
        sizeof(int),                     &(n),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA, (A),        INOUT,\
        sizeof(PLASMA_Complex64_t)*szeA, (NULL),     SCRATCH,\
        sizeof(PLASMA_Complex64_t)*szeF1, (fake1),     paramF1,\
        sizeof(PLASMA_Complex64_t)*szeF2, (fake2),     paramF2,\
        0);}
#pragma CORE_zgetrip_f2 A

#define QUARK_CORE_zhegst(quark, task_flags, itype, uplo, n, A, lda, B, ldb, sequence, request, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zhegst_quark, (task_flags),\
        sizeof(int),                        &(itype),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*n*n,    (A),                 INOUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*n*n,    (B),                 INOUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_sequence*),           &(sequence),  VALUE,\
        sizeof(PLASMA_request*),            &(request),   VALUE,\
        sizeof(int),                        &(iinfo),     VALUE,\
        0);}
#pragma CORE_zhegst A B

#define QUARK_CORE_zhemm(quark, task_flags, side, uplo, m, n, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zhemm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),    VALUE,\
        sizeof(PLASMA_enum),                &(uplo),    VALUE,\
        sizeof(int),                        &(m),       VALUE,\
        sizeof(int),                        &(n),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),               INPUT,\
        sizeof(int),                        &(lda),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),               INPUT,\
        sizeof(int),                        &(ldb),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),               INOUT,\
        sizeof(int),                        &(ldc),     VALUE,\
        0);}
#pragma CORE_zhemm A B C

#define QUARK_CORE_zher2k(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zher2k_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(double),                     &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zher2k A B C

#define QUARK_CORE_zherfb(quark, task_flags, uplo, n, k, ib, nb, A, lda, T, ldt, C, ldc) {\
    QUARK_Insert_Task(\
        quark, (CORE_zherfb_quark), task_flags,\
        sizeof(PLASMA_enum),                     &(uplo),  VALUE,\
        sizeof(int),                             &(n),     VALUE,\
        sizeof(int),                             &(k),     VALUE,\
        sizeof(int),                             &(ib),    VALUE,\
        sizeof(int),                             &(nb),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,        (A),          uplo == PlasmaUpper ? INPUT|QUARK_REGION_U : INPUT|QUARK_REGION_L,\
        sizeof(int),                             &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,        (T),          INPUT,\
        sizeof(int),                             &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,        (C),          INOUT,\
        sizeof(int),                             &(ldc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*2*nb*nb,    (NULL),         SCRATCH,\
        sizeof(int),                             &(nb),    VALUE,\
        0);}
#pragma CORE_zherfb T C

#define QUARK_CORE_zherk(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zherk_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(double),                     &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(double),                     &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zherk A C

#define QUARK_CORE_zlacpy(quark, task_flags, uplo, m, n, nb, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_zlacpy_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             OUTPUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma CORE_zlacpy A B

#define QUARK_CORE_zlag2c(quark, task_flags, m, n, nb, A, lda, B, ldb, sequence, request) {\
    QUARK_Insert_Task((quark), CORE_zlag2c_quark, (task_flags),\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex32_t)*nb*nb,    (B),                 OUTPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_sequence*),           &(sequence),  VALUE,\
        sizeof(PLASMA_request*),            &(request),   VALUE,\
        0);}
#pragma CORE_zlag2c A B

#define QUARK_CORE_clag2z(quark, task_flags, m, n, nb, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_clag2z_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex32_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             INOUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma CORE_clag2z A B

#define QUARK_CORE_zlange(quark, task_flags, norm, M, N, A, LDA, szeA, szeW, result) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlange_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        0);}
#pragma CORE_zlange A result

#define QUARK_CORE_zlange_f1(quark, task_flags, norm, M, N, A, LDA, szeA, szeW, result, fake, szeF) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlange_f1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        sizeof(double)*szeF,                 (fake),          OUTPUT | GATHERV,\
        0);}
#pragma CORE_zlange_f1 A result

#define QUARK_CORE_zlanhe(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlanhe_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                     (result),         OUTPUT,\
        0);}
#pragma CORE_zlanhe A result

#define QUARK_CORE_zlanhe_f1(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result, fake, szeF) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlanhe_f1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm), VALUE,\
        sizeof(PLASMA_enum),                &(uplo), VALUE,\
        sizeof(int),                        &(N),    VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),  VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        sizeof(double)*szeF,                 (fake),          OUTPUT | GATHERV,\
        0);}
#pragma CORE_zlanhe_f1 A result

#define QUARK_CORE_zlansy(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlansy_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        0);}
#pragma CORE_zlansy A result

#define QUARK_CORE_zlansy_f1(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result, fake, szeF) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlansy_f1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        sizeof(double)*szeF,                 (fake),          OUTPUT | GATHERV,\
        0);}
#pragma CORE_zlansy_f1 A result

#define QUARK_CORE_zlaset2(quark, task_flags, uplo, M, N, alpha, A, LDA) {\
    QUARK_Insert_Task((quark), CORE_zlaset2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha), VALUE,\
        sizeof(PLASMA_Complex64_t)*M*N,     (A),      OUTPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        0);}
#pragma CORE_zlaset2 A

#define QUARK_CORE_zlaset(quark, task_flags, uplo, M, N, alpha, beta, A, LDA) {\
    QUARK_Insert_Task((quark), CORE_zlaset_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha), VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),  VALUE,\
        sizeof(PLASMA_Complex64_t)*M*N,     (A),      OUTPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        0);}
#pragma CORE_zlaset A

#define QUARK_CORE_zlaswp(quark, task_flags, n, A, lda, i1, i2, ipiv, inc) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_quark), task_flags,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n,  (A),        INOUT | LOCALITY,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(i1),   VALUE,\
        sizeof(int),                      &(i2),   VALUE,\
        sizeof(int)*n,                     (ipiv),     INPUT,\
        sizeof(int),                      &(inc),  VALUE,\
        0);}
#pragma CORE_zlaswp ipiv

#define QUARK_CORE_zlaswp_f2(quark, task_flags, n, A, lda, i1, i2, ipiv, inc, fake1, szefake1, flag1, fake2, szefake2, flag2) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_f2_quark), task_flags,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n,    (A),         INOUT | LOCALITY,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(int),                        &(i1),    VALUE,\
        sizeof(int),                        &(i2),    VALUE,\
        sizeof(int)*n,                       (ipiv),      INPUT,\
        sizeof(int),                        &(inc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1),     flag1,\
        sizeof(PLASMA_Complex64_t)*szefake2, (fake2),     flag2,\
        0);}
#pragma CORE_zlaswp_f2 ipiv

#define QUARK_CORE_zlaswp_ontile(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, fakepanel) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_ontile_quark), task_flags,\
        sizeof(PLASMA_desc),              &(descA),     VALUE,\
        sizeof(PLASMA_Complex64_t)*1,      (Aij),           INOUT | LOCALITY,\
        sizeof(int),                      &(i1),        VALUE,\
        sizeof(int),                      &(i2),        VALUE,\
        sizeof(int)*(i2-i1+1)*abs(inc),   (ipiv),           INPUT,\
        sizeof(int),                      &(inc),       VALUE,\
        sizeof(PLASMA_Complex64_t)*1,      (fakepanel),     INOUT,\
        0);}
#pragma CORE_zlaswp_ontile ipiv fakepanel

#define QUARK_CORE_zlaswp_ontile_f2(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, fake1, szefake1, flag1, fake2, szefake2, flag2) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_ontile_f2_quark), task_flags,\
        sizeof(PLASMA_desc),                &(descA), VALUE,\
        sizeof(PLASMA_Complex64_t)*1,        (Aij),       INOUT | LOCALITY,\
        sizeof(int),                        &(i1),    VALUE,\
        sizeof(int),                        &(i2),    VALUE,\
        sizeof(int)*(i2-i1+1)*abs(inc),      (ipiv),      INPUT,\
        sizeof(int),                        &(inc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1), flag1,\
        sizeof(PLASMA_Complex64_t)*szefake2, (fake2), flag2,\
        0);}
#pragma CORE_zlaswp_ontile_f2 ipiv

#define QUARK_CORE_zswptr_ontile(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, Akk, ldak) {\
    QUARK_Insert_Task(\
        quark, (CORE_zswptr_ontile_quark), task_flags,\
        sizeof(PLASMA_desc),              &(descA), VALUE,\
        sizeof(PLASMA_Complex64_t)*1,      (Aij),       INOUT | LOCALITY,\
        sizeof(int),                      &(i1),    VALUE,\
        sizeof(int),                      &(i2),    VALUE,\
        sizeof(int)*(i2-i1+1)*abs(inc),    (ipiv),      INPUT,\
        sizeof(int),                      &(inc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ldak,   (Akk),       INPUT,\
        sizeof(int),                      &(ldak),  VALUE,\
        0);}
#pragma CORE_zswptr_ontile ipiv Akk

#define QUARK_CORE_zlauum(quark, task_flags, uplo, n, nb, A, lda) {\
    QUARK_Insert_Task((quark), CORE_zlauum_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        0);}
#pragma CORE_zlauum A

#define QUARK_CORE_zplghe(quark, task_flags, bump, m, n, A, lda, bigM, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zplghe_quark, (task_flags),\
        sizeof(double),                   &(bump), VALUE,\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),         OUTPUT,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(bigM), VALUE,\
        sizeof(int),                      &(m0),   VALUE,\
        sizeof(int),                      &(n0),   VALUE,\
        sizeof(unsigned long long int),   &(seed), VALUE,\
        0);}
#pragma CORE_zplghe A

#define QUARK_CORE_zplgsy(quark, task_flags, bump, m, n, A, lda, bigM, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zplgsy_quark, (task_flags),\
        sizeof(PLASMA_Complex64_t),       &(bump), VALUE,\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),         OUTPUT,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(bigM), VALUE,\
        sizeof(int),                      &(m0),   VALUE,\
        sizeof(int),                      &(n0),   VALUE,\
        sizeof(unsigned long long int),   &(seed), VALUE,\
        0);}
#pragma CORE_zplgsy A

#define QUARK_CORE_zplrnt(quark, task_flags, m, n, A, lda, bigM, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zplrnt_quark, (task_flags),\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),         OUTPUT,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(bigM), VALUE,\
        sizeof(int),                      &(m0),   VALUE,\
        sizeof(int),                      &(n0),   VALUE,\
        sizeof(unsigned long long int),   &(seed), VALUE,\
        0);}
#pragma CORE_zplrnt A

#define QUARK_CORE_zpotrf(quark, task_flags, uplo, n, nb, A, lda, sequence, request, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zpotrf_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INOUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_sequence*),           &(sequence),  VALUE,\
        sizeof(PLASMA_request*),            &(request),   VALUE,\
        sizeof(int),                        &(iinfo),     VALUE,\
        0);}
#pragma CORE_zpotrf A

#define QUARK_CORE_zshiftw(quark, task_flags, s, cl, m, n, L, A, W) {\
    QUARK_Insert_Task((quark), CORE_zshiftw_quark, (task_flags),\
        sizeof(int),                      &(s),   VALUE,\
        sizeof(int),                      &(cl),  VALUE,\
        sizeof(int),                      &(m),   VALUE,\
        sizeof(int),                      &(n),   VALUE,\
        sizeof(int),                      &(L),   VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n*L, (A),        INOUT,\
        sizeof(PLASMA_Complex64_t)*L,     (W),        INPUT,\
        0);}
#pragma CORE_zshiftw A W

#define QUARK_CORE_zshift(quark, task_flags, s, m, n, L, A) {\
    QUARK_Insert_Task((quark), CORE_zshift_quark, (task_flags),\
        sizeof(int),                      &(s),    VALUE,\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(int),                      &(L),    VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n*L, (A),        INOUT | GATHERV,\
        sizeof(PLASMA_Complex64_t)*L,     (NULL),     SCRATCH,\
        0);}
#pragma CORE_zshift

#define QUARK_CORE_zssssm(quark, task_flags, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, L1, ldl1, L2, ldl2, IPIV) {\
    QUARK_Insert_Task((quark), CORE_zssssm_quark, (task_flags),\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (L1),            INPUT,\
        sizeof(int),                        &(ldl1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (L2),            INPUT,\
        sizeof(int),                        &(ldl2),  VALUE,\
        sizeof(int)*nb,                      (IPIV),          INPUT,\
        0);}
#pragma CORE_zssssm A1 L1 L2 IPIV

#define QUARK_CORE_zswpab(quark, task_flags, i, n1, n2, A, szeA) {\
    QUARK_Insert_Task(\
        quark, (CORE_zswpab_quark), task_flags,\
        sizeof(int),                           &(i),   VALUE,\
        sizeof(int),                           &(n1),  VALUE,\
        sizeof(int),                           &(n2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,       (A),            INOUT,\
        sizeof(PLASMA_Complex64_t)*min(n1,n2), (NULL),         SCRATCH,\
        0);}
#pragma CORE_zswpab A

#define QUARK_CORE_zsymm(quark, task_flags, side, uplo, m, n, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zsymm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),    VALUE,\
        sizeof(PLASMA_enum),                &(uplo),    VALUE,\
        sizeof(int),                        &(m),       VALUE,\
        sizeof(int),                        &(n),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),               INPUT,\
        sizeof(int),                        &(lda),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),               INPUT,\
        sizeof(int),                        &(ldb),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),               INOUT,\
        sizeof(int),                        &(ldc),     VALUE,\
        0);}
#pragma CORE_zsymm A B C

#define QUARK_CORE_zsyr2k(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zsyr2k_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zsyr2k A B C

#define QUARK_CORE_zsyrk(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zsyrk_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma CORE_zsyrk A C

#define QUARK_CORE_ztrdalg(quark, task_flags, uplo, N, NB, A, V, TAU, i, j, m, grsiz, BAND, PCOL, ACOL, MCOL) {\
    QUARK_Insert_Task((quark), CORE_ztrdalg_quark,   (task_flags),\
        sizeof(int),               &(uplo),               VALUE,\
        sizeof(int),                  &(N),               VALUE,\
        sizeof(int),                 &(NB),               VALUE,\
        sizeof(PLASMA_desc),           (A),               NODEP,\
        sizeof(PLASMA_Complex64_t),    (V),               NODEP,\
        sizeof(PLASMA_Complex64_t),    (TAU),               NODEP,\
        sizeof(int),                  &(i),               VALUE,\
        sizeof(int),                  &(j),               VALUE,\
        sizeof(int),                  &(m),               VALUE,\
        sizeof(int),              &(grsiz),               VALUE,\
        sizeof(int),                (PCOL),               INPUT,\
        sizeof(int),                (ACOL),               INPUT,\
        sizeof(int),                (MCOL),               OUTPUT | LOCALITY,\
        0);}
#pragma CORE_ztrdalg PCOL ACOL

#define QUARK_CORE_ztrmm(quark, task_flags, side, uplo, transA, diag, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_ztrmm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INOUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma CORE_ztrmm A B

#define QUARK_CORE_ztrmm_p2(quark, task_flags, side, uplo, transA, diag, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_ztrmm_p2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t*),         (B),                 INOUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma CORE_ztrmm_p2 A B

#define QUARK_CORE_ztrsm(quark, task_flags, side, uplo, transA, diag, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_ztrsm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INOUT | LOCALITY,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma CORE_ztrsm A

#define QUARK_CORE_ztrtri(quark, task_flags, uplo, diag, n, nb, A, lda, sequence, request, iinfo) {\
    QUARK_Insert_Task(\
        quark, (CORE_ztrtri_quark), task_flags,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INOUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_sequence*),           &(sequence),  VALUE,\
        sizeof(PLASMA_request*),            &(request),   VALUE,\
        sizeof(int),                        &(iinfo),     VALUE,\
        0);}
#pragma CORE_ztrtri A

#define QUARK_CORE_ztslqt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_ztslqt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT | QUARK_REGION_D | QUARK_REGION_L,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma CORE_ztslqt T

#define QUARK_CORE_ztsmlq(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmlq_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma CORE_ztsmlq A1 V T

#define QUARK_CORE_ztsmlq_corner(quark, task_flags, m1, n1, m2, n2, m3, n3, k, ib, nb, A1, lda1, A2, lda2, A3, lda3, V, ldv, T, ldt) {\
    int ldwork = nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmlq_corner_quark, (task_flags),\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(m3),    VALUE,\
        sizeof(int),                        &(n3),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(int),                        &(nb),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A3),            INOUT,\
        sizeof(int),                        &(lda3),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*4*nb*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma CORE_ztsmlq_corner A1 A2 A3 V T

#define QUARK_CORE_ztsmlq_hetra1(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmlq_hetra1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma CORE_ztsmlq_hetra1 A1 A2 V T

#define QUARK_CORE_ztsmqr(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmqr_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma CORE_ztsmqr A1 V T

#define QUARK_CORE_ztsmqr_corner(quark, task_flags, m1, n1, m2, n2, m3, n3, k, ib, nb, A1, lda1, A2, lda2, A3, lda3, V, ldv, T, ldt) {\
    int ldwork = nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmqr_corner_quark, (task_flags),\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(m3),    VALUE,\
        sizeof(int),                        &(n3),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(int),                        &(nb),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A3),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda3),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*4*nb*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma CORE_ztsmqr_corner A2 V T

#define QUARK_CORE_ztsmqr_hetra1(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmqr_hetra1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma CORE_ztsmqr_hetra1 A1 A2 V T

#define QUARK_CORE_ztsqrt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_ztsqrt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT | QUARK_REGION_D | QUARK_REGION_U,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma CORE_ztsqrt T

#define QUARK_CORE_ztstrf(quark, task_flags, m, n, ib, nb, U, ldu, A, lda, L, ldl, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_ztstrf_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(int),                        &(ib),            VALUE,\
        sizeof(int),                        &(nb),            VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (U),                     INOUT | QUARK_REGION_D | QUARK_REGION_U,\
        sizeof(int),                        &(ldu),           VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT | LOCALITY,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (L),                     OUTPUT,\
        sizeof(int),                        &(ldl),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),                  SCRATCH,\
        sizeof(int),                        &(nb),            VALUE,\
        sizeof(PLASMA_sequence*),           &(sequence),      VALUE,\
        sizeof(PLASMA_request*),            &(request),       VALUE,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma CORE_ztstrf L IPIV

#define QUARK_CORE_zttlqt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zttlqt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma CORE_zttlqt T

#define QUARK_CORE_zttmlq(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_zttmlq_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork),    VALUE,\
        0);}
#pragma CORE_zttmlq A1 A2 T

#define QUARK_CORE_zttmqr(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_zttmqr_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork),    VALUE,\
        0);}
#pragma CORE_zttmqr A1 A2 T

#define QUARK_CORE_zttqrt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zttqrt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma CORE_zttqrt T

#define QUARK_CORE_zunmlq(quark, task_flags, side, trans, m, n, k, ib, nb, A, lda, T, ldt, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zunmlq_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT | QUARK_REGION_U,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),             INOUT,\
        sizeof(int),                        &(ldc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(nb),    VALUE,\
        0);}
#pragma CORE_zunmlq T C

#define QUARK_CORE_zunmqr(quark, task_flags, side, trans, m, n, k, ib, nb, A, lda, T, ldt, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zunmqr_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT | QUARK_REGION_L,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),             INOUT,\
        sizeof(int),                        &(ldc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(nb),    VALUE,\
        0);}
#pragma CORE_zunmqr T C
