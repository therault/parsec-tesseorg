/**
 *
 * @file flops.h
 *
 *  File provided by Univ. of Tennessee,
 *
 * @version 1.0.0
 * @author Damien Genet
 * @date 2016-10-16
 *
 **/
/*
 * This file provide the flops formula for the summa kernel Each macro uses the
 * same size parameters as the function associated and provide one formula for
 * additions and one for multiplications. Example to use these macros:
 *
 *    FLOPS_ZSUMMA( m, n, k )
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */
#ifndef _FLOPS_H_
#define _FLOPS_H_


#define FMULS_SUMMA(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_SUMMA(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))

/*
 * Level 3 BLAS
 */
#define FLOPS_ZSUMMA(__m, __n, __k) (6. * FMULS_SUMMA((__m), (__n), (__k)) + 2.0 * FADDS_SUMMA((__m), (__n), (__k)) )
#define FLOPS_CSUMMA(__m, __n, __k) (6. * FMULS_SUMMA((__m), (__n), (__k)) + 2.0 * FADDS_SUMMA((__m), (__n), (__k)) )
#define FLOPS_DSUMMA(__m, __n, __k) (     FMULS_SUMMA((__m), (__n), (__k)) +       FADDS_SUMMA((__m), (__n), (__k)) )
#define FLOPS_SSUMMA(__m, __n, __k) (     FMULS_SUMMA((__m), (__n), (__k)) +       FADDS_SUMMA((__m), (__n), (__k)) )


#endif /* _FLOPS_H_ */
