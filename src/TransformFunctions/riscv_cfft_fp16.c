#include "riscv_math.h"

extern void riscv_radix4_butterfly_fp16(
    float16_t * pSrc,
    uint32_t fftLen,
    float16_t * pCoef,
    uint32_t twidCoefModifier);

extern void riscv_radix4_butterfly_inverse_fp16(
    float16_t * pSrc,
    uint32_t fftLen,
    float16_t * pCoef,
    uint32_t twidCoefModifier);

extern void riscv_bitreversal_16(
    uint16_t * pSrc,
    const uint16_t bitRevLen,
    const uint16_t * pBitRevTable);

void riscv_cfft_radix4by2_fp16(
    float16_t * pSrc,
    uint32_t fftLen,
    const float16_t * pCoef);

void riscv_cfft_radix4by2_inverse_fp16(
    float16_t * pSrc,
    uint32_t fftLen,
    const float16_t * pCoef);

void riscv_cfft_fp16( 
    const riscv_cfft_instance_fp16 * S, 
    float16_t * p1,
    uint8_t ifftFlag,
    uint8_t bitReverseFlag)
{
    uint32_t L = S->fftLen;

    if(ifftFlag == 1u)
    {
        switch (L) 
        {
        case 16: 
        case 64:
        case 256:
        case 1024:
        case 4096:
            riscv_radix4_butterfly_inverse_fp16(p1, L, (float16_t*)S->pTwiddle, 1);
            break;
            
        case 32:
        case 128:
        case 512:
        case 2048:
            riscv_cfft_radix4by2_inverse_fp16(p1, L, S->pTwiddle);
            break;
        }  
    }
    else
    {
        switch (L) 
        {
        case 16: 
        case 64:
        case 256:
        case 1024:
        case 4096:
            riscv_radix4_butterfly_fp16(p1, L, (float16_t*)S->pTwiddle, 1);
            break;
            
        case 32:
        case 128:
        case 512:
            riscv_cfft_radix4by2_fp16(p1, L, S->pTwiddle);
            break;
        case 2048:
            riscv_cfft_radix4by2_fp16(p1, L, S->pTwiddle);
            break;
        }  
    }
    
    if(bitReverseFlag)
        riscv_bitreversal_16((uint16_t*)p1, S->bitRevLength, S->pBitRevTable);    
}

void riscv_cfft_radix4by2_fp16(
    float16_t * pSrc,
    uint32_t fftLen,
    const float16_t * pCoef) 
{    
    uint32_t i;
    uint32_t n2;
    float16_t p0, p1, p2, p3;

    uint32_t ia, l;
    float16_t xt, yt, cosVal, sinVal;

    n2 = fftLen >> 1; 

    ia = 0;
    for (i = 0; i < n2; i++)
    {
        cosVal = pCoef[ia * 2];
        sinVal = pCoef[(ia * 2) + 1];
        ia++;
        
        l = i + n2;        
        
        xt = (pSrc[2 * i] - pSrc[2 * l]) / 2;
        pSrc[2 * i] = (pSrc[2 * i] + pSrc[2 * l]) / 2;
        
        yt = (pSrc[2 * i + 1] - pSrc[2 * l + 1]) / 2;
        pSrc[2 * i + 1] = (pSrc[2 * l + 1] + pSrc[2 * i + 1]) / 2;

        pSrc[2u * l] = (xt * cosVal + yt * sinVal);
        pSrc[2u * l + 1u] = (yt * cosVal - xt * sinVal);
    } 

    riscv_radix4_butterfly_fp16(pSrc, n2, (float16_t*)pCoef, 2u);
    riscv_radix4_butterfly_fp16(pSrc + fftLen, n2, (float16_t*)pCoef, 2u);

    for (i = 0; i < fftLen >> 1; i++)
    {
        p0 = pSrc[4*i+0];
        p1 = pSrc[4*i+1];
        p2 = pSrc[4*i+2];
        p3 = pSrc[4*i+3];
        
        p0 *= 2;
        p1 *= 2;
        p2 *= 2;
        p3 *= 2;
        
        pSrc[4*i+0] = p0;
        pSrc[4*i+1] = p1;
        pSrc[4*i+2] = p2;
        pSrc[4*i+3] = p3;
    }
}

void riscv_cfft_radix4by2_inverse_fp16(
    float16_t * pSrc,
    uint32_t fftLen,
    const float16_t * pCoef) 
{    
    uint32_t i;
    uint32_t n2;
    float16_t p0, p1, p2, p3;

    uint32_t ia, l;
    float16_t xt, yt, cosVal, sinVal;

    n2 = fftLen >> 1; 

    ia = 0;
    for (i = 0; i < n2; i++)
    {
        cosVal = pCoef[ia * 2];
        sinVal = pCoef[(ia * 2) + 1];
        ia++;
        
        l = i + n2;
        xt = (pSrc[2 * i] - pSrc[2 * l]) / 2;
        pSrc[2 * i] = (pSrc[2 * i] + pSrc[2 * l]) / 2;
        
        yt = (pSrc[2 * i + 1] - pSrc[2 * l + 1]) / 2;
        pSrc[2 * i + 1] = (pSrc[2 * l + 1] + pSrc[2 * i + 1]) / 2;
        
        pSrc[2u * l] = (xt * cosVal - yt * sinVal);
        pSrc[2u * l + 1u] = (yt * cosVal + xt * sinVal);
    } 

    riscv_radix4_butterfly_inverse_fp16(pSrc, n2, (float16_t*)pCoef, 2u);
    riscv_radix4_butterfly_inverse_fp16(pSrc + fftLen, n2, (float16_t*)pCoef, 2u);

    for (i = 0; i < fftLen >> 1; i++)
    {
        p0 = pSrc[4*i+0];
        p1 = pSrc[4*i+1];
        p2 = pSrc[4*i+2];
        p3 = pSrc[4*i+3];
        
        p0 *= 2;
        p1 *= 2;
        p2 *= 2;
        p3 *= 2;
        
        pSrc[4*i+0] = p0;
        pSrc[4*i+1] = p1;
        pSrc[4*i+2] = p2;
        pSrc[4*i+3] = p3;
    }
}