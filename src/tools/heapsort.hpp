#pragma once

#include <assert.h>
#include "tools/types.hpp"


///  Index corresponding to the maximmum value
///    @param[in] a : vector of values
///    @param[in] n : length of vector a
///    @param[in] i : first index
///    @param[in] j : second index
///    @param[in] k : third index
///    @return index corresponding to the largest value in vector a
///////////////////////////////////////////////////////////////////
template <typename type>
inline long max (const vector<type>& a, const long n, const long i, const long j, const long k)
{
    long m = i;

    if (j < n) {if (a[j] > a[m]) {m = j;}}
    if (k < n) {if (a[k] > a[m]) {m = k;}}

    return m;
}


///  Downheap sort
///    @param[in] a : vector of values
///    @param[in] b : vector of indices
///    @param[in] n : length of vectors a and b
///    @param[in] i : starting index
///////////////////////////////////////////////
template <typename type>
inline void downheap (vector<type>& a, Size1& b, long n, long i)
{
    while (1)
    {
        long j = max (a, n, i, 2*i+1, 2*i+2);

        if (j == i) {break;}

        type temp1 = a[i];
        Size temp2 = b[i];

        a[i] = a[j];
        a[j] = temp1;

        b[i] = b[j];
        b[j] = temp2;

        i = j;
    }
}


///  Heapsort of a vector (a) which keeps track of the original indices (b)
///    @param[in] a : vector of values to sort
///    @param[in] b : vector of original indices
///////////////////////////////////////////////////////////////////////////
template <typename type>
inline void heapsort (vector<type>& a, Size1& b)
{
    // Get vector length
    const long n = a.size();   // long required, because of for loop below!

    // Assert that both vectors have the same size
    assert (n == b.size());

    for (long i = (n-2)/2; i >=0 ; i--) // Warning: long is required here, couting down!!!
    {
        downheap (a, b, n, i);
    }

    for (long i = 0; i < n; i++)
    {
        type temp1 = a[n-i-1];
        Size temp2 = b[n-i-1];

        a[n-i-1] = a[0];
        a[0]     = temp1;

        b[n-i-1] = b[0];
        b[0]     = temp2;

        downheap (a, b, n-i-1, 0);
    }
}
