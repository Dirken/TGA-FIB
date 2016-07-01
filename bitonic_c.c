#include <stdio.h>

void swap (int i, int j, int v[]) {
    int aux = v[j];
    v[j] = v[i];
    v[i] = aux;
    
}

void bitonic_sort(int v[], int n) {
    int k, j, i;
    for (k = 2; k <= n; k = 2 * k) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            for (i = 0; i < n; ++i) {
                int ixj = i ^ j;
                if ((ixj) > i) {
                    if ((i & k) == 0 && v[i] > v[ixj]) swap(i, ixj, v);
                    if ((i & k) != 0 && v[i] < v[ixj]) swap(i, ixj, v);
                }
            }
        }
    }
}

int greatestPowerOfTwoLessThan (int n) {
    int k = 1;
    while (k < n) k = k << 1;
    return k;
}


int main() {
    int n;
    scanf("%d", &n);
    
    int m = greatestPowerOfTwoLessThan(n);
    
    int v[m];
    int x, i;
    for (i = 0; i < n; ++i ) {
        scanf("%d", &x);
        v[i] = x;
    }
    for (i = n; i < m; ++i)
        v[i] = 0x7FFFFFFF;
    
    
    bitonic_sort(v, m);
    
    for (i = 0; i < m; ++i) {
        printf("%d ", v[i]);
    }
    printf("\n");
}


