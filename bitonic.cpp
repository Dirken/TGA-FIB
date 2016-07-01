#include <iostream>
#include <vector>
using namespace std;

void bitonic_sort (vector<int> &v);
void bitonic_build (int l, int n, bool dir, vector<int> &v);
void bitonic_merge ( int l, int n, bool dir, vector<int> &v);
int greatestPowerOfTwoLessThan (int n);
void compare (int i, int j, bool dir, vector<int> &v);


int main() {
    int n;
    cin >> n;
    
    vector<int> v(n);
    
    for (int &i : v) cin >> i;
    
    bitonic_sort(v);
    for(int i : v) cout << i << " ";
    cout << endl;
    
}


void bitonic_sort (vector<int> &v) {
    bool ascending = true;
    bitonic_build(0, v.size(), ascending, v);
}

void bitonic_build (int l, int n, bool dir, vector<int> &v) {
    if (n > 1) {
        int m = n / 2;
        bitonic_build(l, m, !dir, v);
        bitonic_build(l + m, n - m, dir, v);
        bitonic_merge(l, n, dir, v);
    }
}

void bitonic_merge ( int l, int n, bool dir, vector<int> &v) {
    if (n > 1) {
        int m = greatestPowerOfTwoLessThan(n);
        for (int i = l; i < l + n - m; ++i) {
            compare(i, i+m, dir, v);
            bitonic_merge(l, m, dir, v);
            bitonic_merge(l + m, n - m, dir, v);
        }
    }
}

int greatestPowerOfTwoLessThan (int n) {
    int k = 1;
    while (k < n) k = k << 1;
    return k >> 1;
}

void compare (int i, int j, bool dir, vector<int> &v) {
    if(dir == (v[i] > v[j])) swap(v[i], v[j]);
}







