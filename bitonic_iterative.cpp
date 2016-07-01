#include <iostream>
#include <vector>
using namespace std;

void bitonic_sort(vector<int> &v);
int greatestPowerOfTwoLessThan (int n);


int main() {
    int n;
    cin >> n;
    
    int m = greatestPowerOfTwoLessThan(n);
    
    vector<int> v(m);
    
    for (int i = 0; i < n; ++i ) {
        cin >> v[i];
    }
    for (int i = n; i < m; ++i) v[i] = 0x7FFFFFFF;
    
    
    bitonic_sort(v);
    
    for(int i : v) cout << i << " ";
    cout << endl;
}


void bitonic_sort(vector<int> &v) {
    for (int k = 2; k <= v.size(); k = 2 * k) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            for (int i = 0; i < v.size(); ++i) {
                int ixj = i ^ j;
                if ((ixj) > i) {
                    if ((i & k) == 0 and v[i] > v[ixj]) swap(v[i], v[ixj]);
                    if ((i & k) != 0 and v[i] < v[ixj]) swap(v[i], v[ixj]);
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