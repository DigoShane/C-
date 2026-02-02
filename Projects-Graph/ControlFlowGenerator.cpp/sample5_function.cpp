#include <iostream>
using namespace std;

int abs_val(int x) {
    if (x < 0) {
        return -x;
    }
    return x;
}

int main() {
    int a = -12;
    cout << abs_val(a) << "\n";

    if (a == 0) {
        return 0;
    }

    cout << "done\n";
    return 0;
}

