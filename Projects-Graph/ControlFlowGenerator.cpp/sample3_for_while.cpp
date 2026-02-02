#include <iostream>
using namespace std;

int main() {
    int sum = 0;

    for (int i = 0; i < 10; i++) {
        if (i == 3) {
            continue;
        }
        sum += i;
        if (sum > 20) {
            break;
        }
    }

    int x = 5;
    while (x > 0) {
        x--;
    }

    cout << sum << "\n";
    return 0;
}

