#include <iostream>
using namespace std;

int main() {
    int x = 7;

    if (x < 0) {
        cout << "negative\n";
    } else if (x == 0) {
        cout << "zero\n";
    } else {
        cout << "positive\n";
    }

    return 0;
}

