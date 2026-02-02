#include <iostream>
using namespace std;

int main() {
    int code = 2;

    switch (code) {
        case 1: {
            cout << "one\n";
            break;
        }
        case 2: {
            cout << "two\n";
            break;
        }
        default: {
            cout << "other\n";
            break;
        }
    }

    return 0;
}

