#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 3; i++) {
        if (i == 1) {
            cout << "skip\n";
        } else {
            int code = i;
            switch (code) {
                case 0: {
                    cout << "zero\n";
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
        }
    }

    return 0;
}
