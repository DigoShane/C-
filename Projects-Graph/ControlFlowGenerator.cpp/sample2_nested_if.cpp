#include <iostream>
using namespace std;

int main() {
    int age = 20;
    bool has_id = true;

    if (age >= 18) {
        if (has_id) {
            cout << "Allowed\n";
        } else {
            cout << "Need ID\n";
        }
    } else {
        cout << "Too young\n";
    }

    return 0;
}

