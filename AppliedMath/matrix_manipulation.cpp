//  matrix_manipulation.cpp
//
//  Created by slasker1.
//
#include <iostream>
using namespace std;
int main(){
    int a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
    
    cout << "The first matrix is:" << endl << "\n";
    for(i=0; i<3; ++i) {
        for(j=0; j<3; ++j)
            cout << a[i][j] << " ";
        cout << endl;
    }
}
