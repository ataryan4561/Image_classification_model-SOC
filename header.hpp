#include<bits/stdc++.h>
#include"ap_int.h"
#include"hls_stream.h"
#include<stdio.h>
#include<math.h>
#include<hls_math.h>
using namespace std;
struct axis_data
{
	float data;
	ap_int<1> last;
};
void neural_nn(hls::stream<axis_data> &input,hls::stream<axis_data> &output);
