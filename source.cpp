#include"header.hpp"
#include"data.h"
#define ffloat half
void neural_nn1(hls::stream<axis_data> &input,hls::stream<axis_data> &output)
{
	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis register both port=input
	#pragma HLS INTERFACE axis register both port=output
	float x_input[12288]={};
//	#pragma HLS ARRAY_PARTITION variable=x_input complete dim=1
	axis_data in,out;
	for(int i=0; i<12288; i++)
	{
		#pragma HLS PIPELINE
		in = input.read();
		x_input[i]=in.data;
	}
//	#pragma HLS ARRAY_PARTITION variable=W1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=W2 complete dim=2
	#pragma HLS ARRAY_PARTITION variable=W3 complete dim=2
	#pragma HLS ARRAY_PARTITION variable=W4 complete dim=2
	float layer_1[20]={};
	#pragma HLS ARRAY_PARTITION variable=layer_1 complete dim=1
	float layer_2[7]={};
	#pragma HLS ARRAY_PARTITION variable=layer_2 complete dim=1
	float layer_3[5]={};
	#pragma HLS ARRAY_PARTITION variable=layer_3 complete dim=1
	float layer_4[1]={};
	#pragma HLS ARRAY_PARTITION variable=layer_4 complete dim=1
	for(int i=0; i<20; i++)
	{
//		#pragma HLS UNROLL factor = 2
//		#pragma HLS PIPELINE
		for(int j=0; j<12288; j++)
		{
			#pragma HLS UNROLL factor = 300
			#pragma HLS PIPELINE
			layer_1[i]+=x_input[j]*W1[i][j];
		}
		layer_1[i]+=b1[i];
		if(layer_1[i]<=0)
			layer_1[i]=0;
//		layer_1[i]=max(0,layer_1[i]);
	}
	for(int i=0; i<7; i++)
	{
		#pragma HLS UNROLL
//		factor = 2
		#pragma HLS PIPELINE
		for(int j=0; j<20; j++)
		{
			#pragma HLS UNROLL
//			factor = 4
			#pragma HLS PIPELINE
			layer_2[i]+=layer_1[j]*W2[i][j];
		}
		layer_2[i]+=b2[i];
		if(layer_2[i]<=0)
			layer_2[i]=0;
//		layer_2[i]=max(0,layer_2[i]);
	}
	for(int i=0; i<5; i++)
	{
		#pragma HLS UNROLL
//		factor = 2
		#pragma HLS PIPELINE
		for(int j=0; j<7; j++)
		{
			#pragma HLS UNROLL
//			factor = 4
			#pragma HLS PIPELINE
			layer_3[i]+=layer_2[j]*W3[i][j];
		}
		layer_3[i]+=b3[i];
		if(layer_3[i]<=0)
			layer_3[i]=0;
//		layer_3[i]=max(0,layer_3[i]);
	}
	for(int i=0; i<1; i++)
	{
		#pragma HLS UNROLL
//		factor = 2
		#pragma HLS PIPELINE
		for(int j=0; j<5; j++)
		{
			#pragma HLS UNROLL
//			factor = 4
			#pragma HLS PIPELINE
			layer_4[i]+=layer_3[j]*W4[i][j];
		}
		layer_4[i]+=b4[i];
		layer_4[i] = 1.0/(1.0 + hls::exp(-1*layer_4[i]));
	}
	if(layer_4[0]>0.5)
	{
		layer_4[0]=1;
	}
	else
	{
		layer_4[0]=0;
	}
	out.data = layer_4[0];
	out.last=1;
	output.write(out);
}
