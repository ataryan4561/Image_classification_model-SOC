#include"header_mm.hpp"
#include"data.h"
void neural_nn2(float *input,float *output)
{
	#pragma HLS INTERFACE s_axilite port=return
	#pragma HLS INTERFACE m_axi depth=12288 port=input offset=slave
	#pragma HLS INTERFACE m_axi depth=1 port=output offset=slave
	float x_input[12288]={};
	memcpy(x_input,(const float*)input,12288*sizeof(float));
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
	memcpy((float*)output,layer_4,1*sizeof(float));
}
