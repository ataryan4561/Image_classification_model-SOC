#include"header_mm.hpp"
#include"data_2.hpp"
// Function to multiply two float numbers using bit manipulation
float multiply_floats(float f1, float f2) 
{
    // Step 1: Interpret the floats as 32-bit integers
    uint32_t a = *((uint32_t*)&f1);
    uint32_t b = *((uint32_t*)&f2);

    // Extract sign, exponent, and mantissa from both floats
    uint32_t sign1 = (a >> 31) & 1;
    uint32_t exp1 = (a >> 23) & 0xFF;
    uint32_t mant1 = a & 0x7FFFFF;

    uint32_t sign2 = (b >> 31) & 1;
    uint32_t exp2 = (b >> 23) & 0xFF;
    uint32_t mant2 = b & 0x7FFFFF;

    // Step 2: Calculate the sign of the result (XOR of the signs)
    uint32_t result_sign = sign1 ^ sign2;

    // Step 3: Add exponents and subtract the bias (127)
    uint32_t result_exp = exp1 + exp2 - 127;

    // Step 4: Multiply the mantissas
    // Add the implicit leading 1 (1.m format in IEEE 754)
    uint64_t result_mant = (uint64_t)(mant1 | 0x800000) * (mant2 | 0x800000);

    // Normalize the result mantissa if necessary
    if (result_mant & 0x800000000000) 
    {
        result_mant >>= 24;      // Shift down to fit 23 bits
        result_exp += 1;         // Adjust exponent
    }
    else 
    {
        result_mant >>= 23;
    }

    // Mask to fit the 23-bit mantissa field
    result_mant &= 0x7FFFFF;

    // Step 5: Assemble the result
    uint32_t result = (result_sign << 31) | (result_exp << 23) | result_mant;

    // Step 6: Interpret the result bits as a float
    return *((float*)&result);
}
void neural_nn2(float *input,float *output)
{
	#pragma HLS INTERFACE s_axilite port=return
	#pragma HLS INTERFACE m_axi depth=12288 port=input offset=slave
	#pragma HLS INTERFACE m_axi depth=1 port=output offset=slave
	float x_input[12288]={};
	memcpy(x_input,(const float*)input,12288*sizeof(float));
//	#pragma HLS ARRAY_PARTITION variable=W1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=W2 complete dim=2
	float layer_1[20]={};
	#pragma HLS ARRAY_PARTITION variable=layer_1 complete dim=1
	float layer_2[1]={};
	#pragma HLS ARRAY_PARTITION variable=layer_2 complete dim=1
	for(int i=0; i<20; i++)
	{
//		#pragma HLS UNROLL factor = 2
//		#pragma HLS PIPELINE
		for(int j=0; j<12288; j++)
		{
			#pragma HLS UNROLL factor = 300
			#pragma HLS PIPELINE
			layer_1[i]+=multiply_floats(x_input[j],W1[i][j]);
			// layer_1[i]+=x_input[j]*W1[i][j];
		}
		layer_1[i]+=b1[i];
		if(layer_1[i]<=0)
			layer_1[i]=0;
//		layer_1[i]=max(0,layer_1[i]);
	}
	for(int i=0; i<1; i++)
	{
		#pragma HLS UNROLL
//		factor = 2
		#pragma HLS PIPELINE
		for(int j=0; j<20; j++)
		{
			#pragma HLS UNROLL
//			factor = 4
			#pragma HLS PIPELINE
			layer_2[i]+=multiply_floats(layer_1[j],W2[i][j]);
			// layer_2[i]+=layer_1[j]*W2[i][j];
		}
		layer_2[i]+=b2[i];
		layer_2[i] = 1.0/(1.0 + hls::exp(-1*layer_2[i]));
//		layer_2[i]=max(0,layer_2[i]);
	}
	if(layer_2[0]>0.5)
	{
		layer_2[0]=1;
	}
	else
	{
		layer_2[0]=0;
	}
	memcpy((float*)output,layer_2,1*sizeof(float));
}
