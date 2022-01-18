#include "simple_neural_networks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


double single_in_single_out_nn(double  input, double weight) {
	// TODO: Return the result of multiplication of input and its weight.
   	return input*weight;
}


double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN){
	double output = 0;
	int i = 0;
	for(i = 0; i < INPUT_LEN; i++){
		output += input[i] * weight[i]; //summing input*weight
	}
	// TODO: Use for loop to multiply all inputs with their weights
 	return output;
}


double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN){
	double predicted_value = 0;
	
	predicted_value = weighted_sum(input, weight, INPUT_LEN);
	return predicted_value;
}

/*
	for (i=0; i < 5; i++) {
		double training_eg1[3] = {temperature[i],humidity[i], air_quality[i]};

		printf("Prediction from training example %d is : %f\n ", i+1,
				multiple_inputs_single_output_nn(training_eg1, weight, NUM_OF_INPUTS));
	}
}

#define NUM_OF_INPUTS 	3

double temperature[5] = {12,23,50,-10,16};
double humidity[5] =    {60,67,45,65,63};
double air_quality[5] = {60,47,157,187,94};
double weight[3] = 		{-2,2,1};	// Weights for each input

so input every iteration is:

{12,60,60}, {-2,2,1}, 3 (num inputs)

*/


void elementwise_multiple( double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN) {
	// TODO: Use for loop to calculate output_vector
	int i = 0;
	for(i = 0; i < VECTOR_LEN; i++){
		output_vector[i] = input_scalar * weight_vector[i];

	}
}


void single_input_multiple_output_nn(double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN){
  elementwise_multiple(input_scalar, weight_vector,output_vector,VECTOR_LEN);
}

/*
single_input_multiple_output_nn(SAD, weights, predicted_results, OUT_LEN);

double predicted_results[3];
double weights[3] = {-20.2, 95, 201.0};	// Weights for each input: temp, hum, air_quality

#define SAD   0.0		// TODO: Find appropriate sad value

#define TEMPERATURE_PREDICTION_IDX  0
#define HUMIDITY_PREDICTION_IDX 	1
#define AIR_QUALITY_PREDICTION_IDX  2

#define OUT_LEN     3
*/


void matrix_vector_multiplication(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {

	// TODO: Use two for loops to calculate output vector based on the input vector and weights matrix
	//matrix weights 3x3 , input vector 1x3 , output vector 1x3
	int i = 0, j = 0;
	for (i = 0; i < OUTPUT_LEN; i++){
		for (j = 0; j < INPUT_LEN; j++){
			output_vector[i] += input_vector[j] * weights_matrix[i][j];
		}
	}
}

/*
#define OUT_LEN     3
#define IN_LEN		3

double predicted_output[OUT_LEN];
								   //temp, hum,  air_q
double weights[OUT_LEN][IN_LEN] ={  {-2.0, 9.5, 2.0 },        // sad?
									{-0.8, 7.2, 6.3 },        // sick?
									{-0.5, 0.4, 0.9 }  };     // active ?

double input_vector[IN_LEN] = {30.0, 87.0, 110.0};	// temp, hum, air_q input values	

i.e.: output = 30*-2 + 87 * 9.5 + 2 * 110
*/


void multiple_inputs_multiple_outputs_nn(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	matrix_vector_multiplication(input_vector,INPUT_LEN,output_vector,OUTPUT_LEN,weights_matrix);
}


void hidden_nn( double *input_vector, uint32_t INPUT_LEN,
				uint32_t HIDDEN_LEN, double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
				uint32_t OUTPUT_LEN, double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN], double *output_vector) {
	/* TODO: Use matrix_vector_multiplication to calculate values for hidden_layer. Make sure that when you initialize
	   hidden_pred_vector variable then zero its value with for loop */
	   
	double hidden_layer[] = {0,0,0}; //matrix_vector_multiplication returns 1xINPUT_LEN

	matrix_vector_multiplication(input_vector, INPUT_LEN, hidden_layer, HIDDEN_LEN, in_to_hid_weights); //input -> hidden layer

	// TODO: Use matrix_vector_multiplication to calculate output layer values from hidden layer
	matrix_vector_multiplication(hidden_layer, HIDDEN_LEN, output_vector, OUTPUT_LEN, hid_to_out_weights); //hidden layer -> output

	

}

/*
double predicted_output[OUT_LEN];
								   	   	   	   	   //temp, hum,  air_q
double input_to_hidden_weights[HID_LEN][IN_LEN] ={  {-2.0, 9.5, 2.0},   	//hid[0]
													{-0.8, 7.2, 6.3},      //hid[1]
													{-0.5, 0.4, 0.9}};     //hid[2]

												   //hid[0] hid[1] hid[2]
double hidden_to_output_weights[OUT_LEN][HID_LEN] ={{-1.0,  1.15,  0.11},   //sad?
													{-0.18, 0.15, -0.01},   //sick?
													{0.25, -0.25, -0.1 }};  //active?

double input_vector[IN_LEN] = {30.0, 87.0, 110.0};	// temp, hum, air_q input values

hidden_nn(input_vector,IN_LEN,HID_LEN,input_to_hidden_weights,OUT_LEN,hidden_to_output_weights,predicted_output);
*/


// Calculate the error using yhat (predicted value) and y (expected value)
double find_error(double yhat, double y) {
	// TODO: Use math.h functions to calculate the error with double precision
	return pow(y-yhat, 2.0);
}





void brute_force_learning( double input, double weight, double expected_value, double step_amount, uint32_t itr) {
   double prediction,error;
   double up_prediction, down_prediction, up_error, down_error;
   int i;
   int counter = 0;
	 for(i=0, counter = 0; (i < itr && counter < 10); i++){

		 prediction  = input * weight;
		 // TODO: Calculate the error
		 error = find_error(prediction, expected_value);

		 if(error <= 1e-8){
			counter += 1;			
		 }

		 printf("Step: %d   Error: %f    Prediction: %f    Weight: %f\n", i, error, prediction, weight);

		 up_prediction =  input * (weight + step_amount);
		 up_error      =   powf((up_prediction - expected_value),2);

		 // TODO: Calculate down_prediction and down_error on the same way as up_prediction and up_error
		 down_prediction =  input * (weight - step_amount);
		 down_error      =  powf((down_prediction - expected_value),2);

		 if(down_error <  up_error)
			 // TODO: Change weight value accordingly if down_error is smaller than up_error
			   weight -= step_amount;
		 if(down_error >  up_error)
			 // TODO: Change weight value accordingly if down_error is larger than up_error
			   weight += step_amount;
	 }
}



void linear_forward_nn(double *input_vector, uint32_t INPUT_LEN,
						double *output_vector, uint32_t OUTPUT_LEN,
						double weights_matrix[OUTPUT_LEN][INPUT_LEN], double *weights_b) {

	matrix_vector_multiplication(input_vector,INPUT_LEN, output_vector,OUTPUT_LEN,weights_matrix);
	int k;
	
	for(k=0;k<OUTPUT_LEN;k++){
		output_vector[k]+=weights_b[k];
	}
	//linear_forward_nn(train_x, NUM_OF_FEATURES, z1, NUM_OF_HID1_NODES, w1, b1);
}


double relu(double x){
	// TODO: Calculate ReLu based on its mathematical formula
	if(x > 0.0)
	{
		return x;
	}
		
	return 0.1*x;
}


void vector_relu(double *input_vector, double *output_vector, uint32_t LEN) {
	  int i;
	  for(i =0;i<LEN;i++){
		  output_vector[i] =  relu(input_vector[i]);
		}
}


double sigmoid(double x) {
	// TODO: Calculate sigmoid based on its mathematical formula
	 double result =  0;
	 result = exp(x)/(1 + exp(x));
	 return result;
}


void vector_sigmoid(double * input_vector, double * output_vector, uint32_t LEN) {
	int i;
	for (i = 0; i < LEN; i++) {
		output_vector[i] = sigmoid(input_vector[i]);
	}
}


double compute_cost(uint32_t m, double yhat[m][1], double y[m][1]) {
	double cost = 0;
	// TODO: Calculate cost based on mathematical cost function formula
	int i;
	double margin = 0.000001;
	//printf("cost %i: %d\n", i, cost);
	for(i=0; i<m; i++)
	{

		cost += y[i][0] * log(yhat[i][0] + margin) + (1.0 - y[i][0])*log(1.0 - (yhat[i][0] + margin));
		printf("cost %i: %f\n", i, y[i][0] * log(yhat[i][0] + margin) + (1.0 - y[i][0])*log(1.0 - (yhat[i][0] + margin)));

		// printf("testy y[i][0] * log(yhat[i][0] + margin)): %d\n", y[i][0] * log(yhat[i][0] + margin));
		// printf("testy (1.0 - y[i][0])*log(1.0 - (yhat[i][0] + margin)): %d\n", (1.0 - y[i][0])*log(1.0 - (yhat[i][0] + margin)));
		//printf("Test first one: %f\n\n", (1*(log(1.0-0.688184+0.001))));

		// formula:
		// https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11
	}
	return -cost/m;
}


void normalize_data_2d(uint32_t ROW, uint32_t COL, double input_matrix[COL][ROW], double output_matrix[COL][ROW]){
	int i;
	double max, min;
	for(i =0;i<ROW;i++){
		max = -99999999;
		min = 99999999;
		int j;
		for(j =0;j<COL;j++){
			if(input_matrix[j][i] >max){
				max = input_matrix[j][i];
				}
			
			if(input_matrix[j][i] < min){
				min = input_matrix[j][i];
				}
		}
		for(j =0;j<COL;j++){
			output_matrix[j][i] =  (input_matrix[j][i] - min)/(max-min);
		}
	}

	    	



	// double max =  -99999999;
	// double max_per_column[] = {max, max, max};
	// int i;
	// for(i =0;i<ROW;i++){
	// 	int j;
	// 	for(j =0;j<COL;j++){
	// 		if(input_matrix[i][j] > max_per_column[j]){
	// 			max_per_column[j] = input_matrix[i][j];
	// 			}
	// 		}
	// }

	// for(i=0;i<ROW;i++){
	// 	int j;
	// 	for(j=0;j<COL;j++){
	//     	output_matrix[i][j] =  input_matrix[i][j]/max_per_column[j];
	// 	}
	// }
}


// Use this function to print matrix values for debugging
void matrix_print(uint32_t ROW, uint32_t COL, double A[ROW][COL]) {
	int i;
	for(i=0; i<ROW; i++){
		int j;
		for(j=0; j<COL; j++){
			printf(" %f ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n\r");
}


void weights_random_initialization(uint32_t HIDDEN_LEN, uint32_t INPUT_LEN, double weight_matrix[HIDDEN_LEN][INPUT_LEN]) {
	double d_rand;

	/*Seed random number generator*/
	srand(1);
	int i;
	for (i = 0; i < HIDDEN_LEN; i++) {
		int j;
		for (j = 0; j < INPUT_LEN; j++) {
			/*Generate random numbers between 0 and 1*/
			d_rand = (rand() % 10);
			d_rand /= 10;
			weight_matrix[i][j] = d_rand;
		}
	}
}


void weightsB_zero_initialization(double * weightsB, uint32_t LEN){
	memset(weightsB, 0, LEN*sizeof(weightsB[0]));
}


void relu_backward(uint32_t m, uint32_t LAYER_LEN, double dA[m][LAYER_LEN], 
		double Z[m][LAYER_LEN], double dZ[m][LAYER_LEN]) {
	//TODO: implement derivative of relu function  You can can choose either to calculate for all example at the same time
	//or make iteratively. Check formula for derivative lecture 5 on slide 24
	// relu' (x) = {0 for x < 0; 1 for x >= 0}
	// relu' = dA
	int i,j;
	double dRelu_x;
	for(i=0; i<m; i++){
		for(j=0; j<LAYER_LEN; j++)
		{
			if(Z[i][j] <= 0.0){
				dRelu_x = 0.1;
			}
			else{
				dRelu_x = 1.0;
			}
			dZ[i][j] = dA[i][j]*dRelu_x;

		}
	}

}


void linear_backward(uint32_t LAYER_LEN, uint32_t PREV_LAYER_LEN, uint32_t m, double dZ[m][LAYER_LEN],
		double A_prev[m][PREV_LAYER_LEN], double dW[LAYER_LEN][PREV_LAYER_LEN], double * db ){
	// TODO: implement linear backward. You can can choose either to calculate for all example at the same time (dw= 1/m *A_prev[T]*dZ;)
	//or make iteratively  (dw_iter= dZ*A_prev[T];)
	// 1 3 1
	//	 linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, NUM_OF_SAMPLES, dZ3, a2, dW3, db3);
	
	int i,j;

	
	for (i = 0 ; i < m ; i++){
		for (j = 0 ; j < LAYER_LEN ; j++){
			
			db[j] += dZ[i][j]/m;
		}

		//db[i] = sum_dZ;
	}
	
	double dZ_t[LAYER_LEN][m];
	matrix_transpose(m, LAYER_LEN, dZ, dZ_t);

	// printf("dZt_t size: %d x %d\n", LAYER_LEN, m);
	// printf("dZ_t\n");
	// matrix_print(LAYER_LEN, m, dZ_t);

	// printf("A_prev size: %d x %d\n", m, PREV_LAYER_LEN);
	// printf("A_prev\n");
	// matrix_print(m, PREV_LAYER_LEN, A_prev);

	double matmul_dZ_t_Aprev[LAYER_LEN][PREV_LAYER_LEN];
	matrix_matrix_multiplication(LAYER_LEN, m, PREV_LAYER_LEN, dZ_t, A_prev, matmul_dZ_t_Aprev);

	// printf("matmul_dZ_t_Aprev size: %d x %d\n", LAYER_LEN, PREV_LAYER_LEN);
	// printf("matmul_dZ_t_Aprev\n");
	// matrix_print(LAYER_LEN, PREV_LAYER_LEN, matmul_dZ_t_Aprev);

	matrix_divide_scalar(LAYER_LEN,PREV_LAYER_LEN, 40.0, matmul_dZ_t_Aprev, dW);
	// printf("dW size: %d x %d with m division %d\n", LAYER_LEN, PREV_LAYER_LEN, 1.0);
	// printf("matmul_dZ_t_Aprev\n");
	// matrix_print(LAYER_LEN, PREV_LAYER_LEN, dW);
	//CHANGED ORDER OF MULTIPLICATIONS BECAUSE dZ*A_t is just not possible!!
}


void matrix_matrix_sum(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_COL][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	int c, d;
	for (c = 0; c < MATRIX_ROW; c++) {
	      for (d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix1[c][d]+input_matrix2[c][d];
	      }
	 }
}


void matrix_divide_scalar(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double scalar,
									double input_matrix[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	int c,d;
	for (c = 0; c < MATRIX_ROW; c++) {
	      for (d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix[c][d]/scalar;
	      }
	 }
}


void matrix_matrix_multiplication(uint32_t MATRIX1_ROW, uint32_t MATRIX1_COL, uint32_t MATRIX2_COL,
									double input_matrix1[MATRIX1_ROW][MATRIX1_COL],
									double input_matrix2[MATRIX1_COL][MATRIX2_COL],
									double output_matrix[MATRIX1_ROW][MATRIX2_COL]) {

	int k,c,d;
	for(k=0;k<MATRIX1_ROW;k++){
		 memset(output_matrix[k], 0, MATRIX2_COL*sizeof(output_matrix[0][0]));
	}
	double sum=0;
	for (c = 0; c < MATRIX1_ROW; c++) {
	      for (d = 0; d < MATRIX2_COL; d++) {
	        for (k = 0; k < MATRIX1_COL; k++) {
	          sum += input_matrix1[c][k]*input_matrix2[k][d];
	        }
	        output_matrix[c][d] = sum;
	        sum = 0;
	      }
	 }
}


void matrix_matrix_sub(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	int c,d;
	for (c = 0; c < MATRIX_ROW; c++) {
	      for (d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix1[c][d]-input_matrix2[c][d];
	      }
	 }
}


void weights_update(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double learning_rate,
									double dW[MATRIX_ROW][MATRIX_COL],
									double W[MATRIX_ROW][MATRIX_COL]) {
	//TODO: implement weights_update function
	int i, j;
	for(i = 0; i < MATRIX_ROW; i++) {
		for(j = 0; j < MATRIX_COL; j++){
			W[i][j] = W[i][j] - learning_rate*dW[i][j];
		}
	}
	// new weight = new weight - learning_rate*(dW)
}


void matrix_multiply_scalar(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double scalar,
									double input_matrix[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	int c,d;									
	for (c = 0; c < MATRIX_ROW; c++) {
	      for (d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix[c][d]*scalar;
	      }
	 }
}


void matrix_transpose(uint32_t ROW, uint32_t COL, double A[ROW][COL], double A_T[COL][ROW]) {
	int i,j;
	for(i=0; i<ROW; i++){
		for(j=0; j<COL; j++){
			A_T[j][i]=A[i][j];
		}
	}
}


