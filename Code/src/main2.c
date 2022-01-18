/*
 ============================================================================
 Name        : Homework 1 - ML for ES
 Author      : Omar El Nahhas
 Version     : 1
 Copyright   : 
 Description : Homework assignment 1, neural network in C

 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

// Size of the layers
#define NUM_OF_FEATURES   	3  	// input values
#define NUM_OF_HID1_NODES	5
#define NUM_OF_HID2_NODES	4
#define NUM_OF_OUT_NODES	1	// output classes
#define NUM_OF_SAMPLES		40
#define MAX_EPOCH			7

double learning_rate= 0.35;



/*Input layer to hidden layer1*/
double a1[NUM_OF_SAMPLES][NUM_OF_HID1_NODES] = {{0}};		// activation function
double b1[NUM_OF_HID1_NODES] = {0};							// bias
double z1[NUM_OF_SAMPLES][NUM_OF_HID1_NODES] = {{0}};		// hlayer 1 output vector

// Input layer to hidden layer1 weight matrix
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] =    {{-0.25, 0.5,   -0.89},   	 //hid1[0]
													{0.8,  -0.82,  0.3 },     //hid1[1]
													{-0.5,  0.45,  0.49},	 //hid1[2]
													{0.29,  -0.63,  0.44},
													{0.69,  -0.77,  0.21}};   


/*hidden layer 1  to hidden layer 2*/
double a2[NUM_OF_SAMPLES][NUM_OF_HID2_NODES]= {{0}};		// activation function
double b2[NUM_OF_HID2_NODES]= {0};							// bias
double z2[NUM_OF_SAMPLES][NUM_OF_HID2_NODES]= {{0}};		// hlayer 2 output vector

// hidden layer1 to hidden layer 2 weight matrix
double w2[NUM_OF_HID2_NODES][NUM_OF_HID1_NODES] =    {{0.12, -0.21, 0.30, -0.78, 0.29},   	 //hid2[0]
													{-0.97,  0.43,  -0.22, -0.33, 0.59},     //hid2[1]
													{0.76,  -0.32,  0.89, 0.43, -0.88},	 	//hid2[2]
													{0.2,  -0.36,  -0.77, 0.66, -0.27}};   


/*Hidden layer2 to output layer*/
double b3[NUM_OF_OUT_NODES]= {0};
double z3[NUM_OF_SAMPLES][NUM_OF_OUT_NODES]= {{0}};	// Predicted output vector

// Hidden layer 2 to output layer weight matrix
double w3[NUM_OF_OUT_NODES][NUM_OF_HID2_NODES] =    {{0.48, -0.73, 0.23, -0.69}};

// Predicted values
double yhat[NUM_OF_SAMPLES][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

// Training data
double train_x[NUM_OF_SAMPLES][NUM_OF_FEATURES];				// Training data after normalization
double train_y_norm[NUM_OF_SAMPLES][NUM_OF_OUT_NODES];
double train_y[NUM_OF_SAMPLES][NUM_OF_OUT_NODES] = {  	{0},
														{0},
														{1},
														{1},
														{1},
														{1},
														{0},
														{1},
														{0},
														{1},
														{0},
														{0},
														{1},
														{0},
														{1},
														{1},
														{0},
														{0},
														{0},
														{0},
														{1},
														{0},
														{0},
														{1},
														{1},
														{1},
														{0},
														{0},
														{1},
														{1},
														{0},
														{1},
														{0},
														{0},
														{1},
														{0},
														{1},
														{1},
														{0},
														{1}
													};


void main(void) {
	// Raw training data
	double raw_x[NUM_OF_SAMPLES][NUM_OF_FEATURES] = { 	{-10, 51, 252},	
														{60, 75, 341},	
														{20, 25, 86},	
														{28, 9, 61},	
														{19, 21, 78},	
														{14, 44, 144},	
														{57, 48, 212},	
														{10, 21, 23},	
														{-42, 65, 242},	
														{7, 7, 149},	
														{-19, 55, 219},	
														{-48, 91, 190},	
														{25, 44, 23},	
														{53, 53, 279},	
														{0, 5, 64},	
														{-4, 35, 103},	
														{47, 86, 452},	
														{59, 73, 246},	
														{45, 48, 208},	
														{-42, 60, 499},	
														{32, 26, 55},	
														{40, 97, 156},	
														{-7, 89, 494},	
														{6, 6, 82},	
														{35, 43, 48},	
														{20, 23, 34},	
														{-16, 79, 497},	
														{39, 96, 261},	
														{-1, 36, 27},	
														{11, 27, 72},	
														{38, 56, 170},	
														{1, 43, 52},	
														{55, 64, 364},	
														{-13, 94, 153},	
														{27, 1, 114},	
														{-52, 80, 209},	
														{7, 40, 18},	
														{15, 7, 7},	
														{58, 54, 291},	
														{-3, 20, 118}
														};

	normalize_data_2d(NUM_OF_FEATURES, NUM_OF_SAMPLES, raw_x, train_x);	// Data normalization, swapped axis
	printf("train_x (normalized) \n");
	matrix_print(NUM_OF_SAMPLES, NUM_OF_FEATURES, train_x);

	normalize_data_2d(NUM_OF_OUT_NODES, NUM_OF_SAMPLES, train_y, train_y_norm);
	printf("train_y (normalized) \n");
	matrix_print(NUM_OF_SAMPLES, NUM_OF_OUT_NODES, train_y_norm);

	weightsB_zero_initialization(b1, NUM_OF_HID1_NODES);
	weightsB_zero_initialization(b2, NUM_OF_HID2_NODES);
	weightsB_zero_initialization(b3, NUM_OF_OUT_NODES);

    
    // printf("z1 all \n");
    // matrix_print(NUM_OF_SAMPLES, NUM_OF_HID1_NODES, z1);
	
	//backpropagation init
	double dA1[NUM_OF_SAMPLES][NUM_OF_HID1_NODES] = {{0, 0, 0, 0, 0}};
	double dA2[NUM_OF_SAMPLES][NUM_OF_HID2_NODES] = {{0}};
	double dA3[NUM_OF_SAMPLES][NUM_OF_OUT_NODES] = {{0}};

	double dZ1[NUM_OF_SAMPLES][NUM_OF_HID1_NODES] = {{0, 0, 0}};
	double dZ2[NUM_OF_SAMPLES][NUM_OF_HID2_NODES] = {{0}};
	double dZ3[NUM_OF_SAMPLES][NUM_OF_OUT_NODES] = {{0}};


	double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] = {	{0, 0, 0}, 
														{0, 0, 0}, 
														{0, 0, 0},
														{0, 0, 0}, 
														{0, 0, 0}};

	double dW2[NUM_OF_HID2_NODES][NUM_OF_HID1_NODES] = {{0, 0, 0, 0, 0}, 
														{0, 0, 0, 0, 0},  
														{0, 0, 0, 0, 0}, 
														{0, 0, 0, 0, 0}};

	double dW3[NUM_OF_OUT_NODES][NUM_OF_HID2_NODES] = {{0, 0, 0, 0}};

	double db1[NUM_OF_HID1_NODES] = {0, 0, 0, 0, 0};
	double db2[NUM_OF_HID2_NODES] = {0, 0, 0, 0};
	double db3[NUM_OF_OUT_NODES] = {0};



	int epoch, train_set;

	for(epoch = 0; epoch<MAX_EPOCH;epoch++){
		printf("------------------ EPOCH %i ------------------ \n", epoch);
		for(train_set=0; train_set<NUM_OF_SAMPLES; train_set++){
            //forward prop

            linear_forward_nn(train_x[train_set], NUM_OF_FEATURES, z1[train_set], NUM_OF_HID1_NODES, w1, b1);

            // printf("z1 \n");
            // matrix_print(1, NUM_OF_HID1_NODES, z1[train_set]);

            vector_relu(z1[train_set],a1[train_set],NUM_OF_HID1_NODES);
            // printf("relu_a1 \n");
            // matrix_print(1, NUM_OF_HID1_NODES, a1[train_set]);

            //hid 1 to hid 2
            linear_forward_nn(a1[train_set], NUM_OF_HID1_NODES, z2[train_set], NUM_OF_HID2_NODES, w2, b2);

            vector_relu(z2[train_set],a2[train_set],NUM_OF_HID2_NODES);
            // printf("relu_a2 \n");
            // matrix_print(1, NUM_OF_HID2_NODES, a2[train_set]);



            // hid 2 to out
            linear_forward_nn(a2[train_set], NUM_OF_HID2_NODES, z3[train_set], NUM_OF_OUT_NODES, w3, b3);
            // printf("Z3 output \n");
            // matrix_print(1, 1, z3[train_set]);

            /*compute yhat*/
            vector_sigmoid(z3[train_set],yhat[train_set], NUM_OF_OUT_NODES);
            // printf("yhat:  %f\n\r", yhat[0][0]);
            // printf("train_y:  %f\n\r", train_y[0][0]);

            // printf("y_hat \n");
            // matrix_print(1, NUM_OF_OUT_NODES, yhat[train_set]);

        }

        double cost = 0.0;
        cost = compute_cost(NUM_OF_SAMPLES, yhat, train_y_norm);
        printf("cost:  %f\r\n", cost);

		printf("z1:\n");
        matrix_print(NUM_OF_SAMPLES, NUM_OF_HID1_NODES, z1);

		printf("a1:\n");
        matrix_print(NUM_OF_SAMPLES, NUM_OF_HID1_NODES, a1);

		printf("z2:\n");
        matrix_print(NUM_OF_SAMPLES, NUM_OF_HID2_NODES, z2);

		printf("a2:\n");
        matrix_print(NUM_OF_SAMPLES, NUM_OF_HID2_NODES, a2);

		printf("z3:\n");
        matrix_print(NUM_OF_SAMPLES, NUM_OF_OUT_NODES, z3);

        printf("yhat:\n");
        matrix_print(NUM_OF_SAMPLES, 1, yhat);
		
        matrix_matrix_sub(NUM_OF_SAMPLES, NUM_OF_OUT_NODES, yhat, train_y, dZ3);

        // printf("dZ3 \n");
        // matrix_print(NUM_OF_SAMPLES, 1, dZ3);


        //check for formula on slide 31 (lecture 5)
        linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, NUM_OF_SAMPLES, dZ3, a2, dW3, db3);


        printf("dW3 \n");
        matrix_print(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, dW3);

        printf("db3 \n");
        matrix_print(NUM_OF_OUT_NODES, 1, db3);

        // double W3_T[NUM_OF_HID2_NODES][NUM_OF_OUT_NODES] = {{0},
        //                                                     {0},
        //                                                     {0},
        //                                                     {0}};

        // matrix_transpose(NUM_OF_OUT_NODES,NUM_OF_HID2_NODES,w3, W3_T);

        // printf("W3_T \n");
        // matrix_print(NUM_OF_HID2_NODES, NUM_OF_OUT_NODES, W3_T);

		//// NOT USING TRANSPOSE BECAUSE OF DIMENSION INCOMPATIBILITY
        matrix_matrix_multiplication(NUM_OF_SAMPLES, 1, NUM_OF_HID2_NODES, dZ3, w3, dA2); //       OG: matrix_matrix_multiplication(NUM_OF_HID2_NODES, NUM_OF_OUT_NODES, 1, W3_T, dZ3, dA2);
        // printf("dA2 \n");
        // matrix_print(NUM_OF_SAMPLES, NUM_OF_HID2_NODES, dA2);

    /// update 2nd layer

        relu_backward(NUM_OF_SAMPLES, NUM_OF_HID2_NODES, dA2, z2, dZ2);
        // printf("dZ2 \n");
        // matrix_print(NUM_OF_SAMPLES, NUM_OF_HID2_NODES, dZ2);

        linear_backward(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, NUM_OF_SAMPLES, dZ2, a1, dW2, db2);


        printf("dW2 \n");
        matrix_print(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, dW2);

        printf("db2 \n");
        matrix_print(NUM_OF_HID2_NODES, 1, db2);

        // double W2_T[NUM_OF_HID1_NODES][NUM_OF_HID2_NODES] = {	{0,0,0,0},
        //                                                         {0,0,0,0},
        //                                                         {0,0,0,0},
        //                                                         {0,0,0,0},
        //                                                         {0,0,0,0}};


        // matrix_transpose(NUM_OF_HID2_NODES,NUM_OF_HID1_NODES,w2, W2_T);

        // printf("W2_T \n");
        // matrix_print(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, W2_T);


        matrix_matrix_multiplication(NUM_OF_SAMPLES, NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, dZ2, w2, dA1); // OG: matrix_matrix_multiplication(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, 1, W2_T, dZ2, dA1);
        // printf("dA1 \n");
        // matrix_print(NUM_OF_SAMPLES, NUM_OF_HID1_NODES, dA1);

        /* Input layer */

        relu_backward(NUM_OF_SAMPLES, NUM_OF_HID1_NODES, dA1, z1, dZ1);
        // printf("dZ1 \n");
        // matrix_print(NUM_OF_SAMPLES, NUM_OF_HID1_NODES, dZ1);

        linear_backward(NUM_OF_HID1_NODES, NUM_OF_FEATURES, NUM_OF_SAMPLES, dZ1, train_x, dW1, db1); // OG: linear_backward(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, NUM_OF_SAMPLES, dZ1, train_x, dW1, db1); 
        printf("dW1 \n");
        matrix_print(NUM_OF_HID1_NODES, NUM_OF_FEATURES, dW1);

        printf("db1  \n");
        matrix_print(NUM_OF_HID1_NODES, 1, db1);
        /*UPDATE PARAMETERS*/



        weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, dW1, w1);

        printf("updated W1  \n");
        matrix_print( NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1);



        weights_update(NUM_OF_HID1_NODES, 1, learning_rate, db1, b1);
        printf("updated b1  \n");
        matrix_print(NUM_OF_HID1_NODES, 1, b1);



        weights_update(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, learning_rate, dW2, w2);

        printf("updated W2  \n");
        matrix_print( NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, w2);



        weights_update(NUM_OF_HID2_NODES, 1, learning_rate, db2, b2);
        printf("updated b2  \n");
        matrix_print( NUM_OF_HID2_NODES, 1, b2);


        weights_update(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, learning_rate, dW3, w3);

        printf("updated W3  \n");
        matrix_print( NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, w3);




        weights_update(NUM_OF_OUT_NODES, 1, learning_rate, db3, b3);
        printf("updated b3  \n");
        matrix_print( NUM_OF_OUT_NODES, 1, b3);

    }

	/*PREDICT*/
	printf("-------- PREDICT --------\n");
	double input_x_eg[1][NUM_OF_FEATURES] = {{-60, 70, 420}};
	double input_x[1][NUM_OF_FEATURES] = {{0}};

	//normalize_data_2d(3,1, input_x_eg, input_x);
	printf("New input: {-60, 70, 420}\n");
	printf("Normalized:\n");
	normalize_data_2d(1, NUM_OF_FEATURES, input_x_eg, input_x);
	matrix_print(1, 3, input_x);
	/*compute z1*/
	linear_forward_nn(input_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);

	/*compute a1*/
	vector_relu(z1[0],a1,NUM_OF_HID1_NODES);

	/*compute z2*/
	linear_forward_nn(a1[0], NUM_OF_HID1_NODES, z2[0], NUM_OF_HID2_NODES, w2, b2);

	/*compute a2*/
	vector_relu(z2[0],a2[0],NUM_OF_HID1_NODES);

	/*compute z3*/
	linear_forward_nn(a2, NUM_OF_HID2_NODES, z3[0], NUM_OF_OUT_NODES, w3, b3);
	printf("z3_eg1:  %f \n",z3[0][0]);

	/*compute yhat*/
	vector_sigmoid(z3[0],yhat_eg, NUM_OF_OUT_NODES);
	printf("predicted:  %f\n\r", yhat_eg[0]);

	if(yhat_eg[0] >=0.5){
		printf("Satisfactory! :)\n\n");
	}
	else{
		printf("Non satisfactory! :(\n\n");
	}


	int numba;
	int correct = 0;
	for(numba = 0; numba < NUM_OF_SAMPLES; numba++){
		linear_forward_nn(train_x[numba], NUM_OF_FEATURES, z1[numba], NUM_OF_HID1_NODES, w1, b1);

		// printf("z1 \n");
		// matrix_print(1, NUM_OF_HID1_NODES, z1[train_set]);

		vector_relu(z1[numba],a1[numba],NUM_OF_HID1_NODES);
		// printf("relu_a1 \n");
		// matrix_print(1, NUM_OF_HID1_NODES, a1[train_set]);

		//hid 1 to hid 2
		linear_forward_nn(a1[numba], NUM_OF_HID1_NODES, z2[numba], NUM_OF_HID2_NODES, w2, b2);

		vector_relu(z2[numba],a2[numba],NUM_OF_HID2_NODES);
		// printf("relu_a2 \n");
		// matrix_print(1, NUM_OF_HID2_NODES, a2[train_set]);



		// hid 2 to out
		linear_forward_nn(a2[numba], NUM_OF_HID2_NODES, z3[numba], NUM_OF_OUT_NODES, w3, b3);
		// printf("Z3 output \n");
		// matrix_print(1, 1, z3[train_set]);

		/*compute yhat*/
		vector_sigmoid(z3[numba],yhat[numba], NUM_OF_OUT_NODES);
		// printf("yhat:  %f\n\r", yhat[0][0]);
		// printf("train_y:  %f\n\r", train_y[0][0]);
		if(yhat[numba][0] >= 0.5 && train_y[numba][0] == 1){
			correct++;
		}
		else if(yhat[numba][0] < 0.5 && train_y[numba][0] == 0){
			correct++;
		}
		//printf("y_hat \n");
		// matrix_print(1, NUM_OF_OUT_NODES, yhat[train_set]);
	}
	printf("\nTrain - Correct: %i/40\n", correct);




	int numba_v;
	int correct_v = 0;
	double valid_x[10][NUM_OF_FEATURES] = {{-55, 99, 499}, //0	
									{55, 78, 350}, //0	
									{-65, 89, 333}, //0
									{-40, 66, 290},	//0
									{57, 65, 212},//0	
									{1, 2, 3},	//1
									{10, 25, 86}, //1
									{19, 21, 78},	//1
									{-14, 44, 77}, //1	
									{5, 20, 99}};	//1

	//double valid_x_norm[10][NUM_OF_FEATURES];					
	//normalize_data_2d(10, NUM_OF_FEATURES, valid_x, valid_x_norm);
	for(numba_v = 0; numba_v < 10; numba_v++){
		linear_forward_nn(valid_x[numba_v], NUM_OF_FEATURES, z1[numba_v], NUM_OF_HID1_NODES, w1, b1);

		// printf("z1 \n");
		// matrix_print(1, NUM_OF_HID1_NODES, z1[train_set]);

		vector_relu(z1[numba_v],a1[numba_v],NUM_OF_HID1_NODES);
		// printf("relu_a1 \n");
		// matrix_print(1, NUM_OF_HID1_NODES, a1[train_set]);

		//hid 1 to hid 2
		linear_forward_nn(a1[numba_v], NUM_OF_HID1_NODES, z2[numba_v], NUM_OF_HID2_NODES, w2, b2);

		vector_relu(z2[numba_v],a2[numba_v],NUM_OF_HID2_NODES);
		// printf("relu_a2 \n");
		// matrix_print(1, NUM_OF_HID2_NODES, a2[train_set]);



		// hid 2 to out
		linear_forward_nn(a2[numba_v], NUM_OF_HID2_NODES, z3[numba_v], NUM_OF_OUT_NODES, w3, b3);
		// printf("Z3 output \n");
		// matrix_print(1, 1, z3[train_set]);

		/*compute yhat*/
		vector_sigmoid(z3[numba_v],yhat[numba_v], NUM_OF_OUT_NODES);
		// printf("yhat:  %f\n\r", yhat[0][0]);
		// printf("train_y:  %f\n\r", train_y[0][0]);
		if(yhat[numba_v][0] >= 0.5 && numba_v > 4){
			correct_v++;
		}
		else if(yhat[numba_v][0] < 0.5 && numba_v <= 4){
			correct_v++;
		}
		//printf("y_hat \n");
		// matrix_print(1, NUM_OF_OUT_NODES, yhat[train_set]);
	}
	printf("Validation - Correct: %i/10\n\n", correct_v);
	
}
