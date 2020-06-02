// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <stdlib.h>
#include "stdafx.h"
#include "common.h"

double dataLuminance[8][8] = {
	{16, 11, 10, 16, 24, 40, 51, 61},
	{12, 12, 14, 19, 26, 58, 60, 55},
	{14, 13, 16, 24, 40, 57, 69, 56},
	{14, 17, 22, 29, 51, 87, 80, 62},
	{18, 22, 37, 56, 68, 109, 103, 77},
	{24, 35, 55, 64, 81, 104, 113, 92},
	{49, 64, 78, 87, 103, 121, 120, 101},
	{72, 92, 95, 98, 112, 100, 103, 99}
};

double dataChrominance[8][8] = {
	{17, 18, 24, 27, 99, 99, 99, 99},
	{18, 21, 26, 66, 99, 99, 99, 99},
	{24, 26, 56, 99, 99, 99, 99, 99},
	{47, 66, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99}
};

char* enc;

#define MAX_TREE_HT 100 

// This constant can be avoided by explicitly 
// calculating height of Huffman Tree 
#define MAX_TREE_HT 100 

// This constant can be avoided by explicitly 
// calculating height of Huffman Tree 
#define MAX_TREE_HT 100 

// A Huffman tree node 
struct MinHeapNode {

	// One of the input characters 
	int data;

	// Frequency of the character 
	unsigned freq;

	// Left and right child of this node 
	struct MinHeapNode *left, *right;
};

// A Min Heap:  Collection of 
// min-heap (or Huffman tree) nodes 
struct MinHeap {

	// Current size of min heap 
	unsigned size;

	// capacity of min heap 
	unsigned capacity;

	// Array of minheap node pointers 
	struct MinHeapNode** array;
};

// A utility function allocate a new 
// min heap node with given character 
// and frequency of the character 
struct MinHeapNode* newNode(int data, unsigned freq)
{
	struct MinHeapNode* temp
		= (struct MinHeapNode*)malloc
		(sizeof(struct MinHeapNode));

	temp->left = temp->right = NULL;
	temp->data = data;
	temp->freq = freq;

	return temp;
}

// A utility function to create 
// a min heap of given capacity 
struct MinHeap* createMinHeap(unsigned capacity)

{

	struct MinHeap* minHeap
		= (struct MinHeap*)malloc(sizeof(struct MinHeap));

	// current size is 0 
	minHeap->size = 0;

	minHeap->capacity = capacity;

	minHeap->array
		= (struct MinHeapNode**)malloc(minHeap->
			capacity * sizeof(struct MinHeapNode*));
	return minHeap;
}

// A utility function to 
// swap two min heap nodes 
void swapMinHeapNode(struct MinHeapNode** a,
	struct MinHeapNode** b)

{

	struct MinHeapNode* t = *a;
	*a = *b;
	*b = t;
}

// The standard minHeapify function. 
void minHeapify(struct MinHeap* minHeap, int idx)

{

	int smallest = idx;
	int left = 2 * idx + 1;
	int right = 2 * idx + 2;

	if (left < minHeap->size && minHeap->array[left]->
		freq < minHeap->array[smallest]->freq)
		smallest = left;

	if (right < minHeap->size && minHeap->array[right]->
		freq < minHeap->array[smallest]->freq)
		smallest = right;

	if (smallest != idx) {
		swapMinHeapNode(&minHeap->array[smallest],
			&minHeap->array[idx]);
		minHeapify(minHeap, smallest);
	}
}

// A utility function to check 
// if size of heap is 1 or not 
int isSizeOne(struct MinHeap* minHeap)
{

	return (minHeap->size == 1);
}

// A standard function to extract 
// minimum value node from heap 
struct MinHeapNode* extractMin(struct MinHeap* minHeap)

{

	struct MinHeapNode* temp = minHeap->array[0];
	minHeap->array[0]
		= minHeap->array[minHeap->size - 1];

	--minHeap->size;
	minHeapify(minHeap, 0);

	return temp;
}

// A utility function to insert 
// a new node to Min Heap 
void insertMinHeap(struct MinHeap* minHeap,
	struct MinHeapNode* minHeapNode)

{

	++minHeap->size;
	int i = minHeap->size - 1;

	while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {

		minHeap->array[i] = minHeap->array[(i - 1) / 2];
		i = (i - 1) / 2;
	}

	minHeap->array[i] = minHeapNode;
}

// A standard function to build min heap 
void buildMinHeap(struct MinHeap* minHeap)

{

	int n = minHeap->size - 1;
	int i;

	for (i = (n - 1) / 2; i >= 0; --i)
		minHeapify(minHeap, i);
}

// A utility function to print an array of size n 
void printArr(int arr[], int n)
{
	int i;
	for (i = 0; i < n; ++i)
		printf("%d", arr[i]);

	printf("\n");
}

// Utility function to check if this node is leaf 
int isLeaf(struct MinHeapNode* root)

{

	return !(root->left) && !(root->right);
}

// Creates a min heap of capacity 
// equal to size and inserts all character of 
// data[] in min heap. Initially size of 
// min heap is equal to capacity 
struct MinHeap* createAndBuildMinHeap(int data[], int freq[], int size)

{

	struct MinHeap* minHeap = createMinHeap(size);

	for (int i = 0; i < size; ++i)
		minHeap->array[i] = newNode(data[i], freq[i]);

	minHeap->size = size;
	buildMinHeap(minHeap);

	return minHeap;
}

// The main function that builds Huffman tree 
struct MinHeapNode* buildHuffmanTree(int data[], int freq[], int size)

{
	struct MinHeapNode *left, *right, *top;

	// Step 1: Create a min heap of capacity 
	// equal to size.  Initially, there are 
	// modes equal to size. 
	struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);

	// Iterate while size of heap doesn't become 1 
	while (!isSizeOne(minHeap)) {

		// Step 2: Extract the two minimum 
		// freq items from min heap 
		left = extractMin(minHeap);
		right = extractMin(minHeap);

		// Step 3:  Create a new internal 
		// node with frequency equal to the 
		// sum of the two nodes frequencies. 
		// Make the two extracted node as 
		// left and right children of this new node. 
		// Add this node to the min heap 
		// '$' is a special value for internal nodes, not used 
		top = newNode('$', left->freq + right->freq);

		top->left = left;
		top->right = right;

		insertMinHeap(minHeap, top);
	}

	// Step 4: The remaining node is the 
	// root node and the tree is complete. 
	return extractMin(minHeap);
}

// Prints huffman codes from the root of Huffman Tree. 
// It uses arr[] to store codes 
void printCodes(struct MinHeapNode* root, int arr[], int top, int data, char out[], int* res)

{

	// Assign 0 to left edge and recur 
	if (root->left) {

		arr[top] = 0;
		printCodes(root->left, arr, top + 1, data, out, res);
	}

	// Assign 1 to right edge and recur 
	if (root->right) {

		arr[top] = 1;
		printCodes(root->right, arr, top + 1, data, out, res);
	}

	// If this is a leaf node, then 
	// it contains one of the input 
	// characters, print the character 
	// and its code from arr[] 
	if (isLeaf(root) && root ->data == data) {

		//printf("%d: ", root->data);
		//printArr(arr, top);
		int i;
		for (i = 0; i < top; ++i)
			out[i] = arr[i];
		out[top] = '\0';
		*res = top;
	}
}

// The main function that builds a 
// Huffman Tree and print codes by traversing 
// the built Huffman Tree 
MinHeapNode* HuffmanCodes(int data[], int freq[], int size)

{
	// Construct Huffman Tree 
	struct MinHeapNode* root
		= buildHuffmanTree(data, freq, size);

	// Print Huffman codes using 
	// the Huffman tree built above 
	int arr[MAX_TREE_HT], top = 0;

	//printCodes(root, arr, top);

	return root;
}

char fname[MAX_PATH];

Mat_<Vec3b> openImage() {
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src;
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		return src;
	}
}

long GetFileSize(std::string filename)
{
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

Mat_<Vec3b> getYCrCb() {

	Mat_<Vec3b> src = openImage();
	Mat_<Vec3b> dst(src.rows, src.cols);

	cvtColor(src, dst, COLOR_BGR2YCrCb);

	Mat_<uchar> dstY(dst.rows, dst.cols);
	Mat_<uchar> dstCr(dst.rows, dst.cols);
	Mat_<uchar> dstCb(dst.rows, dst.cols);

	for(int i = 0; i < dst.rows; i ++)
		for (int j = 0; j < dst.cols; j++)
		{
			dstY(i, j) = dst(i, j)[0];
			dstCr(i, j) = dst(i, j)[1];
			dstCb(i, j) = dst(i, j)[2];
		}

	//imshow("CONVERSION Y", dstY);
	//imshow("CONVERSION CR", dstCr);
	//imshow("CONVERSION CB", dstCb);
	//waitKey(0);

	return dst;
}

Mat_<float> getZigZagTraversal(Mat_<float> src) {
	Mat_<float> traversal(1, src.rows * src.cols);
	int dimension = src.rows;
	int lastValue = dimension * dimension - 1;
	int currNum = 0;
	int currDiag = 0;
	int loopFrom;
	int loopTo;
	int i;
	int row;
	int col;
	do
	{
		if (currDiag < dimension) 
		{
			loopFrom = 0;
			loopTo = currDiag;
		}
		else
		{
			loopFrom = currDiag - dimension + 1;
			loopTo = dimension - 1;
		}

		for (i = loopFrom; i <= loopTo; i++)
		{
			if (currDiag % 2 == 0) 
			{
				row = loopTo - i + loopFrom;
				col = i;
			}
			else
			{
				row = i;
				col = loopTo - i + loopFrom;
			}

			traversal(0, row * 8 + col) = src(row, col);
		}

		currDiag++;
	} while (currDiag <= lastValue);

	return traversal;
}

Mat_<uchar> getReverseZigZagTraversal(Mat_<uchar> traversal) {
	Mat_<uchar> src(8, 8);
	int dimension = src.rows;
	int lastValue = dimension * dimension - 1;
	int currNum = 0;
	int currDiag = 0;
	int loopFrom;
	int loopTo;
	int i;
	int row;
	int col;
	do
	{
		if (currDiag < dimension) 
		{
			loopFrom = 0;
			loopTo = currDiag;
		}
		else 
		{
			loopFrom = currDiag - dimension + 1;
			loopTo = dimension - 1;
		}

		for (i = loopFrom; i <= loopTo; i++)
		{
			if (currDiag % 2 == 0) 
			{
				row = loopTo - i + loopFrom;
				col = i;
			}
			else
			{
				row = i;
				col = loopTo - i + loopFrom;
			}

			src(row, col) = traversal(0, row * 8 + col);
		}

		currDiag++;
	} while (currDiag <= lastValue);

	return src;
}

Mat_<int> getFrequencyArray(Mat_<uchar> src) {
	
	Mat_<int> frequencyArray(1, 256);

	for (int i = 0; i < 256; i++)
		frequencyArray(0, i) = 0;

	for (int i = 0; i < src.cols; i++) {
		int index = src(0, i);
		frequencyArray(0, index)++;
	}

	return frequencyArray;

}

void decode_file2(char s[], int len, char codes[][1000], int size, int symbols[], int result[], int lengths[])
{
	char ans[1000] = "";
	int index = 0;
	int resIndex = 0;

	for (int i = 0; i < len; i++)
	{
		ans[index++] = s[i];
		
		for (int j = 0; j < size; j++) {
			int valid = 1;
			for (int it = 0; it < lengths[j]; it++) {
				if (ans[it] != codes[j][it] || lengths[j] != index
					) {
					valid = 0;
				}
			}
			if (valid == 1) {
				result[resIndex++] = symbols[j];
				index = 0;
				strcpy(ans, "");
				break;
			}
		}

	}

}

void decompress_file(int h, int w) {

	FILE* fptr;
	Mat_<Vec3b> img(h, w);
	fptr = fopen("compressed", "rb");
	int blockNr = 0;
	int readBytes = 0;
	int imI = 0;
	int imJ = 0;
	int channel = 0;
	int maxI = 0, maxJ = 0;

	while (!feof(fptr)) {
		blockNr++;
		char* blockSizeChar = (char*)malloc(sizeof(char));
		char* nrSymbolsChar = (char*)malloc(sizeof(char));
		readBytes = fread(blockSizeChar, 1, 1, fptr);

		if (readBytes == 0)
			break;

		readBytes = fread(nrSymbolsChar, 1, 1, fptr);

		if (readBytes == 0)
			break;

		sprintf(blockSizeChar, "%d", *blockSizeChar);
		sprintf(nrSymbolsChar, "%d", *nrSymbolsChar);
		int blockSize = atoi(blockSizeChar);
		if (blockSize < 0)
			blockSize += 256;
		int nrSymbols = atoi(nrSymbolsChar);
		if (nrSymbols < 0)
			nrSymbols += 256;

		int symbols[64];
		int codes[64];
		int lengths[64];
		char result[1000];

		if (blockSize == 0)
			blockSize++;
		
		for (int i = 0; i < nrSymbols; i++) {
			char* newSymbolChar = (char*)malloc(sizeof(char));
			char* newCodeChar = (char*)malloc(sizeof(char));
			char* newLengthChar = (char*)malloc(sizeof(char));
			fread(newSymbolChar, 1, 1, fptr);
			fread(newCodeChar, 1, 1, fptr);
			fread(newLengthChar, 1, 1, fptr);
			sprintf(newCodeChar, "%d", *newCodeChar);
			sprintf(newSymbolChar, "%d", *newSymbolChar);
			sprintf(newLengthChar, "%d", *newLengthChar);
			symbols[i] = atoi(newSymbolChar);
			if (symbols[i] < 0)
				symbols[i] += 256;
			codes[i] = atoi(newCodeChar);
			lengths[i] = atoi(newLengthChar);
		}

		
		int len = 0;
		if (blockSize % 8 == 0)
			len = blockSize / 8;
		else len = ceil((float) blockSize / 8.0f) ;

		unsigned char mybyte = 0;
		int byteIndex = 1;
		int charIndex = 1;
		int currLength = 0;
		int resultIndex = 0;
		int checkLength = 0;
		
		for (int i = 0; i < len; i++) {
			char* dataChar = (char*)malloc(sizeof(char));
			fread(dataChar, 1, 1, fptr);
			sprintf(dataChar, "%d", *dataChar);
			int data = atoi(dataChar);
			int data2 = data;

			while (charIndex <= 8) {

				if (data2 & 128) {
					if (byteIndex == 0)
						mybyte |= 256;
					else if (byteIndex == 1)
						mybyte |= 128;
					else if (byteIndex == 2)
						mybyte |= 64;
					else if (byteIndex == 3)
						mybyte |= 32;
					else if (byteIndex == 4)
						mybyte |= 16;
					else if (byteIndex == 5)
						mybyte |= 8;
					else if (byteIndex == 6)
						mybyte |= 4;
					else if (byteIndex == 7)
						mybyte |= 2;

				}

				int found = 0;

				for (int j = 0; j < nrSymbols; j++) {

					int currSymbol = symbols[j];
					int currCode = codes[j];

					if (byteIndex != lengths[j])
						continue;

					int codeIndex = 0;
					int valid = 1;
					int data3 = mybyte;

					while (codeIndex < byteIndex) {

						if ((data3 & 128) == 128) {
							if ((currCode & 128) != 128) {
								valid = 0;
							}
						}

						if ((data3 & 128) == 0) {
							if ((currCode & 128) != 0) {
								valid = 0;
							}
						}
						
						if ((currCode & 128) == 0) {
							if ((data3 & 128) != 0) {
								valid = 0;
							}
						}


						if ((currCode & 128) == 128) {
							if ((data3 & 128) != 128) {
								valid = 0;
							}
						}

						data3 = data3 << 1;
						currCode = currCode << 1;

						codeIndex++;
					}


					if (valid == 1) {
						checkLength += byteIndex;
						result[resultIndex++] = currSymbol;
						byteIndex = 0;
						mybyte = 0;
						found = 1;
						break;
					}

					if (found == 1)
						break;

				}

				byteIndex++;
				data2 = data2 << 1;
				charIndex++;
				if (checkLength == blockSize)
					break;
			}

			charIndex = 1;

		}

		while (resultIndex < 64)
			result[resultIndex++] = 128;

		Mat_<double> block(8, 8);
		Mat luminance = Mat(8, 8, CV_64FC1, &dataLuminance);
		Mat chrominance = Mat(8, 8, CV_64FC1, &dataChrominance);

		Mat_<uchar>decodedMat(1, 64);
		Mat_<uchar>decodedBlock(8, 8);

		for (int it = 0; it < 64; it++)
			decodedMat(0, it) = result[it];

		decodedBlock = getReverseZigZagTraversal(decodedMat);

		block = decodedBlock;

		block -= 128.0;

		if (channel == 0) {
			multiply(block, luminance, block);
		}
		else {
			multiply(block, chrominance, block);
		}

		idct(block, block);

		block += 128.0;

		

		for (int bi = 0; bi < 8 && bi + imI < img.rows; bi++) {
			for (int bj = 0; bj < 8 && bj + imJ < img.cols; bj++) {
				img(imI + bi, imJ + bj)[channel] = block(bi, bj);
				if (imJ + bj > maxJ)
					maxJ = imJ + bj;
			}
			if (imI + bi > maxI)
				maxI = imI + bi;
		}

		channel++;

		if (channel > 2) {

			channel = 0;
			imJ += 8;

			if (imJ > w) {
				imJ = 0;
				imI += 8;
			}

		}
		
	}

	cvtColor(img, img, COLOR_YCrCb2BGR);

	imshow("decompressed", img);
	waitKey(0);
}

void compressionAlgorithm(Mat_<Vec3b> img) {

	Mat_<Vec3b> dst(img.rows, img.cols);
	Mat luminance = Mat(8, 8, CV_64FC1, &dataLuminance);
	Mat chrominance = Mat(8, 8, CV_64FC1, &dataChrominance);
	FILE *fptr = fopen("compressed", "wb");
	long totalLen = 0;
	long originalLen = 0;
	int blockNr = 0;
	int fileSize = 0;
	int symbolSize = 0;
	int codeSize = 0;

	for (int i = 0; i <= img.rows; i += 8)
		for (int j = 0; j <= img.cols; j += 8) {
			for (int channel = 0; channel < 3; channel++) {
				blockNr++;
				Mat_<double> block(8, 8);
				originalLen += 64;
				int bi, bj;

				for (bi = 0; bi < 8 && bi + i < img.rows; bi++)
					for (bj = 0; bj < 8 && bj + j < img.cols; bj++) {
						block(bi, bj) = img(i + bi, j + bj)[channel];
						
					}

				block -= 128.0;

				dct(block, block);

				if (channel == 0) {
					divide(block, luminance, block);
				}
				else {
					divide(block, chrominance, block);
				}

				Mat_<uchar> blockU(8, 8);

				block += 128.0; 	
				block.convertTo(blockU, CV_8UC1);

				Mat_<uchar> zigZag = getZigZagTraversal(blockU);

				int data[64];
				int freq[255];
				int symbols[64];
				int index = 0;

				for (int it = 0; it < zigZag.cols; it++)
				{
					freq[it] = 0;
					symbols[it] = 0;
				}

				int zeroIndex = 63;

				for (int it = 63; it >= 0; it--)
				{
					if (zigZag(0, it) != 128) {
						break;
					}
					zeroIndex--;
				}				

				if (zeroIndex < 0)
					zeroIndex = 1;

				Mat_<int> frequencyArray = getFrequencyArray(zigZag);

				for (int it = 0; it < zigZag.cols; it++)
					data[it] = zigZag(0, it);

				for (int it = 0; it < zigZag.cols; it++) {
					int good = 1;
					for (int it2 = 0; it2 < zigZag.cols; it2++)
						if (data[it] == symbols[it2])
							good = 0;
					if (good == 1) {
						symbols[index] = data[it];
						freq[index++] = frequencyArray(0, data[it]);
					}
				}

				int size = sizeof(data) / sizeof(data[0]);

				MinHeapNode *root = HuffmanCodes(symbols, freq, index);

				char out[64][1000];
				int arr[MAX_TREE_HT];
				int lengths[64];

				for (int it = 0; it < index; it++) {
					int top = 0;
					printCodes(root, arr, 0, symbols[it], out[it], &top);
					lengths[it] = top;
				}

				char result[64 * 1000];
				int finalLength = 0;
				int byteIndex = 0;
				unsigned char mybyte = 0;

				for (int it = 0; it <= zeroIndex; it++) {
					int symbolIndex = 0;
					for (int it2 = 0; it2 < index; it2++)
						if (symbols[it2] == zigZag(0, it)) {
							symbolIndex = it2;
							break;
						}
					finalLength += lengths[symbolIndex];
				}


					fileSize += fwrite(&finalLength, 1, 1, fptr);
					fileSize += fwrite(&index, 1, 1, fptr);

				for (int it = 0; it < index; it++) {
					int printNext = 0;
					int currentSymbol = symbols[it];

					int inc1 = fwrite(&currentSymbol, 1, 1, fptr);
					symbolSize += inc1;
					fileSize += inc1;
					for (int it2 = 0; it2 < lengths[it]; it2++) {
						result[finalLength + it2] = out[it][it2];
						byteIndex++;

						if (out[it][it2] == 1) {
							if(byteIndex == 1)
								mybyte |= 128;
							else if (byteIndex == 2)
								mybyte |= 64;
							else if (byteIndex == 3)
								mybyte |= 32;
							else if (byteIndex == 4)
								mybyte |= 16;
							else if (byteIndex == 5)
								mybyte |= 8;
							else if (byteIndex == 6)
								mybyte |= 4;
							else if (byteIndex == 7)
								mybyte |= 2;
							else if (byteIndex == 8)
								mybyte |= 1;

						}
					}

					int inc = fwrite(&mybyte, 1, 1, fptr);
					codeSize += inc;
					fileSize += inc;
					int currentLength = lengths[it];
					fileSize += fwrite(&currentLength, 1, 1, fptr);
					byteIndex = 0;
					mybyte = 0;

				}

				finalLength = 0; int printNext = 0;
				byteIndex = 0;

				for (int it = 0; it <= zeroIndex; it++) {
					int symbolIndex = 0;
					

					for (int it2 = 0; it2 < index; it2++)
						if (symbols[it2] == zigZag(0, it)) {
							symbolIndex = it2;
							break;
						}
					for (int it2 = 0; it2 < lengths[symbolIndex]; it2++) {
						result[finalLength + it2] = out[symbolIndex][it2];
						if (byteIndex <= 7) {
							byteIndex++;
							mybyte = mybyte << 1;
							if (out[symbolIndex][it2] == 1) {
								mybyte |= 1;
							}
						}else
						if (byteIndex == 8) {
							if (printNext == 1) {
								printNext = 0;
							}
							if (mybyte == 106) {
								printNext = 1;
							}
							fileSize += fwrite(&mybyte, 1, 1, fptr);
							byteIndex = 1;
							mybyte = 0;
							mybyte = mybyte << 1;
							if (out[symbolIndex][it2] == 1) {
								mybyte |= 1;

							}
							totalLen += 1;
						}
					}

					finalLength += lengths[symbolIndex];
				}

				if (totalLen < (ceil(finalLength / 8.0f)) || finalLength == 0)
				{
					mybyte = mybyte << (8 - byteIndex);
					fileSize += fwrite(&mybyte, 1, 1, fptr);
					byteIndex = 0;
				}

				totalLen = 0;

				int decoded[1000];
				int decodedlength = 0;

				decode_file2(result, finalLength,out,index,symbols,decoded,lengths);

				Mat_<uchar>decodedMat(1, 64);
				Mat_<uchar>decodedBlock(8, 8);

				for (int it = 0; it < 64; it++)
					decodedMat(0, it) = decoded[it];

				decodedBlock = getReverseZigZagTraversal(decodedMat);

				block = decodedBlock;

				block -= 128.0;
				
				if (channel == 0) {
					multiply(block, luminance, block);
				}
				else {
					multiply(block, chrominance, block);
				}

				idct(block, block);

				block += 128.0;

				for (int bi = 0; bi < 8 && bi + i < img.rows; bi++)
					for (int bj = 0; bj < 8 && bj + j < img.cols; bj++) {
						dst(i + bi, j + bj)[channel] = block(bi, bj);
					
					}
				

			}
		}


	fclose(fptr);
	decompress_file(dst.rows - (dst.rows % 8), dst.cols - (dst.cols % 8));
	cvtColor(dst, dst, COLOR_YCrCb2BGR);
	printf("COMPRESSION RATIO : %lf", (double) fileSize / (double) GetFileSize(fname));
}

int main()
{
	compressionAlgorithm(getYCrCb());
	return 0;
}