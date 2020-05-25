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



void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat_<Vec3b> openImage() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src;
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		return src;
	}
}

Mat_<Vec3b> step1() {

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
		if (currDiag < dimension) // if doing the upper-left triangular half
		{
			loopFrom = 0;
			loopTo = currDiag;
		}
		else // doing the bottom-right triangular half
		{
			loopFrom = currDiag - dimension + 1;
			loopTo = dimension - 1;
		}

		for (i = loopFrom; i <= loopTo; i++)
		{
			if (currDiag % 2 == 0) // want to fill upwards
			{
				row = loopTo - i + loopFrom;
				col = i;
			}
			else // want to fill downwards
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
		if (currDiag < dimension) // if doing the upper-left triangular half
		{
			loopFrom = 0;
			loopTo = currDiag;
		}
		else // doing the bottom-right triangular half
		{
			loopFrom = currDiag - dimension + 1;
			loopTo = dimension - 1;
		}

		for (i = loopFrom; i <= loopTo; i++)
		{
			if (currDiag % 2 == 0) // want to fill upwards
			{
				row = loopTo - i + loopFrom;
				col = i;
			}
			else // want to fill downwards
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

void decode_file(struct MinHeapNode* root, char s[], int len)
{
	char ans[1000] = "";

	struct MinHeapNode* curr = root;
	for (int i = 0; i < len; i++)
	{
		//printf("%d", s[i]);
		if (s[i] == '0')
			curr = curr->left;
		else
			curr = curr->right;

		// reached leaf node 
		if (curr->left == NULL && curr->right == NULL)
		{
			//ans += curr->data;
			//printf("%d ", curr->data);
			curr = root;
		}
	}
	// cout<<ans<<endl; 
	//return ans + '\0';
}

void decode_file2(char s[], int len, char codes[][1000], int size, int symbols[], int result[], int lengths[])
{
	char ans[1000] = "";
	int index = 0;
	int resIndex = 0;
	//printf("!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	//printf("!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	//printf("Size: %d\n", size);

	for (int i = 0; i < len; i++)
	{
		//printf("%d", s[i]);
		ans[index++] = s[i];
		//printf("%d", s[i]);
		
		for (int j = 0; j < size; j++) {
			int valid = 1;
			//printf("\nCHECKING FOR LENGTH: %d ", lengths[j]);
			//printf("Code: ");
			for (int it = 0; it < lengths[j]; it++) {
				//printf("COMPARING %d AGAINST %d ", ans[it], codes[j][it]);
				//printf("%X", codes[j][it]);
				if (ans[it] != codes[j][it] || lengths[j] != index
					) {
					valid = 0;
					//printf("FAILED AT COMPARING %d AGAINST %d ", ans[it], codes[j][it]);
				}
			}
			//printf(" Size: %d\n", strlen(codes[j]));
			if (valid == 1) {
				result[resIndex++] = symbols[j];
				//printf(" DECODED TO %d\n", symbols[j]);
				index = 0;
				strcpy(ans, "");
				break;
			}
		}

	}
	// cout<<ans<<endl; 
	//return ans + '\0';
}

void decompress_file(int h, int w) {

	FILE* fptr;
	Mat_<Vec3b> img(h, w);
	fptr = fopen("letters3", "rb");
	int blockNr = 0;
	int readBytes = 0;
	int imI = 0;
	int imJ = 0;
	int channel = 0;

	while (!feof(fptr)) {
		blockNr++;
		//printf("READING BLOCK: %d\n", blockNr);
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
		//printf("\nSIZE : %d SYMBOLS: %d\n", blockSize, nrSymbols);

		int symbols[64];
		int codes[64];
		int lengths[64];
		char result[1000];
		
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
			//printf("%d %d %d\n", symbols[i], codes[i], lengths[i]);
		}

		
		int len = 0;
		if (blockSize % 8 == 0)
			len = blockSize / 8;
		else len = ceil((float) blockSize / 8.0f);

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

				//printf("MYBYTE = %d\n", mybyte);

				int found = 0;

				for (int j = 0; j < nrSymbols; j++) {
					//printf("FOR BYTEINDEX = %d\n", byteIndex);
					int currSymbol = symbols[j];
					int currCode = codes[j];

					if (byteIndex != lengths[j])
						continue;

					//printf("TEST: %d CODES: %d\n", mybyte, currCode);

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

						if (valid == 1) {
							//printf("currCode = %d data3 = %d\n", currCode & 128, data3 & 128);
						}

						data3 = data3 << 1;
						currCode = currCode << 1;

						codeIndex++;
					}


					if (valid == 1) {
						//printf("FOUND! %d IN BYTE %d POSITION %d WITH BYTEINDEX %d\n", currSymbol, i, charIndex, byteIndex);
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

			//printf("%d ", data);
		}

		/*
		//printf("\nLEN: %d\nRESULT: \n", checkLength);
		for (int k = 0; k < resultIndex; k++)
			if (result[k] > 0)
				//printf("%d ", result[k]);
			else printf("%d ", result[k] + 256);
			*/
		Mat_<double> block(8, 8);
		Mat luminance = Mat(8, 8, CV_64FC1, &dataLuminance);
		Mat chrominance = Mat(8, 8, CV_64FC1, &dataChrominance);

		Mat_<uchar>decodedMat(1, 64);
		Mat_<uchar>decodedBlock(8, 8);

		for (int it = 0; it < 64; it++)
			decodedMat(0, it) = result[it];

		//printf("\nDECODED: \n");

		decodedBlock = getReverseZigZagTraversal(decodedMat);




		//std::cout << "\n\n\n\n";
		block = decodedBlock;


		for (int it = 0; it < 8; it++) {
			for (int it2 = 0; it2 < 8; it2++) {}
			//	printf("%d ", decodedBlock(it, it2));
			//printf("\n");
		}


		//blockU.convertTo(block, CV_64FC1);

		block -= 128.0;

		if (channel == 0) {
			multiply(block, luminance, block);
		}
		else {
			multiply(block, chrominance, block);
		}

		idct(block, block);

		block += 128.0;

		for (int bi = 0; bi < 8 && bi + imI < img.rows; bi++)
			for (int bj = 0; bj < 8 && bj + imJ < img.cols; bj++) {
				img(imI + bi, imJ + bj)[channel] = block(bi, bj);

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
		
		//printf("\n\n\n\n");
		
	}

	cvtColor(img, img, COLOR_YCrCb2BGR);
	imshow("decompressed", img);
	waitKey(0);
}

void step2(Mat_<Vec3b> img) {

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

				for (int bi = 0; bi < 8 && bi + i < img.rows; bi++)
					for (int bj = 0; bj < 8 && bj + j < img.cols; bj++) {
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
			//	printf("ORIGINAL: \n");
				
				

				block.convertTo(blockU, CV_8UC1);

				for (int it = 0; it < 8; it++) {
					for (int it2 = 0; it2 < 8; it2++){}
					//	printf("%d ", blockU(it, it2));
				//	printf("\n");
				}
				
				Mat_<uchar> zigZag = getZigZagTraversal(blockU);

				//printf("\n ZIGZAG: \n");

				//for (int it = 0; it < 64; it++)
					//printf("%d ", zigZag(0, it));


				int data[64];
				int freq[255];
				int symbols[64];
				int index = 0;

				for (int it = 0; it < zigZag.cols; it++)
				{
					freq[it] = 0;
					symbols[it] = 0;
				}

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
				
				//printf("\nBLOCK %d\n", blockNr);
				
				/*
				for (int it = 0; it < index; it++) {
				printf("%d - ", symbols[it]);
					for (int it2 = 0; it2 < lengths[it]; it2++) 
						printf("%d", out[it][it2]);
					printf("\n");
				}

								printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
								printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

								printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
*/

				char result[64 * 1000];
				int finalLength = 0;
				int byteIndex = 0;
				unsigned char mybyte = 0;
				//fwrite(&mybyte, 1, 1, fptr);
			//	printf("\nCODED STRING:\n");

				for (int it = 0; it < 64; it++) {
					int symbolIndex = 0;
					for (int it2 = 0; it2 < index; it2++)
						if (symbols[it2] == zigZag(0, it)) {
							symbolIndex = it2;
							break;
						}
					//	printf(" ");
					finalLength += lengths[symbolIndex];
				}
				//printf("LENGTH = %d INDEX = %d\n", finalLength, index);
				/*
				if (finalLength == 106 && index == 13) {
					for (int it = 0; it < index; it++) {
						printf("%d - ", symbols[it]);
						for (int it2 = 0; it2 < lengths[it]; it2++)
							printf("%d", out[it][it2]);
						printf("\n");
					}
				}
				*/
					fileSize += fwrite(&finalLength, 1, 1, fptr);
					fileSize += fwrite(&index, 1, 1, fptr);

				//if (index == 13)
					//printf("FOUND INDEX AT %d\n", finalLength);

				for (int it = 0; it < index; it++) {
					int printNext = 0;
					int currentSymbol = symbols[it];
					//if (currentSymbol == 106)
						//printf("FOUND AT 2 %d\n", symbols[it+1]);
					symbolSize += fwrite(&currentSymbol, 1, 1, fptr);
					for (int it2 = 0; it2 < lengths[it]; it2++) {
						result[finalLength + it2] = out[it][it2];
						byteIndex++;
						//mybyte = mybyte >> 1;
						//printf("%X ", out[symbolIndex][it2]);
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

							//printf("ONE! %X\n", mybyte);
						}
					}

					codeSize += fwrite(&mybyte, 1, 1, fptr);
					int currentLength = lengths[it];
					fwrite(&currentLength, 1, 1, fptr);
					byteIndex = 0;
					mybyte = 0;

				}

				finalLength = 0; int printNext = 0;
				byteIndex = 0;

				for (int it = 0; it < 64; it++) {
					int symbolIndex = 0;
					

					for (int it2 = 0; it2 < index; it2++)
						if (symbols[it2] == zigZag(0, it)) {
							symbolIndex = it2;
							break;
						}
					//printf("%d ", symbols[symbolIndex]);
					for (int it2 = 0; it2 < lengths[symbolIndex]; it2++) {
						result[finalLength + it2] = out[symbolIndex][it2];
						if (byteIndex <= 7) {
							byteIndex++;
							mybyte = mybyte << 1;
							//printf("%X ", out[symbolIndex][it2]);
							if (out[symbolIndex][it2] == 1) {
								mybyte |= 1;
								//printf("ONE! %X\n", mybyte);
							}
						}else
						if (byteIndex == 8) {
							//printf("%X ", mybyte);
							if (printNext == 1) {
								printNext = 0;
								//printf("%d\n", mybyte);
							}
							if (mybyte == 106) {
								printNext = 1;
								//printf("FOUND AT 0\n!");
							}
							fwrite(&mybyte, 1, 1, fptr);
							byteIndex = 1;
							mybyte = 0;
							mybyte = mybyte << 1;
							//printf("%d ", out[symbolIndex][it2]);
							if (out[symbolIndex][it2] == 1) {
								mybyte |= 1;
								//printf("ONE! %X\n", mybyte);
							}
							totalLen += 1;
						}
					}
				//	printf(" ");
					finalLength += lengths[symbolIndex];
				}
				
				if (byteIndex != 0 && mybyte != 0) {
					mybyte = mybyte << (8 - byteIndex);
					fwrite(&mybyte, 1, 1, fptr);
					byteIndex = 0;
				}
				
				//printf("\nENCODED: ");
				//for (int i = 0; i < finalLength; i++)
				//	printf("%X ", result[i]);

				int decoded[1000];
				int decodedlength = 0;
				//void decode_file2(char s[], int len, char codes[][1000], int size, int symbols[], int result[], int lengths[])

				decode_file2(result, finalLength,out,index,symbols,decoded,lengths);

				//printf("\nDECODED ZIGZAG:\n");
				//for (int it = 0; it < 64; it++)
					//printf("%d ", decoded[it]);

				Mat_<uchar>decodedMat(1, 64);
				Mat_<uchar>decodedBlock(8, 8);

				for (int it = 0; it < 64; it++)
					decodedMat(0, it) = decoded[it];

				//printf("\nDECODED: \n");

				decodedBlock = getReverseZigZagTraversal(decodedMat);

				


				//std::cout << "\n\n\n\n";
				block = decodedBlock;

				
				for (int it = 0; it < 8; it++) {
					for (int it2 = 0; it2 < 8; it2++){}
					//	printf("%d ", decodedBlock(it, it2));
					//printf("\n");
				}
				

				//blockU.convertTo(block, CV_64FC1);

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
	decompress_file(dst.rows, dst.cols);
	cvtColor(dst, dst, COLOR_YCrCb2BGR);
	//imshow("blocks", dst);
	//waitKey(0);
	printf("%ld", totalLen);
	printf("\n%ld", originalLen);
}

int main()
{
	step2(step1());
	return 0;
}