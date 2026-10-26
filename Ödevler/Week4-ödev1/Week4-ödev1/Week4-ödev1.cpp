#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

//bu ödevi yapay zeka yardımı ile yazdım

using namespace cv;
using namespace std;

static inline int clampIdx(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

//1. Adım Gürültü bastırma - 5x5 Gauss filtrelemesi
//----------------------------------------------------
Mat gaussianBlur5x5(const Mat& src, double sigma = 1.4) {
    const int K = 5, R = K / 2;

    // 5x5 Gauss çekirdeğini oluştur ve toplamı 1 olacak şekilde normalize et
    float kernel[K][K];
    float sum = 0.0f;
    for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
            float v = exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel[i + R][j + R] = v;
            sum += v;
        }
    }
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            kernel[i][j] /= sum;

    // Konvolüsyon
    Mat dst(src.size(), CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float acc = 0.0f;
            for (int i = -R; i <= R; i++) {
                for (int j = -R; j <= R; j++) {
                    int yy = clampIdx(y + i, 0, src.rows - 1);
                    int xx = clampIdx(x + j, 0, src.cols - 1);
                    acc += kernel[i + R][j + R] * src.at<float>(yy, xx);
                }
            }
            dst.at<float>(y, x) = acc;
        }
    }
    return dst;
}

//2. Adım olarka  Sobel filtresi ile gradyan hesaplama
//------------------------------------------------------
void sobelGradient(const Mat& src, Mat& magnitude, Mat& direction) {
    const int Kx[3][3] = { {-1, 0, 1},
                           {-2, 0, 2},
                           {-1, 0, 1} };
    const int Ky[3][3] = { {-1, -2, -1},
                           { 0,  0,  0},
                           { 1,  2,  1} };

    magnitude = Mat(src.size(), CV_32F);
    direction = Mat(src.size(), CV_32F);

    float maxMag = 0.0f;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int yy = clampIdx(y + i, 0, src.rows - 1);
                    int xx = clampIdx(x + j, 0, src.cols - 1);
                    float p = src.at<float>(yy, xx);
                    gx += Kx[i + 1][j + 1] * p;
                    gy += Ky[i + 1][j + 1] * p;
                }
            }
            float mag = sqrt(gx * gx + gy * gy);
            magnitude.at<float>(y, x) = mag;
            maxMag = max(maxMag, mag);

            float angle = atan2(gy, gx) * 180.0f / (float)CV_PI;
            if (angle < 0) angle += 180.0f;
            direction.at<float>(y, x) = angle;
        }
    }

    if (maxMag > 0)
        magnitude *= (255.0f / maxMag);
}

//3. Adım olarak Non-maximum suppression (NMS)
//-----------------------------------------------
Mat nonMaximumSuppression(const Mat& magnitude, const Mat& direction) {
    Mat nms = Mat::zeros(magnitude.size(), CV_32F);

    for (int y = 1; y < magnitude.rows - 1; y++) {
        for (int x = 1; x < magnitude.cols - 1; x++) {
            float angle = direction.at<float>(y, x);
            float m = magnitude.at<float>(y, x);
            float n1, n2;

            // Açıyı 4 yöne indir: 0, 45, 90, 135
            if (angle < 22.5f || angle >= 157.5f) {          
                n1 = magnitude.at<float>(y, x - 1);
                n2 = magnitude.at<float>(y, x + 1);
            }
            else if (angle < 67.5f) {                      
                n1 = magnitude.at<float>(y - 1, x - 1);
                n2 = magnitude.at<float>(y + 1, x + 1);
            }
            else if (angle < 112.5f) {                     
                n1 = magnitude.at<float>(y - 1, x);
                n2 = magnitude.at<float>(y + 1, x);
            }
            else {                                        
                n1 = magnitude.at<float>(y - 1, x + 1);
                n2 = magnitude.at<float>(y + 1, x - 1);
            }

            // Yerel maksimumsa koru değilse 0 olarak bırak
            if (m >= n1 && m >= n2)
                nms.at<float>(y, x) = m;
        }
    }
    return nms;
}

// 4. Adım olarka Hysteresis thresholding
// ---------------------------------------------------------------
Mat hysteresisThreshold(const Mat& nms, float lowT, float highT) {
    Mat edges = Mat::zeros(nms.size(), CV_8U);
    vector<Point> stack;

    // Güçlü kenarları bul ve başlangıç noktaları olarak yığına ekle
    for (int y = 0; y < nms.rows; y++) {
        for (int x = 0; x < nms.cols; x++) {
            if (nms.at<float>(y, x) >= highT) {
                edges.at<uchar>(y, x) = 255;
                stack.push_back(Point(x, y));
            }
        }
    }

    // Güçlü kenarlardan yayılarak zayıf kenarları bağla
    while (!stack.empty()) {
        Point p = stack.back();
        stack.pop_back();

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = p.x + dx, ny = p.y + dy;
                if (nx < 0 || ny < 0 || nx >= nms.cols || ny >= nms.rows) continue;

                if (edges.at<uchar>(ny, nx) == 0 && nms.at<float>(ny, nx) >= lowT) {
                    edges.at<uchar>(ny, nx) = 255;
                    stack.push_back(Point(nx, ny));
                }
            }
        }
    }
    return edges;
}

//Tüm adımları sırayla çalıştıran ana Canny fonksiyonu
//--------------------------------------------------------
Mat myCanny(const Mat& gray, float lowT, float highT) {
    Mat f;
    gray.convertTo(f, CV_32F);

    Mat blurred = gaussianBlur5x5(f);

    Mat magnitude, direction;
    sobelGradient(blurred, magnitude, direction);

    Mat nms = nonMaximumSuppression(magnitude, direction);

    return hysteresisThreshold(nms, lowT, highT);
}

int main(int argc, char** argv) {
    string path = (argc > 1) ? argv[1]
        : "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png";

    Mat gray = imread(path, IMREAD_GRAYSCALE);
    if (gray.empty()) {
        cerr << "Resim okunamadi: " << path << endl;
        return -1;
    }

    float lowT = (argc > 2) ? atof(argv[2]) : 20.0f;   // 0-255 ölçeğinde
    float highT = (argc > 3) ? atof(argv[3]) : 50.0f;

    Mat result = myCanny(gray, lowT, highT);

    // Karşılaştırma için OpenCV'nin kendi Canny'si sadece kontrol etmek amaçlı kullanalım
    Mat ref;
    Canny(gray, ref, 50, 150);

    imshow("Orijinal", gray);
    imshow("Benim Canny", result);
    imshow("OpenCV Canny (karsilastirma)", ref);
    imwrite("canny_sonuc.png", result);

    waitKey(0);
    return 0;
}