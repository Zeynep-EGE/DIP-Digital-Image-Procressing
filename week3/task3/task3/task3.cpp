#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


cv::Mat claheElle(const cv::Mat& src, double clipLimit = 2.0,
    int tilesX = 8, int tilesY = 8)
{
    CV_Assert(src.type() == CV_8UC1);

    int rows = src.rows, cols = src.cols;
    double tw = cols / (double)tilesX;   
    double th = rows / (double)tilesY;   

    std::vector<std::vector<uchar>> lut(tilesX * tilesY, std::vector<uchar>(256));

    for (int ty = 0; ty < tilesY; ty++) {
        for (int tx = 0; tx < tilesX; tx++) {
            int x0 = (int)std::lround(tx * tw), x1 = (int)std::lround((tx + 1) * tw);
            int y0 = (int)std::lround(ty * th), y1 = (int)std::lround((ty + 1) * th);

            std::vector<int> hist(256, 0);
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                    hist[src.at<uchar>(y, x)]++;

            int area = (x1 - x0) * (y1 - y0);

            int clip = std::max(1, (int)(clipLimit * area / 256.0));

            int excess = 0;
            for (int i = 0; i < 256; i++) {
                if (hist[i] > clip) {
                    excess += hist[i] - clip;
                    hist[i] = clip;
                }
            }

            int batch = excess / 256;
            int residual = excess - batch * 256;
            for (int i = 0; i < 256; i++) hist[i] += batch;
            if (residual > 0) {
                int step = std::max(256 / residual, 1);
                for (int i = 0; i < 256 && residual > 0; i += step, residual--)
                    hist[i]++;
            }

            int sum = 0;
            std::vector<uchar>& L = lut[ty * tilesX + tx];
            for (int i = 0; i < 256; i++) {
                sum += hist[i];
                double v = sum * 255.0 / area;
                L[i] = (uchar)std::min(255.0, v + 0.5);
            }
        }
    }
    std::vector<int> cx1(cols), cx2(cols);
    std::vector<double> ax(cols);
    for (int x = 0; x < cols; x++) {
        double g = (x + 0.5) / tw - 0.5;      
        int t1 = (int)std::floor(g);
        ax[x] = g - t1;
        cx1[x] = std::max(t1, 0);
        cx2[x] = std::min(t1 + 1, tilesX - 1);
    }

    cv::Mat dst(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; y++) {
        double g = (y + 0.5) / th - 0.5;
        int t1 = (int)std::floor(g);
        double ay = g - t1;
        int ry1 = std::max(t1, 0);
        int ry2 = std::min(t1 + 1, tilesY - 1);

        for (int x = 0; x < cols; x++) {
            int v = src.at<uchar>(y, x);

            double a = lut[ry1 * tilesX + cx1[x]][v];  
            double b = lut[ry1 * tilesX + cx2[x]][v];  
            double c = lut[ry2 * tilesX + cx1[x]][v];  
            double d = lut[ry2 * tilesX + cx2[x]][v];  

            double ust = (1 - ax[x]) * a + ax[x] * b;
            double alt = (1 - ax[x]) * c + ax[x] * d;
            double sonuc = (1 - ay) * ust + ay * alt;

            dst.at<uchar>(y, x) = (uchar)(sonuc + 0.5);
        }
    }
    return dst;
}

// ---------------------------------------------------------
int main()
{
    cv::Mat img = cv::imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Resim okunamadi!" << std::endl;
        return -1;
    }

    int tilesX = 8, tilesY = 8;
    img = img(cv::Rect(0, 0, img.cols / tilesX * tilesX, img.rows / tilesY * tilesY)).clone();

    double clipLimit = 2.0;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tilesX, tilesY));
    cv::Mat hazir;
    clahe->apply(img, hazir);

    cv::Mat elle = claheElle(img, clipLimit, tilesX, tilesY);

    long long toplamFark = 0;
    int maxFark = 0;
    long long farkliPiksel = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int fark = std::abs((int)hazir.at<uchar>(i, j) - (int)elle.at<uchar>(i, j));
            toplamFark += fark;
            if (fark > maxFark) maxFark = fark;
            if (fark != 0) farkliPiksel++;
        }
    }
    double ortFark = (double)toplamFark / (img.rows * img.cols);

    std::cout << "Ortalama mutlak fark : " << ortFark << std::endl;
    std::cout << "Maksimum fark        : " << maxFark << std::endl;
    std::cout << "Farkli piksel orani  : "
        << 100.0 * farkliPiksel / (img.rows * img.cols) << " %" << std::endl;

    cv::imwrite("clahe_hazir.png", hazir);
    cv::imwrite("clahe_elle.png", elle);

    cv::imshow("Orijinal", img);
    cv::imshow("CLAHE - Hazir", hazir);
    cv::imshow("CLAHE - Elle", elle);
    cv::waitKey(0);
    return 0;
}