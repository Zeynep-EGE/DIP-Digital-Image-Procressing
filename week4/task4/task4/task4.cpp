#include <opencv2/opencv.hpp>
#include <iostream>

// CV_32F sonucu mutlak değerle 8 bit'e çevirir
static cv::Mat to8bit(const cv::Mat& m32)
{
    cv::Mat out;
    cv::convertScaleAbs(m32, out);
    return out;
}

int main(int argc, char** argv)
{
    std::string yol = (argc > 1) ? argv[1] : "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png";

    cv::Mat gri = cv::imread(yol, cv::IMREAD_GRAYSCALE);
    if (gri.empty()) {
        std::cerr << "Görüntü okunamadı: " << yol << std::endl;
        return -1;
    }

    cv::Mat yumusak;
    cv::GaussianBlur(gri, yumusak, cv::Size(3, 3), 1.0);

    cv::Mat lap4 = (cv::Mat_<float>(3, 3) <<
        0, 1, 0,
        1, -4, 1,
        0, 1, 0);

    cv::Mat lap8 = (cv::Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    cv::Mat l4_32, l8_32;
    cv::filter2D(yumusak, l4_32, CV_32F, lap4);
    cv::filter2D(yumusak, l8_32, CV_32F, lap8);

    cv::Mat sobelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat sobelY = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat prewX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat prewY = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

    cv::Mat sx, sy, px, py, sobelMag32, prewMag32;
    cv::filter2D(yumusak, sx, CV_32F, sobelX);
    cv::filter2D(yumusak, sy, CV_32F, sobelY);
    cv::filter2D(yumusak, px, CV_32F, prewX);
    cv::filter2D(yumusak, py, CV_32F, prewY);
    cv::magnitude(sx, sy, sobelMag32);
    cv::magnitude(px, py, prewMag32);

    cv::Mat laplacian4 = to8bit(l4_32);
    cv::Mat laplacian8 = to8bit(l8_32);
    cv::Mat sobel = to8bit(sobelMag32);
    cv::Mat prewitt = to8bit(prewMag32);

    cv::imshow("Orijinal (gri)", gri);
    cv::imshow("Laplacian 4 komsulu", laplacian4);
    cv::imshow("Laplacian 8 komsulu", laplacian8);
    cv::imshow("Sobel buyukluk", sobel);
    cv::imshow("Prewitt buyukluk", prewitt);

    cv::imwrite("laplacian4.png", laplacian4);
    cv::imwrite("laplacian8.png", laplacian8);
    cv::imwrite("sobel_buyukluk.png", sobel);
    cv::imwrite("prewitt_buyukluk.png", prewitt);

    cv::waitKey(0);
    return 0;
}