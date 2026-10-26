#include <opencv2/opencv.hpp>
#include <iostream>

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

    cv::Mat sobelX = (cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

    cv::Mat sobelY = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1);

    cv::Mat gx32, gy32;
    cv::filter2D(yumusak, gx32, CV_32F, sobelX);
    cv::filter2D(yumusak, gy32, CV_32F, sobelY);

    cv::Mat gx, gy;
    cv::convertScaleAbs(gx32, gx);
    cv::convertScaleAbs(gy32, gy);

    cv::Mat buyukluk32, buyukluk;
    cv::magnitude(gx32, gy32, buyukluk32);
    cv::convertScaleAbs(buyukluk32, buyukluk);

    cv::imshow("Orijinal (gri)", gri);
    cv::imshow("Gauss sonrasi", yumusak);
    cv::imshow("Sobel X (yatay gradyan)", gx);
    cv::imshow("Sobel Y (dikey gradyan)", gy);
    cv::imshow("Sobel buyukluk", buyukluk);

    cv::imwrite("sobel_x.png", gx);
    cv::imwrite("sobel_y.png", gy);
    cv::imwrite("sobel_buyukluk.png", buyukluk);

    cv::waitKey(0);
    return 0;
}