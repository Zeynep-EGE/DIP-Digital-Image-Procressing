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

    cv::Mat prewitt = (cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1);

    cv::Mat sonuc32;
    cv::filter2D(yumusak, sonuc32, CV_32F, prewitt);

    cv::Mat sonuc;
    cv::convertScaleAbs(sonuc32, sonuc);

    cv::imshow("Orijinal (gri)", gri);
    cv::imshow("Gauss sonrasi", yumusak);
    cv::imshow("Prewitt (yatay kenarlar)", sonuc);
    cv::imwrite("prewitt2_sonuc.png", sonuc);

    cv::waitKey(0);
    return 0;
}