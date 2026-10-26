#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat contrastStretch(const cv::Mat& src)
{
    CV_Assert(src.type() == CV_8UC1);

    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);

    cv::Mat dst = src.clone();
    if (minVal == maxVal) return dst;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = cv::saturate_cast<uchar>(
                (src.at<uchar>(i, j) - minVal) * 255.0 / (maxVal - minVal));
        }
    }
    return dst;
}

int main()
{
    // 1) Resmi gri seviye olarak oku
    cv::Mat img = cv::imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/test.jpeg", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Resim okunamadi! Dosya yolunu kontrol et." << std::endl;
        return -1;
    }

    // 2) Fonksiyonu cagir
    cv::Mat sonuc = contrastStretch(img);

    // 3) Min/max'i yazdirip kontrol et
    double mn, mx;
    cv::minMaxLoc(img, &mn, &mx);
    std::cout << "Onceki  -> min: " << mn << "  max: " << mx << std::endl;
    cv::minMaxLoc(sonuc, &mn, &mx);
    std::cout << "Sonraki -> min: " << mn << "  max: " << mx << std::endl;

    // 4) Goster ve kaydet
    cv::imshow("Orijinal", img);
    cv::imshow("Contrast Stretch", sonuc);
    cv::imwrite("sonuc.jpg", sonuc);

    cv::waitKey(0);
    return 0;
}